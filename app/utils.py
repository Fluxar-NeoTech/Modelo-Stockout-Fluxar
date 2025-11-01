import os
import io
import redis
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ------------------------------------------
# Carregar variáveis de ambiente
# ------------------------------------------
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")
DB_URL = os.getenv("DATABASE_URL")

if not REDIS_URL or not DB_URL:
    raise ValueError("Variáveis REDIS_URL ou DATABASE_URL não encontradas no .env")

# ------------------------------------------
# Conexão Redis
# ------------------------------------------
parsed = urlparse(REDIS_URL)
r = redis.Redis(
    host=parsed.hostname,
    port=parsed.port,
    username=parsed.username,
    password=parsed.password,
    ssl=parsed.scheme in ["rediss", "redis+ssl"],
    decode_responses=False
)

# ------------------------------------------
# Conexão PostgreSQL
# ------------------------------------------
engine = create_engine(DB_URL)

# ------------------------------------------
# Função para obter dados do Fluxar
# ------------------------------------------
def get_fluxar_data(industria_id: int, setor_id: int, unidade_id: int):
    """
    Retorna dataframe com histórico de estoque filtrado por indústria, setor e unidade.
    Inclui:
      - Dados de vendas e pedidos
      - Capacidade total ocupada
      - Nome e categoria do produto
    """
    query = f"""
        SELECT 
            h.data,
            h.produto_id,
            h.unidade_id,
            p.nome AS produto_nome,
            p.tipo AS category,
            CASE WHEN h.movimentacao = 'S' THEN h.volume_movimentado ELSE 0 END AS units_sold,
            CASE WHEN h.movimentacao = 'E' THEN h.volume_movimentado ELSE 0 END AS units_ordered,
            hc.capacidade_total_ocupada AS inventory_level
        FROM historico_estoque h
        JOIN produto p 
            ON h.produto_id = p.id
        LEFT JOIN historico_capacidade hc
            ON hc.produto_id = h.produto_id
           AND hc.unidade_id = h.unidade_id
           AND hc.industria_id = h.industria_id
           AND hc.data_completa = h.data
        WHERE h.industria_id = {industria_id}
          AND h.setor_id = {setor_id}
          AND h.unidade_id = {unidade_id}
        ORDER BY h.produto_id, h.unidade_id, h.data
    """
    df = pd.read_sql(query, engine)

    # Conversão e ordenação
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    df = df.dropna(subset=['data'])
    df = df.sort_values(['produto_id', 'unidade_id', 'data']).reset_index(drop=True)

    return df


# ------------------------------------------
# Função para carregar modelo do Redis
# ------------------------------------------
def load_model_from_redis():
    """
    Carrega modelo RandomForest do Redis já salvo em chunks.
    """
    parts = int(r.get("ml_model_fluxar_parts"))
    data = b"".join([r.get(f"ml_model_fluxar_part_{i}") for i in range(parts)])
    model = joblib.load(io.BytesIO(data))
    return model

# ------------------------------------------
# Função de pré-processamento
# ------------------------------------------
def preprocess_data(df: pd.DataFrame, model=None):
    """
    Pré-processamento para API:
    - Cria features de tempo
    - Preenche nulos
    - Gera dummies
    - Normaliza contínuas usando limites do treino
    - Garante colunas esperadas pelo modelo
    """
    df = df.copy()

    # Features temporais
    df['dayofweek'] = df['data'].dt.dayofweek
    df['month'] = df['data'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Preencher nulos
    for col in ['inventory_level', 'units_sold', 'units_ordered']:
        df[col] = df[col].fillna(df[col].median())
    df['category'] = df['category'].fillna("desconhecido")

    # One-hot encoding
    df = pd.get_dummies(df, columns=['category'], drop_first=False)

    # Normalização manual MinMax com limites do treino
    min_max_stats = {
        'inventory_level': (0, 1000),
        'units_sold': (0, 500),
        'units_ordered': (0, 500),
        'dayofweek': (0, 6),
        'month': (1, 12),
        'is_weekend': (0, 1)
    }
    for col, (min_val, max_val) in min_max_stats.items():
        if col in df.columns:
            df[col] = (df[col] - min_val) / (max_val - min_val)

    # Garantir colunas esperadas pelo modelo
    if model:
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_names_in_]

    return df
