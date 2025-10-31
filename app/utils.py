# FUNÇÕES AUXILIARES

# -----------------------------------------------------------------------
# Importando bibliotecas
import os
import psycopg
import pandas as pd
import redis
import joblib
import io
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# -----------------------------
# Funções de conexão
# -----------------------------
def get_conn():
    """Cria e retorna uma conexão com o banco Postgres."""
    try:
        conn = psycopg.connect(DATABASE_URL)
        return conn
    except Exception as e:
        raise ConnectionError(f"Erro ao conectar ao Postgres: {str(e)}")

def get_redis():
    """Conecta ao Redis usando a URL do .env"""
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()  # verifica conexão
        return r
    except Exception as e:
        raise ConnectionError(f"Erro ao conectar ao Redis: {str(e)}")

# -----------------------------
# Funções de dados
# -----------------------------
def get_fluxar_data(industria_id: int, setor_id: int, unidade_id: int) -> pd.DataFrame:
    """
    Consulta o histórico de estoque filtrando por indústria, setor e unidade.
    Retorna um DataFrame pronto para pré-processamento.
    """
    query = f"""
        SELECT
            data,
            movimentacao,
            volume_movimentado,
            p.nome as nome_produto,
            produto_id,
            unidade_id,
            he.setor_id,
            industria_id
        FROM historico_estoque he
        join produto p on p.id = he.produto_id
        WHERE industria_id = {industria_id}
        AND he.setor_id = {setor_id}
        AND unidade_id = {unidade_id}
        ORDER BY data;
    """
    try:
        conn = get_conn()
        df = pd.read_sql(query, conn)
        conn.close()
        if df.empty:
            raise ValueError("Nenhum dado encontrado para a indústria/setor informados.")
        return df
    except Exception as e:
        raise RuntimeError(f"Erro ao buscar dados no Postgres: {str(e)}")

def load_model_from_redis():
    """
    Carrega o modelo serializado do Redis, recombinando chunks se necessário.
    Suporta modelos maiores que o limite de 10MB do Redis remoto.
    """
    try:
        r = get_redis()
        # Tenta carregar pelo esquema antigo (chave única)
        model_bytes = r.get("modelo_fluxar_serializado")
        if model_bytes:
            return joblib.load(io.BytesIO(model_bytes))

        # Se não existir chave única, tenta o esquema dividido em partes
        num_parts_bytes = r.get("ml_model_fluxar_parts")
        if not num_parts_bytes:
            raise FileNotFoundError("Modelo não encontrado no Redis (nenhuma chave válida).")
        num_parts = int(num_parts_bytes.decode() if isinstance(num_parts_bytes, bytes) else num_parts_bytes)
        parts = [r.get(f"ml_model_fluxar_part_{i}") for i in range(num_parts)]
        if any(p is None for p in parts):
            raise RuntimeError("Alguma parte do modelo não foi encontrada no Redis.")
        model_rejoined = b"".join(parts)
        return joblib.load(io.BytesIO(model_rejoined))

    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo do Redis: {str(e)}")

import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processa os dados de estoque para previsão de days_to_stockout,
    criando as features compatíveis com o modelo RandomForest salvo no Redis.
    """
    df = df.sort_values(["produto_id", "data"])
    df["volume_movimentado"] = df["volume_movimentado"].astype(float)

    # Histórico e movimento
    df["inventory_level"] = (
        df.groupby("produto_id")["volume_movimentado"].cumsum()
    )
    df["units_sold"] = df["volume_movimentado"]
    df["units_ordered"] = (
        df.groupby("produto_id")["volume_movimentado"].shift(1).fillna(0)
    )

    # Datas
    df["dayofweek"] = df["data"].dt.dayofweek / 6  # normalizado (0 a 1)
    df["month"] = (df["data"].dt.month - 1) / 11   # normalizado (0 a 1)
    df["is_weekend"] = df["dayofweek"].isin([5/6, 1.0]).astype(int)

    # Simulação de categoria e região (placeholder, se não existirem)
    if "categoria" not in df.columns:
        df["categoria"] = np.random.choice(["Groceries", "Electronics", "Furniture", "Toys"], size=len(df))
    if "regiao" not in df.columns:
        df["regiao"] = np.random.choice(["North", "South", "West"], size=len(df))
    if "store_id" not in df.columns:
        df["store_id"] = np.random.choice(["S002", "S003", "S004", "S005"], size=len(df))

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=["categoria", "regiao", "store_id", "produto_id"],
        prefix=["category", "region", "store_id", "product_id"],
        drop_first=False
    )

    # Reordena colunas principais no início (boa prática)
    base_cols = [
        "inventory_level", "units_sold", "units_ordered",
        "dayofweek", "month", "is_weekend"
    ]
    others = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + others]

    return df


def align_features(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """
    Garante que o dataframe tenha as mesmas features que o modelo espera.
    Adiciona colunas faltantes com 0 e remove colunas extras.
    """
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    # Remove colunas extras que o modelo não viu
    extra_cols = [c for c in df.columns if c not in model_features]
    if extra_cols:
        df = df.drop(columns=extra_cols)
    return df[model_features]