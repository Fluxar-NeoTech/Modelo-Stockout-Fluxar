# FUNÇÕES AUXILIARES

# -----------------------------------------------------------------------
# Importando bibliotecas
import os
import psycopg2
import pandas as pd
import redis
import joblib
import io
from dotenv import load_dotenv
import numpy as np

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
        conn = psycopg2.connect(DATABASE_URL)
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
def get_fluxar_data(industria_id: int, setor_id: int) -> pd.DataFrame:
    """
    Consulta o histórico de estoque filtrando por indústria e setor.
    Retorna um DataFrame pronto para pré-processamento.
    """
    query = f"""
        SELECT 
            data,
            movimentacao,
            volume_movimentado,
            produto_id,
            unidade_id,
            setor_id,
            industria_id
        FROM historico_estoque
        WHERE industria_id = {industria_id}
          AND setor_id = {setor_id}
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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-processa os dados para o modelo: média móvel + one-hot."""
    try:
        df = df.sort_values("data")
        df["volume_movimentado"] = df["volume_movimentado"].astype(float)
        # Média móvel de 7 dias
        df["Units_Sold_Rolling7"] = df["volume_movimentado"].rolling(window=7, min_periods=1).mean()
        # One-hot encoding da coluna 'movimentacao'
        df = pd.get_dummies(df, columns=["movimentacao"], drop_first=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Erro no pré-processamento dos dados: {str(e)}")

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
