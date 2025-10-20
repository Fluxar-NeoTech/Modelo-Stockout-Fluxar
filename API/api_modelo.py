from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import pandas as pd
import redis
import joblib
import io
import os
import numpy as np

# =============================
# Configurações
# =============================
app = FastAPI(title="Fluxar IA API", version="1.0")

DATABASE_URL = os.getenv("DATABASE_URL")  # Ex: "postgresql://user:password@host:port/db"
REDIS_URL = os.getenv("REDIS_URL")        # Ex: "redis://default:senha@host:port"

# =============================
# Conexões
# =============================
def get_conn():
    return psycopg2.connect(DATABASE_URL)

def get_redis():
    return redis.from_url(REDIS_URL)

# =============================
# Pydantic model
# =============================
class PredictRequest(BaseModel):
    industria_id: int
    setor_id: int

# =============================
# Funções auxiliares
# =============================
def get_fluxar_data(industria_id: int, setor_id: int) -> pd.DataFrame:
    """Extrai dados do histórico filtrando por indústria e setor."""
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
    conn = get_conn()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_model_from_redis():
    """Carrega o modelo serializado do Redis."""
    r = get_redis()
    model_bytes = r.get("modelo_fluxar_serializado")
    if not model_bytes:
        raise HTTPException(status_code=404, detail="Modelo não encontrado no Redis.")
    return joblib.load(io.BytesIO(model_bytes))

# =============================
# Endpoint principal
# =============================
@app.post("/predict")
def predict(request: PredictRequest):
    df = get_fluxar_data(request.industria_id, request.setor_id)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Nenhum dado encontrado para os filtros informados.")

    # Pré-processamento (exemplo compatível com seu pipeline)
    df["volume_movimentado"] = df["volume_movimentado"].astype(float)
    df = df.sort_values("data")

    # Criar feature da média móvel (igual ao modelo original)
    df["Units_Sold_Rolling7"] = df["volume_movimentado"].rolling(window=7, min_periods=1).mean()
    
    # One-hot encoding de 'movimentacao'
    df = pd.get_dummies(df, columns=["movimentacao"], drop_first=True)

    # Carrega modelo
    model = load_model_from_redis()

    # Selecionar colunas numéricas compatíveis com o modelo treinado
    features = [col for col in df.columns if col not in ["data", "days_to_stockout"]]
    X = df[features].fillna(0)

    # Predição
    preds = model.predict(X)
    preds_original = np.expm1(preds)  # Desfaz o log1p do seu modelo

    df["days_to_stockout_pred"] = preds_original

    # Retornar últimos valores (para visualização)
    result = df[["data", "produto_id", "unidade_id", "days_to_stockout_pred"]].tail(10).to_dict(orient="records")
    return {"status": "ok", "predictions": result}
