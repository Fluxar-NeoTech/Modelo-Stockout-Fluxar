from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from utils import get_fluxar_data, load_model_from_redis, preprocess_data

# Inicializa FastAPI
app = FastAPI(title="Fluxar IA API", version="1.1")

# Modelo de entrada (JSON)
class PredictRequest(BaseModel):
    industria_id: int
    setor_id: int

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Endpoint que recebe indústria e setor,
    retorna previsões de days_to_stockout para os produtos.
    """
    try:
        # 1️) Consulta os dados filtrando indústria e setor
        df = get_fluxar_data(request.industria_id, request.setor_id)
        if df.empty:
            raise HTTPException(status_code=404, detail="Nenhum dado encontrado para essa indústria/setor.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao consultar dados: {str(e)}")

    try:
        # 2️) Pré-processa os dados
        df = preprocess_data(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no pré-processamento dos dados: {str(e)}")

    try:
        # 3️) Carrega o modelo do Redis
        model = load_model_from_redis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo do Redis: {str(e)}")

    try:
        # 4️) Seleciona features compatíveis com o modelo
        features = [col for col in df.columns if col not in ["data", "days_to_stockout"]]
        X = df[features].fillna(0)

        # 5️) Faz a predição e desfaz o log1p
        preds = model.predict(X)
        df["days_to_stockout_pred"] = np.expm1(preds)

        # 6️) Retorna os últimos 10 registros com previsões
        result = df[["data", "produto_id", "unidade_id", "days_to_stockout_pred"]].tail(10).to_dict(orient="records")
        return {"status": "ok", "predictions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar previsões: {str(e)}")
