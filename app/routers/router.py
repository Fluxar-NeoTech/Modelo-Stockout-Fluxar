from app.models.predict_request import PredictRequest 
from fastapi import APIRouter, HTTPException, status, Body
from fastapi.responses import JSONResponse
import numpy as np
from app.utils import get_fluxar_data, load_model_from_redis, preprocess_data, align_features

router = APIRouter(prefix="/predict")

@router.post("")
def predict(request: PredictRequest = Body(...)):
    """
    Endpoint que recebe indústria, unidade e setor,
    retorna previsões de days_to_stockout para os produtos.
    """
    try:
        # 1️) Consulta os dados filtrando indústria e setor
        df = get_fluxar_data(request.industria_id, request.setor_id, request.unidade_id)
        if df.empty:
            raise HTTPException(status_code=404, detail="Nenhum dado encontrado para essa indústria/setor/unidade.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao consultar dados: {str(e)}")

    try:
        # Pré-processa os dados
        df = preprocess_data(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no pré-processamento dos dados: {str(e)}")

    try:
        # Carrega o modelo do Redis
        model = load_model_from_redis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo do Redis: {str(e)}")

    try:
        # Alinha as features com o que o modelo espera
        features = [col for col in df.columns if col not in ["data", "days_to_stockout"]]
        X = align_features(df[features].fillna(0), model.feature_names_in_)

        # Teste 
        print("Features usadas pelo modelo:", model.feature_names_in_)
        print("Primeiras linhas do X:\n", X.head())


        # Faz a predição e desfaz o log1p
        preds = model.predict(X)
        df["days_to_stockout_pred"] = np.expm1(preds)

        # 6️) Retorna os últimos 10 registros com previsões
        result = df[["data", "nome_produto", "produto_id", "unidade_id", "industria_id", "days_to_stockout_pred"]].tail(10).to_dict(orient="records")
        # Pega apenas a última previsão por produto
        df_last = df.sort_values("data").groupby("produto_id").tail(1)[
            ["data", "nome_produto", "produto_id", "unidade_id", "days_to_stockout_pred"]
        ].copy()

        # Converte Timestamp para string
        df_last["data"] = df_last["data"].dt.strftime("%Y-%m-%d %H:%M:%S")

        result = df_last.to_dict(orient="records")

        return JSONResponse(
            content={"predictions": result},
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar previsões: {str(e)}")
