from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.models.predict_request import PredictRequest
from app.utils import get_fluxar_data, load_model_from_redis, preprocess_data
import numpy as np
import pandas as pd

router = APIRouter(prefix="/predict")

@router.post("")
def predict(request: PredictRequest):
    """
    Retorna uma previsão de days_to_stockout por produto_id, considerando a data mais recente de cada produto.
    Reproduz o pré-processamento do modelo sem alterar o pipeline salvo no Redis.
    """
    try:
        # ========================
        # 1. Buscar dados filtrados
        # ========================
        df = get_fluxar_data(
            industria_id=request.industria_id,
            setor_id=request.setor_id,
            unidade_id=request.unidade_id
        )

        if df.empty:
            raise HTTPException(status_code=404, detail="Nenhum dado encontrado para os filtros informados")

        # ========================
        # 2. Carregar modelo salvo
        # ========================
        model = load_model_from_redis()

        # ========================
        # 3. Pré-processar dados
        # ========================
        df_proc = preprocess_data(df, model=model)

        # ========================
        # 4. Gerar previsões
        # ========================
        preds = model.predict(df_proc)
        df['days_to_stockout_pred'] = np.expm1(preds)

        # ========================
        # 5. Converter e normalizar datas
        # ========================
        df['data'] = pd.to_datetime(df['data'], errors='coerce')

        # ========================
        # 6. Selecionar apenas a data mais recente de cada produto
        # ========================
        # Para cada produto_id e unidade_id, pega a linha cuja 'data' é a mais recente
        df_recent = df.loc[df.groupby(['produto_id', 'unidade_id'])['data'].idxmax()]

        # ========================
        # 7. Garantir tipos corretos e preparar saída
        # ========================
        df_recent['days_to_stockout_pred'] = df_recent['days_to_stockout_pred'].astype(float)
        df_recent['data'] = df_recent['data'].dt.strftime('%Y-%m-%d')

        output = df_recent[['data', 'produto_nome', 'produto_id', 'unidade_id', 'days_to_stockout_pred']].to_dict(orient='records')

        return JSONResponse(content={"predictions": output}, status_code=status.HTTP_200_OK)

    except HTTPException as http_error:
        raise http_error

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
