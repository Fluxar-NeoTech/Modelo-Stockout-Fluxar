from pydantic import BaseModel

# Modelo de entrada (JSON)
class PredictRequest(BaseModel):
    industria_id: int
    setor_id: int