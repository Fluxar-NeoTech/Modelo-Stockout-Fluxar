# Documentação da API Fluxar

<div style="background:#F7F3FF; font-family: Poppins, Segoe UI; border-left:8px solid #FFA726; padding:18px; border-radius:10px; margin-top:24px; width: 97%; color:#4B0055">

## Estrutura do Projeto

Projeto está organizado da seguinte forma:  
```
fluxar_api/  
│  
├─ main.py # código principal da API  
├─ .env # variáveis de ambiente (Redis, Postgres)  
├─ requirements.txt # dependências  
├─ routers/ # diretório com os endpoints e utils  
│ ├─ router.py # rotas principais (predict)  
│ └─ utils.py # funções auxiliares  
└─ models/  
└─ predict_request.py # modelo Pydantic para o corpo da requisição  
```

---

## Variáveis de ambiente (.env)

Primeiramente, para você professor rodar na sua máquina, crie um arquivo `.env` na raiz do projeto com as credenciais do banco e Redis:
```
DATABASE_URL=postgresql://usuario:senha@localhost:5432/fluxar_db
REDIS_URL=redis://localhost:6379/0
```

---
## Dependências (requirements.txt)
```
fastapi==0.111.0
uvicorn[standard]==0.25.0
psycopg2-binary==2.9.9
pandas==2.1.1
numpy==1.26.0
redis==5.3.0
joblib==1.3.2
pydantic==2.6.1
scikit-learn==1.3.3
python-dotenv==1.1.0
```
Rode usando <code> pip install -r requirements.txt </code>

---
## Testando o Endpoint

Requisição de exemplo:

<code>POST http://127.0.0.1:8000/predict</code>  
<code>Content-Type: application/json</code>

{  
"industria_id": 1,  
"setor_id": 2  
}  


### Resposta esperada
{  
"predictions": [  
{  
"data": "2025-10-15 00:00:00",  
"produto_id": 1,  
"unidade_id": 1,  
"days_to_stockout_pred": 5.3  
},  
...  
]  
}  

> Essa resposta traz as últimas previsões de `days_to_stockout` para os produtos mais recentes do setor e indústria.

---
## Lógica da API

### 1. models/predict_request.py
Define o modelo Pydantic que valida a entrada JSON:
```python
from pydantic import BaseModel

class PredictRequest(BaseModel):
    industria_id: int
    setor_id: int
```

### 2. routers/utils.py

Contém funções auxiliares:
```
get_conn() → conecta ao Postgres

get_redis() → conecta ao Redis

get_fluxar_data() → busca dados históricos filtrando indústria e setor

preprocess_data() → gera média móvel, faz one-hot e organiza features

load_model_from_redis() → recupera o modelo em memória

align_features() → garante compatibilidade entre colunas do modelo e do dataset
```

### 3. routers/router.py

Endpoint principal da API:
```
router = APIRouter(prefix="/predict")

@router.post("")
def predict(request: PredictRequest):
    """
    Endpoint que recebe indústria e setor,
    retorna previsões de days_to_stockout para os produtos.
    """
    ...
```

### 4. main.py
```
import uvicorn
from app.routers.router import router
from fastapi import FastAPI

app = FastAPI(title="Fluxar IA API", version="1.1")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### 5. Dockerfile
```
FROM python:3.11.2

RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

---
## Como funciona a previsão
O usuário envia <code>industria_id</code> e <code>setor_id</code>.

- A API busca os dados históricos no Postgres.
- O preprocess_data() aplica média móvel e encoding.
- O modelo salvo no Redis é carregado via joblib.
- A API seleciona a data mais recente por produto e prediz days_to_stockout.
- A resposta é formatada em JSON e enviada ao cliente.
- O back-end pode filtrar produtos com days_to_stockout_pred < 5 para enviar notificações.
