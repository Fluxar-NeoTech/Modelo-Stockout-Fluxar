<div style="background:#F7F3FF; font-family: Poppins, Segoe UI; border-left:8px solid #FFA726; padding:18px; border-radius:10px; margin-top:24px; width: 97%; color:#4B0055">

# Tutorial: Rodando a API Fluxar Localmente

Este guia passo a passo ensina como configurar e executar a API Fluxar em sua máquina local.

---

## 1. Estrutura do projeto

Organize seu projeto da seguinte forma:

```
fluxar_api/
│
├─ main.py           # código principal da API
├─ .env              # variáveis de ambiente (Redis, Postgres)
├─ requirements.txt  # dependências
├─ utils.py          # funções auxiliares (conexão banco, Redis, pré-processamento)
└─ models/           # se quiser colocar modelos Pydantic ou schemas separados
```

---

## 2. Variáveis de ambiente (.env)

Crie um arquivo `.env` na raiz do projeto com as credenciais do banco e Redis:

```
DATABASE_URL=postgresql://usuario:senha@localhost:5432/fluxar_db
REDIS_URL=redis://localhost:6379/0
```

> O `/0` indica que está usando o **database 0** do Redis.

---

## 3. Arquivo de dependências (requirements.txt)

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

> Certifique-se de instalar todas essas dependências antes de rodar a API.

---

## 4. Inicializando a API

1. Instale as dependências:

```
pip install -r requirements.txt
```

2. Rode o servidor FastAPI localmente:

```
uvicorn main:app --reload
```

> A flag `--reload` permite que a API reinicie automaticamente ao salvar alterações no código.

---

## 5. Testando o endpoint

Use um cliente HTTP como **Postman** ou o navegador para enviar requisições POST:

```
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "industria_id": 1,
  "setor_id": 2
}
```

### Resposta esperada:

```
{
  "status": "ok",
  "predictions": [
    {"data": "2025-10-15", "produto_id": 1, "unidade_id": 1, "days_to_stockout_pred": 5.3},
    ...
  ]
}
```

> Essa resposta traz as últimas 10 previsões de `days_to_stockout` para os produtos do setor e indústria selecionados.

---

## 6. Estrutura da API

* **FastAPI:** Framework que cria a API rapidamente.
* **Pydantic:** Valida os dados de entrada (JSON).
* **Redis + Joblib:** Carrega o modelo serializado em memória.
* **Pandas + Numpy:** Pré-processa os dados e cria features.
* **Endpoints:** Recebem JSON, processam dados, e retornam previsões.

---

## 7. Dicas de troubleshooting

1. Verifique se o Redis está rodando localmente:

```
redis-cli ping
```

Deve retornar `PONG`.

2. Certifique-se de que o Postgres está acessível e o banco `fluxar_db` existe.
3. Caso a API não encontre dados, o endpoint retornará erro 404.
4. Sempre mantenha o modelo serializado no Redis com a chave `modelo_fluxar_serializado`.

---

Pronto! Agora você consegue rodar e testar a API Fluxar localmente de forma organizada e com todas as configurações corretas.

</div>
