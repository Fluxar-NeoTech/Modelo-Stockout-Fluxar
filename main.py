import uvicorn
from app.routers.router import router
from fastapi import FastAPI

# Inicializa FastAPI
app = FastAPI(title="Fluxar IA API", version="1.1")

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)