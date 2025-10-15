# Modelo de Previsão de Stockout – Projeto Fluxar

## Objetivo
Desenvolver um modelo de aprendizado de máquina capaz de prever o número de dias até o estoque acabar (`days_to_stockout`) para cada produto em cada loja. Essa previsão permitirá:

- Alertas antecipados para reposição de estoque.
- Otimização do gerenciamento de inventário, reduzindo perdas e rupturas.
- Base para integração futura com dashboards e chatbots analíticos do sistema Fluxar.

## Dados

**Origem:** Histórico de estoque da rede varejista.

**Principais colunas utilizadas:**

- `date`: data do registro
- `store_id`: identificação da loja
- `product_id`: identificação do produto
- `category`: categoria do produto
- `region`: região da loja
- `inventory_level`: quantidade em estoque
- `units_sold`: quantidade vendida
- `units_ordered`: quantidade pedida ao fornecedor

**Pré-processamento aplicado:**

- Conversão de datas (`datetime`) e extração de variáveis temporais (dia da semana, mês, fim de semana).
- Tratamento de valores ausentes e outliers (IQR).
- Criação de variáveis categóricas com one-hot encoding.
- Escalonamento de variáveis contínuas (MinMaxScaler e StandardScaler).
- Transformação do target com `log1p` para estabilizar variação e evitar skew.

## Modelos Testados

- LinearRegression
- Ridge
- RandomForestRegressor
- XGBRegressor

**Detalhes de treinamento:**

- Divisão treino/teste: 80/20, sem shuffle (preservando ordem temporal)
- Teste com múltiplas seeds para garantir estabilidade do resultado
- Avaliação com métricas:
  - MAPE (log-transformado) para análises internas
  - MAPE real e MAE no espaço original (`days_to_stockout`) para interpretação prática
- Observação de importância das features: `units_sold` e `inventory_level` são as variáveis mais determinantes

## Resultados

**MAPE real (em dias):**

- LinearRegression / Ridge: ~17%
- RandomForest: 0,3%
- XGBoost: 3,7%

**MAE real:**

- RandomForest: 0,01 dias
- XGBoost: 0,08 dias

**Conclusão:**  
RandomForest apresentou o melhor desempenho para previsões precisas de estoque.

- Validação detalhada: gráficos de predição vs realidade, análise por produto/loja/categoria
- Integração em produção: exportar modelos e pipeline (`joblib`/`pickle`), conectar com painel Fluxar para alertas
- Documentação final: MD/PDF com visual Fluxar, tabelas de métricas, gráficos e feature importance
- Futuras melhorias: modelos de séries temporais (Prophet, LSTM, LightGBM TS), alertas automáticos e RAG/chatbot explicativo
