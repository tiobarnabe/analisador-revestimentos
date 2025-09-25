# Analisador de Revestimentos

API em FastAPI que analisa uma imagem de revestimento:
- Detecta padrão da malha (quadrado/retangular)
- Extrai as 5 cores dominantes com porcentagem
- Estima acabamento (fosco/natural ou brilhante)
- Gera uma sugestão de busca para encontrar produtos semelhantes em lojas brasileiras

## Como rodar localmente

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Depois acesse http://localhost:8000
