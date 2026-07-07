# Imagem do AGENTE (CLI). A API de cotação do desafio (quote-service) roda
# separada — ver README. A chave e a URL vêm por ambiente em runtime; NUNCA
# são cozidas na imagem.
FROM python:3.12-slim

WORKDIR /app

# Só as dependências de runtime do CLI (anthropic + httpx + python-dotenv).
# pandas/pyarrow existem apenas para o replay opcional sobre o dataset — ficam
# fora da imagem para mantê-la enxuta.
RUN pip install --no-cache-dir anthropic httpx python-dotenv

COPY autoseguro ./autoseguro

# Uso:
#   docker build -t autoseguro-agent .
#   docker run -it --rm \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     -e QUOTE_API_URL=http://host.docker.internal:8000 \
#     autoseguro-agent
ENTRYPOINT ["python", "-m", "autoseguro.cli"]
