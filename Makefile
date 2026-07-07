# Makefile — atalhos para rodar o AutoSeguro Agent.
# A API de cotação do desafio (quote-service) roda separada; QUOTE_SERVICE_DIR
# aponta para ela (default: repo irmão namastex-fde-challenge).
QUOTE_SERVICE_DIR ?= ../namastex-fde-challenge/quote-service

.PHONY: help install test run quote-service docker-build docker-run

help:  ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Instala as dependências (uv sync)
	uv sync

test:  ## Roda os testes (sem chave e sem rede)
	uv run pytest -q

run:  ## Roda o agente (CLI). Requer ANTHROPIC_API_KEY no .env e a /quote no ar.
	uv run python -m autoseguro.cli

quote-service:  ## Sobe a /quote do desafio SEM Docker (uv + uvicorn) em :8000
	cd $(QUOTE_SERVICE_DIR) && uv run uvicorn app.main:app --port 8000

docker-build:  ## Builda a imagem do agente
	docker build -t autoseguro-agent .

docker-run:  ## Roda o agente containerizado (passa ANTHROPIC_API_KEY do ambiente)
	docker run -it --rm -e ANTHROPIC_API_KEY \
		-e QUOTE_API_URL=http://host.docker.internal:8000 autoseguro-agent
