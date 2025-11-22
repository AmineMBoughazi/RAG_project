POETRY := $(shell command -v poetry 2>/dev/null)

.PHONY: help install-poetry setup init-django init-fastapi init-infra init-lambdas show-logic

help:
	@echo "Commandes disponibles :"
	@echo "  make install-poetry    -> installe Poetry si absent"
	@echo "  make setup             -> installe les deps de tous les sous-projets"
	@echo "  make show-logic        -> affiche la logique du projet"

install-poetry:
ifeq ($(POETRY),)
	@echo ">> Poetry introuvable, installation via pipx..."
	@python3 -m pip install --user pipx
	@python3 -m pipx ensurepath
	@python3 -m pipx install poetry
	@echo ">> Poetry installé."
	@echo ">> IMPORTANT : rouvre ton terminal ou lance : exec $$SHELL"
else
	@echo ">> Poetry déjà installé : $(POETRY)"
endif

setup: install-poetry init-django init-fastapi init-infra init-lambdas
	@echo ">> Setup terminé."

init-django:
	@echo ">> Django_app_front : installation des dépendances..."
	@cd Django_app_front && poetry install

init-fastapi:
	@echo ">> Fast_API_back : installation des dépendances..."
	@cd Fast_API_back && poetry install

init-infra:
	@echo ">> infra : installation des dépendances..."
	@cd infra && poetry install

init-lambdas:
	@echo ">> Lambdas_functions : installation des dépendances..."
	@cd Lambdas_functions && poetry install

show-logic:
	@echo "LOGIQUE DU PROJET :"
	@echo ""
	@echo "  Django_app_front     -> UI Django (upload + chat) déployée sur Elastic Beanstalk"
	@echo "  Fast_API_back        -> API FastAPI (RAG) qui appelle Bedrock / OpenSearch / DynamoDB"
	@echo "  Lambdas_functions    -> Fonctions AWS Lambda triggered par S3 (ingestion/embeddings)"
	@echo "  infra                -> CDK (infra IaC, déploiement ressources AWS)"
	@echo ""
	@echo "Flow global :"
	@echo "  Upload -> S3 -> Lambda ingestion -> indexing OpenSearch"
	@echo "  Chat -> FastAPI -> RAG -> réponse"
