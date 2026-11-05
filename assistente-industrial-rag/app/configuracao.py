"""Configurações do projeto (lidas do arquivo .env quando ele existir)."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Pasta raiz do projeto (a pasta acima de app/)
RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

# ---------------- Pastas ----------------
PASTA_KNOWLEDGE = RAIZ / "knowledge"          # PDFs oficiais da empresa
PASTA_BASE_VETORIAL = RAIZ / "base_vetorial"  # onde o ChromaDB grava os vetores
NOME_COLECAO = "rocket_industrial"

# ---------------- LLM (quem escreve a resposta) ----------------
# "ollama" = modelo local e gratuito (padrão da aula)
# "openai" = usa a API da OpenAI (precisa de OPENAI_API_KEY no .env)
PROVEDOR_LLM = os.getenv("PROVEDOR_LLM", "ollama").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODELO_OLLAMA = os.getenv("MODELO_OLLAMA", "llama3.2")
MODELO_OPENAI = os.getenv("MODELO_OPENAI", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ---------------- Embeddings (texto -> vetor) ----------------
# Mesmo modelo multilíngue usado na aula; roda no próprio computador.
MODELO_EMBEDDINGS = os.getenv(
    "MODELO_EMBEDDINGS", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ---------------- Chunking e busca ----------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Distância de cosseno máxima para um trecho ser considerado relevante.
# 0 = idêntico, 1 = sem relação. Trechos acima disso são descartados.
DISTANCIA_MAXIMA = float(os.getenv("DISTANCIA_MAXIMA", "0.80"))

# ---------------- Nível 3: resposta obrigatória ----------------
RESPOSTA_SEM_INFO = "Não encontrei essa informação na documentação disponível."
