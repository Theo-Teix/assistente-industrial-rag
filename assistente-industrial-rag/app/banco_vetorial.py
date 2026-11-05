"""ETAPA 3 e 4 do pipeline: embeddings + banco vetorial (ChromaDB via LangChain)."""

import hashlib
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from carregador import carregar_e_dividir, listar_pdfs
from configuracao import MODELO_EMBEDDINGS, NOME_COLECAO, PASTA_BASE_VETORIAL

_embeddings = None


def obter_embeddings():
    """Carrega o modelo de embeddings uma vez só (na 1ª vez ele é baixado)."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=MODELO_EMBEDDINGS,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def definir_embeddings(embeddings):
    """Permite trocar o modelo de embeddings (usado nos testes)."""
    global _embeddings
    _embeddings = embeddings


def abrir_banco() -> Chroma:
    return Chroma(
        collection_name=NOME_COLECAO,
        embedding_function=obter_embeddings(),
        persist_directory=str(PASTA_BASE_VETORIAL),
        collection_metadata={"hnsw:space": "cosine"},  # distância de cosseno
    )


def _id_do_chunk(chunk: Document) -> str:
    """ID fixo: indexar o mesmo trecho de novo não cria duplicata."""
    m = chunk.metadata
    chave = f"{m['fonte']}|{m['pagina']}|{m.get('chunk')}|{chunk.page_content}"
    return hashlib.sha1(chave.encode("utf-8")).hexdigest()


def indexar(arquivos: list[Path] | None = None, recriar: bool = False) -> int:
    """Lê PDFs, gera embeddings e grava no ChromaDB. Devolve nº de chunks.

    recriar=True apaga a base e indexa tudo de novo (todos os PDFs de knowledge/).
    recriar=False só adiciona os arquivos informados (upload de novo PDF).
    """
    if recriar:
        try:
            abrir_banco().delete_collection()
        except Exception:
            shutil.rmtree(PASTA_BASE_VETORIAL, ignore_errors=True)

    chunks = carregar_e_dividir(arquivos)
    if not chunks:
        return 0

    banco = abrir_banco()
    if arquivos:  # remove versões antigas desses arquivos antes de adicionar
        for arquivo in arquivos:
            banco.delete(where={"fonte": arquivo.name})
    banco.add_documents(chunks, ids=[_id_do_chunk(c) for c in chunks])
    return len(chunks)


def total_de_chunks() -> int:
    try:
        return abrir_banco()._collection.count()
    except Exception:
        return 0


def documentos_indexados() -> list[str]:
    try:
        dados = abrir_banco().get(include=["metadatas"])
        return sorted({m["fonte"] for m in dados["metadatas"]})
    except Exception:
        return []


def precisa_indexar() -> bool:
    """True se há PDF em knowledge/ que ainda não está no banco."""
    na_pasta = {p.name for p in listar_pdfs()}
    return bool(na_pasta - set(documentos_indexados()))


def texto_dos_documentos(fontes: set[str]) -> str:
    """Todo o texto indexado dos PDFs informados (usado no controle de alucinação)."""
    if not fontes:
        return ""
    dados = abrir_banco().get(where={"fonte": {"$in": sorted(fontes)}}, include=["documents"])
    return " ".join(dados["documents"])


def buscar(pergunta: str, k: int) -> list[tuple[Document, float]]:
    """Busca semântica: devolve (chunk, distância). Menor distância = mais parecido."""
    if total_de_chunks() == 0:
        return []
    return abrir_banco().similarity_search_with_score(pergunta, k=k)


if __name__ == "__main__":
    # Indexa pela linha de comando: python app/banco_vetorial.py
    n = indexar(recriar=True)
    print(f"Base criada: {len(documentos_indexados())} PDFs, {n} chunks em {PASTA_BASE_VETORIAL}")
