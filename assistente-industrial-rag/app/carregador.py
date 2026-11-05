"""ETAPA 1 e 2 do pipeline: ler os PDFs e dividir em chunks.

Cada chunk guarda de onde veio (arquivo e página), para que a resposta
possa mostrar a fonte exata.

Detalhe importante dos PDFs da Rocket: o texto tem marcadores como
"[Página 34]" indicando a página do manual original. Um PDF de 2 páginas
pode conter as "páginas" 1, 15, 30, 34 e 45 do manual. Por isso usamos esses
marcadores como número de página da fonte. Se o PDF não tiver marcadores,
usamos a página física do PDF.
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from configuracao import CHUNK_OVERLAP, CHUNK_SIZE, PASTA_KNOWLEDGE

# "[Página 8]" ou "(Página 34)", opcionalmente precedido do título da seção
# ("Seção 2: Requisitos de Entrada ... [Página 8]") para o título ficar junto
# do conteúdo dele.
MARCADOR = re.compile(
    r"(?:Seção\s+\d+:[^\[\]]{0,150}?)?(?:\[Página\s+(\d+)\]|\(Página\s+(\d+)\))"
)


def listar_pdfs(pasta: Path = PASTA_KNOWLEDGE) -> list[Path]:
    if not pasta.exists():
        return []
    return sorted(p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def _titulo_do_documento(texto_primeira_pagina: str, arquivo: Path) -> str:
    """Pega o título do manual (texto antes de 'Código do Documento')."""
    titulo = texto_primeira_pagina.split("Código do Documento")[0]
    titulo = titulo.replace("Rocket Industrial Solutions", "").strip(" -:")
    if not titulo or len(titulo) > 150:
        titulo = arquivo.stem.replace("_", " ")
    return titulo


def carregar_pdf(arquivo: Path) -> list[Document]:
    """Lê um PDF e devolve um Document por trecho de página, com metadados."""
    leitor = PdfReader(str(arquivo))
    if not leitor.pages:
        return []

    # O pypdf quebra linha entre quase todas as palavras; juntamos tudo.
    textos = [" ".join((p.extract_text() or "").split()) for p in leitor.pages]
    titulo = _titulo_do_documento(textos[0], arquivo)

    trechos: list[Document] = []
    pagina_atual = None  # página de referência do manual (continua entre páginas físicas)

    for numero_fisico, texto in enumerate(textos, start=1):
        if not texto:
            continue  # página sem texto (PDF escaneado precisaria de OCR)

        # Encontra onde começa cada nova página de referência dentro do texto
        cortes = []  # (posição no texto, número da página)
        for m in MARCADOR.finditer(texto):
            numero = int(m.group(1) or m.group(2))
            ultima = cortes[-1][1] if cortes else pagina_atual
            if numero != ultima:  # "(Página 12)" dentro da página 12 não corta
                cortes.append((m.start(), numero))

        # Texto antes do primeiro marcador continua a página anterior
        inicio_primeiro = cortes[0][0] if cortes else len(texto)
        pedacos = [(0, inicio_primeiro, pagina_atual or numero_fisico)]
        for i, (pos, numero) in enumerate(cortes):
            fim = cortes[i + 1][0] if i + 1 < len(cortes) else len(texto)
            pedacos.append((pos, fim, numero))
            pagina_atual = numero

        for inicio, fim, pagina in pedacos:
            conteudo = texto[inicio:fim].strip()
            if conteudo:
                trechos.append(Document(
                    page_content=conteudo,
                    metadata={
                        "fonte": arquivo.name,
                        "titulo": titulo,
                        "pagina": int(pagina),
                        "pagina_pdf": numero_fisico,
                    },
                ))
    return trechos


def dividir_em_chunks(documentos: list[Document]) -> list[Document]:
    """Quebra os trechos em chunks e coloca o título do manual em cada um.

    O título ajuda a busca e o LLM a saber de qual equipamento é o trecho
    (ex.: não misturar a CNC X200 com o compressor CP-50).
    """
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[". ", "; ", " ", ""],
        keep_separator="end",
    )
    chunks = divisor.split_documents(documentos)
    for i, chunk in enumerate(chunks):
        chunk.page_content = f"[{chunk.metadata['titulo']}] {chunk.page_content}"
        chunk.metadata["chunk"] = i
    return chunks


def carregar_e_dividir(arquivos: list[Path] | None = None) -> list[Document]:
    """Atalho: lê os PDFs (todos da pasta knowledge, se nada for passado) e divide."""
    arquivos = listar_pdfs() if arquivos is None else arquivos
    documentos = []
    for arquivo in arquivos:
        documentos.extend(carregar_pdf(arquivo))
    return dividir_em_chunks(documentos)


if __name__ == "__main__":
    # Teste rápido: python app/carregador.py
    for c in carregar_e_dividir():
        m = c.metadata
        print(f"{m['fonte']} | pág. {m['pagina']} (PDF {m['pagina_pdf']}) | {len(c.page_content)} caracteres")
        print("   ", c.page_content[:140], "...")
