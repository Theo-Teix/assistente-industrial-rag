"""ETAPA 5: o motor RAG (busca + prompt + LLM) com controle de alucinação.

Fluxo de uma pergunta:
  1. Se existe histórico, o LLM reescreve a pergunta para ela fazer sentido
     sozinha ("E esse óleo?" -> "Qual a frequência de troca do óleo do compressor?").
  2. Busca semântica no ChromaDB (top K trechos mais próximos).
  3. Nível 3 - três travas contra invenção ANTES e DEPOIS do LLM:
       a) nenhum trecho perto o bastante da pergunta  -> resposta padrão;
       b) a pergunta cita um código (Y500, E099...) que não existe nos
          documentos encontrados                      -> resposta padrão;
       c) o LLM disse que não encontrou               -> resposta padrão EXATA.
  4. O LLM responde usando só os trechos; o programa anexa as fontes.
"""

import re
import unicodedata

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from banco_vetorial import buscar, texto_dos_documentos
from configuracao import (
    DISTANCIA_MAXIMA, MODELO_OLLAMA, MODELO_OPENAI, OLLAMA_URL,
    OPENAI_API_KEY, PROVEDOR_LLM, RESPOSTA_SEM_INFO, TOP_K,
)

# ---------------------------------------------------------------- prompts

PROMPT_RESPOSTA = ChatPromptTemplate.from_messages([
    ("system",
     "Você é o Assistente Inteligente de Suporte Técnico Industrial da Rocket Industrial Solutions.\n"
     "Responda SOMENTE com informações que estão escritas nos TRECHOS DA DOCUMENTAÇÃO enviados.\n\n"
     "REGRAS OBRIGATÓRIAS:\n"
     "1. Não use conhecimento próprio. Não deduza, não suponha e não invente valores, passos ou recomendações.\n"
     "2. Se os trechos não trazem a resposta, responda exatamente: {sem_info}\n"
     "   Nesse caso não escreva mais nenhuma palavra.\n"
     "3. Não misture equipamentos: o procedimento de uma máquina ou modelo não vale para outro.\n"
     "4. Para procedimentos, use passos numerados (1., 2., 3.) na mesma ordem do documento.\n"
     "5. Quando houver resposta, comece com \"De acordo com o <nome do documento>,\".\n"
     "6. Não escreva nome de arquivo nem página no final: o sistema adiciona as fontes.\n"
     "7. Responda em português do Brasil, de forma direta e técnica."),
    ("human", "TRECHOS DA DOCUMENTAÇÃO:\n{contexto}\n\nPERGUNTA: {pergunta}"),
])

PROMPT_REESCRITA = ChatPromptTemplate.from_messages([
    ("system",
     "Reescreva a ÚLTIMA pergunta do usuário para que ela possa ser entendida sozinha, sem o histórico.\n"
     "Use o histórico apenas para trocar referências como \"esse óleo\", \"dele\", \"essa máquina\" "
     "pelo nome do que está sendo falado.\n"
     "Não responda à pergunta e não acrescente informações. "
     "Se ela já puder ser entendida sozinha, devolva-a igual.\n"
     "Devolva apenas a pergunta reescrita, em uma linha."),
    ("human", "HISTÓRICO:\n{historico}\n\nÚLTIMA PERGUNTA: {pergunta}\n\nPERGUNTA REESCRITA:"),
])

# ---------------------------------------------------------------- LLM


def ollama_ativo() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


def modelos_ollama() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError):
        return []


def obter_llm(modelo: str | None = None):
    """Cria o LLM com temperatura 0 (respostas estáveis, sem 'criatividade')."""
    if PROVEDOR_LLM == "openai":
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY:
            raise RuntimeError("Configure OPENAI_API_KEY no arquivo .env para usar a OpenAI.")
        return ChatOpenAI(model=modelo or MODELO_OPENAI, api_key=OPENAI_API_KEY, temperature=0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=modelo or MODELO_OLLAMA, base_url=OLLAMA_URL, temperature=0)

# ---------------------------------------------------------------- utilitários


def _sem_acento(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.casefold())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


# Códigos de equipamento/erro: X200, E045, Y500, CP-50, NR-12, BT40...
CODIGO = re.compile(r"\b[A-Za-z]{1,4}-?\d{2,}[A-Za-z]?\b")


def _codigos(texto: str) -> set[str]:
    return {c.replace("-", "").upper() for c in CODIGO.findall(texto)}


STOPWORDS = set("""a o e as os um uma de do da dos das em no na nos nas por para com sem
que se ao aos ou mais deve ser sao qual quais como esta este essa esse isso
pelo pela sua seu suas seus entre sobre apos antes ate quando onde voce acordo""".split())


def _palavras(texto: str) -> set[str]:
    return {p for p in re.findall(r"[a-z0-9]+", _sem_acento(texto))
            if p not in STOPWORDS and (len(p) > 2 or p.isdigit())}


def _formatar_contexto(trechos) -> str:
    blocos = []
    for i, (doc, _dist) in enumerate(trechos, start=1):
        m = doc.metadata
        blocos.append(f"[Trecho {i}] Arquivo: {m['fonte']} — Página {m['pagina']}\n{doc.page_content}")
    return "\n\n".join(blocos)


def escolher_fontes(resposta: str, trechos) -> list[dict]:
    """Descobre quais trechos a resposta realmente usou.

    Compara as palavras da resposta com as de cada trecho: fica o trecho com
    mais palavras em comum e os que chegam perto dele. Assim a fonte exibida é
    a página de onde o conteúdo saiu, e não todos os trechos recuperados.
    """
    palavras_resp = _palavras(resposta)
    if not palavras_resp or not trechos:
        return []
    notas = []
    for doc, _ in trechos:
        # ignora o "[título do manual]" colocado no começo de todo chunk
        conteudo = doc.page_content.removeprefix(f"[{doc.metadata.get('titulo', '')}] ")
        notas.append((len(palavras_resp & _palavras(conteudo)), doc))
    melhor = max(n for n, _ in notas)
    if melhor == 0:
        return []
    fontes, vistos = [], set()
    for nota, doc in sorted(notas, key=lambda x: -x[0]):
        chave = (doc.metadata["fonte"], doc.metadata["pagina"])
        if nota >= 0.6 * melhor and chave not in vistos:
            vistos.add(chave)
            fontes.append({"fonte": chave[0], "pagina": chave[1]})
    return fontes


def _sem_informacao(trechos, pergunta_busca=None, motivo="") -> dict:
    return {"resposta": RESPOSTA_SEM_INFO, "fontes": [], "trechos": trechos or [],
            "pergunta_busca": pergunta_busca, "motivo": motivo}

# ---------------------------------------------------------------- motor


def reescrever_pergunta(pergunta: str, historico: list[dict], llm) -> str:
    """Usa o histórico só para resolver referências ("esse óleo", "dele")."""
    if not historico:
        return pergunta
    linhas = []
    for msg in historico[-6:]:  # últimas 3 trocas de mensagens
        quem = "Usuário" if msg["role"] == "user" else "Assistente"
        linhas.append(f"{quem}: {msg['content']}")
    cadeia = PROMPT_REESCRITA | llm | StrOutputParser()
    nova = cadeia.invoke({"historico": "\n".join(linhas), "pergunta": pergunta})
    nova = nova.strip().splitlines()[0].strip(" \"'") if nova.strip() else ""
    return nova if 5 < len(nova) < 300 else pergunta


def responder(pergunta: str, historico: list[dict] | None = None,
              modelo: str | None = None, top_k: int = TOP_K, llm=None) -> dict:
    """Responde uma pergunta. Devolve dict com resposta, fontes e trechos usados."""
    llm = llm or obter_llm(modelo)

    # 1) Histórico: pergunta independente para a busca
    pergunta_busca = reescrever_pergunta(pergunta, historico or [], llm)

    # 2) Busca semântica
    trechos = buscar(pergunta_busca, k=top_k)
    relevantes = [(d, dist) for d, dist in trechos if dist <= DISTANCIA_MAXIMA]

    # 3a) Nada parecido na base
    if not relevantes:
        return _sem_informacao(trechos, pergunta_busca, "nenhum trecho relevante")

    # 3b) Código citado que não existe nos documentos encontrados
    pedidos = _codigos(pergunta) | _codigos(pergunta_busca)
    if pedidos:
        fontes_encontradas = {d.metadata["fonte"] for d, _ in relevantes}
        existentes = _codigos(texto_dos_documentos(fontes_encontradas))
        if not pedidos <= existentes:
            faltando = ", ".join(sorted(pedidos - existentes))
            return _sem_informacao(relevantes, pergunta_busca, f"código não encontrado: {faltando}")

    # 4) Geração com LCEL: prompt | llm | parser
    cadeia = PROMPT_RESPOSTA | llm | StrOutputParser()
    resposta = cadeia.invoke({
        "contexto": _formatar_contexto(relevantes),
        "pergunta": pergunta_busca,
        "sem_info": RESPOSTA_SEM_INFO,
    }).strip()
    # Se o modelo escreveu fontes por conta própria, removemos (o sistema adiciona as corretas)
    resposta = "\n".join(linha for linha in resposta.splitlines()
                         if not re.match(r"\s*(📄|\**fontes?\**\s*:)", linha, re.IGNORECASE)).strip()

    # 3c) LLM disse que não sabe (ou devolveu vazio) -> frase EXATA, sem fontes
    if not resposta or "nao encontrei" in _sem_acento(resposta):
        return _sem_informacao(relevantes, pergunta_busca, "LLM não encontrou nos trechos")

    fontes = escolher_fontes(resposta, relevantes)
    if not fontes:
        return _sem_informacao(relevantes, pergunta_busca, "resposta sem apoio nos trechos")

    return {"resposta": resposta, "fontes": fontes, "trechos": relevantes,
            "pergunta_busca": pergunta_busca, "motivo": ""}


def formatar_fontes(fontes: list[dict]) -> str:
    return "\n".join(f"📄 Fonte: {f['fonte']} — Página {f['pagina']}" for f in fontes)
