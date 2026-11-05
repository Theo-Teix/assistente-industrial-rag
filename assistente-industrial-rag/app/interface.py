"""Interface do assistente (Streamlit). Rode com: streamlit run app/interface.py"""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pypdf import PdfReader  # noqa: E402

from banco_vetorial import (documentos_indexados, indexar, precisa_indexar,  # noqa: E402
                            total_de_chunks)
from carregador import listar_pdfs  # noqa: E402
from configuracao import (MODELO_OLLAMA, MODELO_OPENAI, PASTA_KNOWLEDGE,  # noqa: E402
                          PROVEDOR_LLM, RESPOSTA_SEM_INFO, TOP_K)
from motor_rag import formatar_fontes, modelos_ollama, ollama_ativo, responder  # noqa: E402

st.set_page_config(page_title="Industrial AI Assistant", page_icon="🤖", layout="wide")


def mostrar_mensagem(msg: dict, mostrar_trechos: bool):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("fontes"):
            st.markdown(formatar_fontes(msg["fontes"]).replace("\n", "  \n"))
        if mostrar_trechos and msg.get("trechos"):
            with st.expander("🔎 Trechos recuperados da base (transparência do RAG)"):
                if msg.get("pergunta_busca") and msg["pergunta_busca"] != msg.get("pergunta"):
                    st.caption(f"Pergunta usada na busca: {msg['pergunta_busca']}")
                if msg.get("motivo"):
                    st.caption(f"Motivo da resposta padrão: {msg['motivo']}")
                for i, t in enumerate(msg["trechos"], start=1):
                    st.markdown(f"**Trecho {i}** — {t['fonte']}, página {t['pagina']} "
                                f"— distância {t['distancia']:.3f}")
                    st.caption(t["texto"])


def salvar_upload(arquivo) -> Path:
    """Confere se o PDF tem texto e salva em knowledge/."""
    nome = Path(arquivo.name).name
    try:
        leitor = PdfReader(arquivo)
        tem_texto = any((p.extract_text() or "").strip() for p in leitor.pages)
    except Exception as erro:
        raise ValueError(f"{nome}: não foi possível ler o PDF ({erro}).")
    if not tem_texto:
        raise ValueError(f"{nome}: o PDF não tem texto selecionável (precisaria de OCR).")
    PASTA_KNOWLEDGE.mkdir(exist_ok=True)
    destino = PASTA_KNOWLEDGE / nome
    destino.write_bytes(arquivo.getvalue())
    return destino


# ------------------------------------------------------------------ cabeçalho
st.title("🤖 Assistente Inteligente de Suporte Técnico Industrial")
st.caption("Rocket Industrial Solutions • respostas baseadas somente na documentação interna (RAG)")

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Primeira execução: indexa os PDFs que já estão em knowledge/
if precisa_indexar():
    with st.spinner("Indexando os PDFs da pasta knowledge/ (na 1ª vez baixa o modelo de embeddings)..."):
        indexar(recriar=True)

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("🚀 Rocket Industrial")

    st.subheader("Modelo de linguagem")
    if PROVEDOR_LLM == "openai":
        st.info(f"OpenAI • {MODELO_OPENAI}")
        modelo = MODELO_OPENAI
        llm_ok = True
    else:
        llm_ok = ollama_ativo()
        if llm_ok:
            st.success("Ollama conectado")
            modelos = modelos_ollama() or [MODELO_OLLAMA]
            padrao = modelos.index(MODELO_OLLAMA) if MODELO_OLLAMA in modelos else 0
            modelo = st.selectbox("Modelo", modelos, index=padrao)
        else:
            st.error("Ollama não está respondendo. Abra o Ollama e recarregue a página.")
            modelo = MODELO_OLLAMA

    st.divider()
    st.subheader("Base de conhecimento")
    c1, c2 = st.columns(2)
    c1.metric("Documentos", len(documentos_indexados()))
    c2.metric("Chunks", total_de_chunks())
    for pdf in listar_pdfs():
        st.text(f"📄 {pdf.name}")

    st.markdown("**Adicionar novos PDFs**")
    uploads = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True,
                               label_visibility="collapsed")
    if st.button("➕ Adicionar à base", type="primary", disabled=not uploads,
                 use_container_width=True):
        with st.spinner("Lendo, dividindo e vetorizando os novos PDFs..."):
            try:
                salvos = [salvar_upload(a) for a in uploads]
                n = indexar(salvos)
                st.success(f"{len(salvos)} PDF(s) adicionados • {n} chunks novos.")
            except ValueError as erro:
                st.error(str(erro))
    if st.button("🔄 Reindexar tudo", use_container_width=True):
        with st.spinner("Recriando a base vetorial..."):
            n = indexar(recriar=True)
        st.success(f"Base recriada com {n} chunks.")

    st.divider()
    st.subheader("Opções")
    top_k = st.slider("Trechos recuperados (Top K)", 1, 8, TOP_K)
    mostrar_trechos = st.toggle("Mostrar trechos recuperados", value=True)
    if st.button("🧹 Limpar conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()

# ------------------------------------------------------------------ chat
if not st.session_state.mensagens:
    st.info("Pergunte sobre equipamentos, códigos de erro, segurança ou manutenção.\n\n"
            "Exemplos: *A máquina CNC modelo X200 apresentou erro E045. Qual procedimento devo realizar?* • "
            "*Qual óleo deve ser utilizado no compressor?* • "
            "*Qual procedimento de segurança antes de entrar na área?*")

for msg in st.session_state.mensagens:
    mostrar_mensagem(msg, mostrar_trechos)

pergunta = st.chat_input("Digite sua dúvida técnica ou código de erro...", disabled=not llm_ok)
if pergunta and pergunta.strip():
    pergunta = pergunta.strip()
    historico = [{"role": m["role"], "content": m["content"]} for m in st.session_state.mensagens]
    msg_usuario = {"role": "user", "content": pergunta}
    st.session_state.mensagens.append(msg_usuario)
    mostrar_mensagem(msg_usuario, mostrar_trechos)

    try:
        with st.spinner("Buscando nos documentos e gerando a resposta..."):
            r = responder(pergunta, historico=historico, modelo=modelo, top_k=top_k)
        msg = {
            "role": "assistant",
            "content": r["resposta"],
            "fontes": [] if r["resposta"] == RESPOSTA_SEM_INFO else r["fontes"],
            "pergunta": pergunta,
            "pergunta_busca": r["pergunta_busca"],
            "motivo": r["motivo"],
            "trechos": [{"fonte": d.metadata["fonte"], "pagina": d.metadata["pagina"],
                         "distancia": dist, "texto": d.page_content} for d, dist in r["trechos"]],
        }
        st.session_state.mensagens.append(msg)
        mostrar_mensagem(msg, mostrar_trechos)
    except Exception as erro:
        st.error(f"Não foi possível gerar a resposta: {erro}")
