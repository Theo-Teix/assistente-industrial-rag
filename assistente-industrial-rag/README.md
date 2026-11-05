# Assistente Inteligente de Suporte Técnico Industrial (RAG)

Exercício Rocket Industrial Solutions - UniMAX Indaiatuba - Prof. Rogério Morandi.

Chatbot que responde dúvidas técnicas usando somente os PDFs da pasta `knowledge`.
Cada resposta mostra o arquivo e a página de origem. Quando a informação não está nos
documentos, ele responde: "Não encontrei essa informação na documentação disponível."

Tecnologias: Python, LangChain, ChromaDB, Sentence-Transformers, Ollama (Llama 3.2) e Streamlit.

## Arquivos

- `app/carregador.py` - lê os PDFs e divide em chunks
- `app/banco_vetorial.py` - cria os embeddings e salva no ChromaDB
- `app/motor_rag.py` - busca os trechos, monta o prompt e gera a resposta
- `app/interface.py` - tela de chat com upload de PDF
- `app/testar_cenarios.py` - testa as perguntas do enunciado
- `knowledge/` - PDFs da empresa

## Como rodar

1. Instale o Python 3.12 (marque "Add Python to PATH").
2. Instale o Ollama (ollama.com) e rode no Prompt de Comando: `ollama pull llama3.2`
3. Execute `01_INSTALAR.bat`
4. Execute `02_INICIAR.bat` (abre o chat no navegador)
5. Para testar os cenários, execute `03_TESTAR_CENARIOS.bat`

## Perguntas para testar

- A máquina CNC modelo X200 apresentou erro E045. Qual procedimento devo realizar?
- Qual óleo deve ser utilizado no compressor?
- Qual procedimento de segurança antes de entrar na área?
- Qual o procedimento de manutenção da máquina de corte a laser Modelo Y500? (deve responder que não encontrou)
