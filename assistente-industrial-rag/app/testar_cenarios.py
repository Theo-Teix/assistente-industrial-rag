"""Roda os cenários do enunciado e diz se cada um PASSOU ou FALHOU.

Uso (com o Ollama aberto):  python app/testar_cenarios.py
"""

import sys

from banco_vetorial import indexar, precisa_indexar
from configuracao import RESPOSTA_SEM_INFO
from motor_rag import formatar_fontes, responder


def tem_fonte(r, arquivo, pagina):
    return {"fonte": arquivo, "pagina": pagina} in r["fontes"]


def tem_palavras(r, *palavras):
    texto = r["resposta"].lower()
    return all(p in texto for p in palavras)


CENARIOS = [
    ("Erro CNC (exemplo 1 do enunciado)",
     "A máquina CNC modelo X200 apresentou erro E045. Qual procedimento devo realizar?",
     lambda r: tem_fonte(r, "01_manual_cnc.pdf", 34) and tem_palavras(r, "deslig", "sensor", "calibra")),
    ("Óleo do compressor (exemplo 2)",
     "Qual óleo deve ser utilizado no compressor?",
     lambda r: tem_fonte(r, "03_manual_compressor.pdf", 12) and "vg 68" in r["resposta"].lower()),
    ("Segurança NR-12 (exemplo 3)",
     "Qual procedimento de segurança antes de entrar na área?",
     lambda r: tem_fonte(r, "02_procedimento_seguranca.pdf", 8) and tem_palavras(r, "capacete", "óculos")),
    ("Nível 3: máquina que não existe na base",
     "Qual o procedimento de manutenção da máquina de corte a laser Modelo Y500?",
     lambda r: r["resposta"] == RESPOSTA_SEM_INFO and not r["fontes"]),
    ("Nível 3: assunto fora da documentação",
     "Qual é o salário de um técnico de manutenção na Rocket?",
     lambda r: r["resposta"] == RESPOSTA_SEM_INFO and not r["fontes"]),
]


def main():
    if precisa_indexar():
        print("Indexando a pasta knowledge/ ...")
        indexar(recriar=True)

    aprovados = 0
    for titulo, pergunta, criterio in CENARIOS:
        r = responder(pergunta)
        ok = criterio(r)
        aprovados += ok
        print(f"\n{'✅ PASSOU' if ok else '❌ FALHOU'} — {titulo}\nPergunta: {pergunta}\n")
        print(r["resposta"])
        if r["fontes"]:
            print(formatar_fontes(r["fontes"]))

    # Histórico: pergunta de acompanhamento que depende da anterior
    historico = []
    p1 = "Qual óleo deve ser utilizado no compressor?"
    r1 = responder(p1)
    historico += [{"role": "user", "content": p1}, {"role": "assistant", "content": r1["resposta"]}]
    p2 = "E com qual frequência devo trocar esse óleo?"
    r2 = responder(p2, historico=historico)
    ok = "4.000" in r2["resposta"] or "4000" in r2["resposta"] or "12 meses" in r2["resposta"]
    aprovados += ok
    print(f"\n{'✅ PASSOU' if ok else '❌ FALHOU'} — Histórico de conversa\nPergunta: {p2}")
    print(f"(reescrita para: {r2['pergunta_busca']})\n")
    print(r2["resposta"])
    if r2["fontes"]:
        print(formatar_fontes(r2["fontes"]))

    total = len(CENARIOS) + 1
    print(f"\nResultado: {aprovados}/{total} cenários aprovados.")
    return 0 if aprovados == total else 1


if __name__ == "__main__":
    sys.exit(main())
