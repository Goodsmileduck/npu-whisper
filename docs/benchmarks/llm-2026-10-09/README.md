# LLMs locais para o chat de voz — 2026-10-09

Mesmas conversas em português, três amostras cada (temperature 0.7, top_p 0.8,
top_k 20, repetition_penalty 1.1, pensamento desligado), no Arc iGPU do Core
Ultra 9 185H com OpenVINO GenAI 2026.4.1. Script: `llm_bench.py`; saídas brutas:
`bench_*.txt`.

- **full**: o prompt de persona longo (`VOICE_CHAT_PROMPTS["pt"]` neste dia).
- **short**: um prompt de 4 linhas (no script).
- **clean**: pedir só para corrigir a transcrição, sem responder.

## Conversa

| Modelo (int4) | Download | Prompt | 1º token | tok/s | Sabe o nome do usuário | STT quebrado: pede repetir | Data por extenso e certa | Emoji | Oferta de ajuda no fim | Feminino |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (atual) | 4.6 GB | full | 0.3 s | ~11 | 0/3 ("É o senhor que quer saber o seu nome?") | 1/3 | 0/3 ("09 outubro de 2") | 2 | indireta | – |
| Qwen3-8B | | short | 0.2 s | ~13.5 | 3/3 | 0/3 (responde a data) | 0/3 (dígitos; dia errado 2/3) | 0 | 0 | – |
| Qwen3.5-9B (VLM) | 6.3 GB | full | 2.1 s | ~6 | 3/3 | 2/3 | 3/3 | 0 | 1 | – |
| Qwen3.5-9B (VLM) | | short | 1.1 s | ~8.5 | 3/3 (com ressalvas estranhas) | 0/3 | 3/3 | 0 | 0, mas puxa assunto | – |
| **Gemma 4 E4B (VLM)** | 6.5 GB | **full** | **0.2–0.4 s** | **~15** | **3/3** | **3/3** | **3/3** | **0** | **0** | **sim** ("obrigada", "ótima") |
| Gemma 4 E4B (VLM) | | short | 0.3 s | ~14.5 | 3/3 | 0/3 (responde a data) | 3/3 | 0 | 3/3 no cumprimento | sim |

Exemplos do Gemma 4 E4B com o prompt longo:

- "Olá, tudo bem?" → "Oi! Estou ótima, obrigada por perguntar. E você, como está?"
- "Você sabe que dia que eu agir?" → "Desculpe, não entendi bem. Pode repetir sua pergunta?"
- "Débora, que dia é hoje?" → "Hoje é sexta-feira, nove de outubro de dois mil e vinte e seis."
- "Tô cansado, acho que vou parar um pouco." → "Pode pausar sim. Descanse um pouco se precisar. Estou aqui quando quiser continuar."

Conclusões:

- O Qwen3-8B é o pior dos três; com o prompt longo, a pilha de regras piora as
  respostas em vez de corrigi-las.
- O Gemma 4 E4B segue o prompt longo por inteiro, é o mais rápido e o único
  que fala no feminino sem ser lembrado. Ele é multimodal: precisa do
  `VLMPipeline` (o `LLMPipeline` falha com "Port for tensor name input_ids was
  not found"), o que muda `llm_server.py`.
- O Qwen3.5-9B acerta, mas é ~4x mais lento no 1º token e também é VLM.

## Limpar a transcrição antes de mandar para um harness?

Transcrições reais (as duas primeiras são do `app.log`) e pedidos de código:

| Entrada | Qwen3.5-9B (1.6–2.2 s) | Gemma 4 E4B (0.7–1.2 s) | Claude Code com o texto cru |
|---|---|---|---|
| Você sabe que dia que eu agir? | "...que eu agirei?" ✗ | "...que eu agirei?" ✗ | "provavelmente: que dia é hoje" ✓ |
| E o meu nome é Você Sabe? | igual ✗ | igual ✗ | "E o meu nome, você sabe?" ✓ |
| abre o ríd mi do projeto | README ✓ | README ✓ | – |
| mensagem fiks no bug do áudio | fix ✓ | "fix no bug do áudio" ✓ | – |
| roda os testes do vóis chat | "Vóis Chat" ✗ | "Voice Chat" ✓ | `tests/test_voice_chat.py` ✓ |
| muda o default ... no dictêixon engine | "Dicionário Engine" ✗ | "Decision Engine" ✗ | `dictation_engine.py` ✓ |

Um modelo local não conhece o repositório: corrige o que é óbvio, erra o
que é específico do projeto ("dictêixon engine" virou outra coisa, o que
piora o pedido) e soma ~1 s. O harness entendeu os quatro pedidos crus
porque tem o contexto do repositório. Recomendação: mandar a transcrição
direto ao harness, avisando que ela vem de reconhecimento de voz e pode ter
erros.

## Harnesses (uma pergunta curta, processo novo a cada chamada)

| CLI | Versão | Tempo |
|---|---|---|
| `claude -p` | 2.1.295 | 6.1 s |
| `codex exec` | 0.161.0 | 6.1 s |
| `copilot -s -p` | 1.0.93 | 7.9 s |
| `agy -p` | 1.3.2 | 9.3 s |
