# Pesquisa: áudio em tempo real local com o hardware disponível — 2026-10-14

Pedido: "uma investigação avançada de como fazer áudio em tempo real local com
o hardware que temos disponível".

Este documento separa quatro tipos de afirmação:

- **[medido]** vem desta máquina: logs em `~/.debora/logs`, `nvidia-smi`, CIM.
- **[fonte]** vem de documentação ou artigo, com link.
- **[fonte fraca]** vem de divulgação comercial ou de afirmação sem benchmark reproduzível.
- **[suposição]** é estimativa ainda não verificada; cada uma tem um benchmark
  proposto na seção 8.

Nenhum pacote foi instalado e nenhum modelo foi baixado para esta pesquisa.

## 1. Resumo executivo

**O gargalo hoje é o TTS, não o reconhecimento de voz nem o modelo.**

- O Chatterbox Multilingual sintetiza mais devagar que o tempo real: leva
  1,4 s para cada 1 s de áudio (mediana de 458 frases) **[medido]**.
- A primeira frase de verdade começa a tocar 7–10 s depois que o usuário para
  de falar. O "Um instante." do cache cobre só o primeiro segundo e meio.
- O reconhecimento (0,3 s) e o primeiro token do Claude (mediana 1,3 s) somam
  menos de 2 s.

**Recomendação:**

1. **Curto prazo, mantendo a arquitetura:**
   - Trocar o TTS por um modelo com streaming e RTF bem abaixo de 1. Os
     candidatos são o **Qwen3-TTS 0.6B Base**, que mantém a voz clonada da
     Débora, e o **Kokoro-82M** com a voz pt-BR `pf_dora` (modo rápido, sem
     clonagem).
   - Somar o **Smart Turn v3.2** ao Silero VAD, para encerrar o turno antes
     dos 0,8 s de silêncio quando a frase está completa.
   - Usar WASAPI no lugar de MME.
   - Com o TTS mais rápido que o tempo real, a espera do prebuffer (até 3 s
     hoje, para evitar lacunas) cai sozinha.
   - Meta: primeira frase em ~2–2,5 s com o Claude e ~1–1,5 s com um modelo
     local.
2. **Médio prazo:**
   - STT com streaming de verdade, para transcrever durante a fala, e um
     cancelamento de eco real (WebRTC AEC) para usar alto-falante com
     interrupção (barge-in).
   - Ganho no STT: ~0,2–0,5 s. O ganho maior é poder começar o modelo antes.
3. **Experimental:** ainda não há modelo fala-para-fala (full-duplex) aberto,
   em pt-BR, que rode em 8 GB. Vale acompanhar, não construir em cima.

A NPU fica fora do caminho crítico enquanto houver `DEVICE_LOST`
(`docs/PLANO-DISTRIBUICAO-HARDWARE.md`).

## 2. Hardware **[medido]**

| Componente | Valor |
|---|---|
| CPU | Intel Core Ultra 9 185H (Meteor Lake) |
| RAM | 63,5 GiB |
| dGPU | NVIDIA GeForce RTX 4070 Laptop, 8188 MiB, driver 617.14 (CUDA) |
| iGPU | Intel Arc Graphics, driver 32.0.101.9150 (OpenVINO `GPU.0`), memória compartilhada com a RAM |
| NPU | Intel AI Boost (OpenVINO `NPU`), com histórico de `DEVICE_LOST` no whisper-turbo |
| Áudio | Microfone USB, MME, 16 kHz, latência 13 ms; saída (alto-falante ou fone Bluetooth), MME, 44,1 kHz, latência 91 ms |
| Energia | Notebook sempre na tomada |

Com o app ocioso no modo ditado (modelo carregado, sem transcrever), a dGPU
estava com 1219 MiB em uso; ~1,5 GB é o valor durante a transcrição.

## 3. Pipeline atual e linha de base

Configuração atual (`~/.debora/config.json`):

- STT: `turbo` (faster-whisper) em CUDA.
- Fim de turno no chat de voz: `voice_chat_end_silence_seconds = 0.8`.
- Modelo: backend `claude`, o Claude Code via stream-json.
- TTS: Chatterbox Multilingual 0.1.7 em CUDA, com a voz clonada `debora_v2`,
  em processo separado (`tts_server.py`, torch 2.6 cu124).
- Ligados: prebuffer, filtro de eco, barge-in e pausa ao detectar fala.

### Tempo de cada etapa **[medido]**

Fontes: `app.log` e `tts_server.log`, 10–13/10.

| Etapa | Valor medido | Observação |
|---|---|---|
| Silêncio até o VAD encerrar o turno | 0,8 s | configurado |
| STT do turno inteiro (CUDA) | média 0,29 s (200 turnos; RTF ~0,07) | não é gargalo |
| Primeiro token do Claude | p10 0,6 s · mediana 1,3 s · p90 7,5 s (157 turnos) | o p90 inclui turnos com ferramentas |
| TTS Chatterbox | RTF p10 1,14 · mediana 1,39 · p90 1,68 (458 frases, 1913 s de áudio) | **mais lento que o tempo real** |
| Carga do TTS | ~15 s em CUDA | uma vez por sessão |
| Saída de áudio (MME) | 91 ms | WASAPI deve ficar abaixo de 20 ms **[suposição]** |

Exemplo real (turno de 13/10 às 21:54; tempos contados a partir do fim da
transcrição):

```
chunk=1 enqueued=2.365s synth_start=2.374s synth_end=6.255s audio=2.200s play=9.274s
chunk=2 enqueued=2.365s synth_start=6.256s synth_end=11.757s audio=3.560s play=14.746s
```

- A primeira frase levou 3,9 s para gerar 2,2 s de áudio.
- O chunk 2 só ficou pronto depois do fim do chunk 1, então a fala tem
  lacunas: o TTS não acompanha a reprodução.
- Entre `synth_end` (6,26 s) e `play` (9,27 s) houve 3 s de espera. Outros
  turnos mostram o mesmo padrão: 7,13 → 10,16 s e 6,53 → 9,56 s.
- Essa espera é o **prebuffer** funcionando como projetado
  (`voice_chat.py`, `voice_chat_prebuffer_max = 3.0`). O log do mesmo turno
  diz `prebuffer held 3.00s (predicted gap 6.06s, buffered 2.15s, pending 2)`.
  Com RTF > 1, ele segura a reprodução para não haver lacunas no meio da
  resposta. Tirar a espera agora só trocaria o atraso inicial por buracos na
  fala; ela só cai de verdade com um TTS mais rápido que o tempo real.

### Linha de base completa

Do fim da fala do usuário até a primeira frase real:

```
0,8 s VAD + 0,3 s STT + 1,3 s 1º token + ~1 s até a 1ª frase completa
+ ~4 s síntese + ~0–3 s de prebuffer = ~7–10 s
```

O "Um instante." do cache toca em ~1,5 s e disfarça parte dessa espera.

## 4. Arquiteturas

### 4.1 Em cascata com streaming (STT → modelo → TTS)

É o que a Débora já faz. Para ficar em tempo real, cada estágio precisa
emitir saída parcial:

- STT incremental;
- modelo emitindo tokens;
- TTS começando pela primeira oração e com RTF < 1;
- reprodução sobreposta à síntese.

É a única arquitetura que hoje combina pt-BR, voz clonada, o Claude Code como
cérebro e 8 GB de VRAM.

### 4.2 Fala-para-fala nativo, full-duplex

| Modelo | pt-BR | Roda local em 8 GB? | Situação |
|---|---|---|---|
| Moshi (Kyutai, 7B) | Não: inglês; a revisão de 2026 diz "English-only" ([localaimaster](https://localaimaster.com/blog/moshi-realtime-speech-guide)) | Só quantizado **[suposição]** | Existe fine-tune para hindi trocando o tokenizer ([Human-1](https://huggingface.co/JoshTalksAI/Human-1/blob/main/README.md)). Um fine-tune pt-BR seria possível, mas é um projeto de pesquisa |
| Qwen3-Omni (30B-A3B MoE) | Sim, entrada e saída de fala ([repo](https://github.com/QwenLM/Qwen3-Omni)) | Não | Um tutorial de 2026 conseguiu rodar só a parte que gera texto ("Thinker") no vLLM. A parte que sintetiza voz ("Talker") não roda localmente ([arXiv 2603.05413](https://arxiv.org/html/2603.05413v2)) |
| Kyutai Unmute | É cascata (STT + LLM + TTS da Kyutai), não fala-para-fala; STT/TTS em inglês e francês | – | Útil como referência de arquitetura |

Conclusão: **não há opção viável hoje**. Além disso, um modelo fala-para-fala
substitui o cérebro: o Claude Code deixaria de ser quem responde, e isso vai
contra a ideia central do projeto (`docs/VOICE_HARNESS_IDEA.md`).

## 5. Componentes

### 5.1 TTS: a troca de maior impacto

| Modelo | pt-BR | Streaming / 1º áudio | Clonagem | Licença | Onde roda aqui |
|---|---|---|---|---|---|
| **Chatterbox Multilingual 0.1.7 (atual)** | Sim | Não; RTF 1,39 **[medido]** | Sim | MIT | CUDA. Marca o áudio com a marca d'água PerTh |
| **Qwen3-TTS 0.6B / 1.7B Base** | Sim, entre 10 línguas | Sim; "a partir de 97 ms", hardware não informado **[fonte]** ([repo](https://github.com/QwenLM/Qwen3-TTS)) | Sim, com 3 s de referência e transcrição | Apache-2.0 | CUDA (bf16; FlashAttention 2 opcional). VRAM não publicada **[suposição: 0.6B < 3 GB]** |
| **Kokoro-82M** | Sim: `pf_dora` (feminina), `pm_alex`, `pm_santa`; G2P via espeak-ng `pt-br` ([VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)) | Sim; ~1,5 GB e "~100x tempo real" em GPU, número de página comercial, não benchmark **[fonte fraca]** ([tts.ai](https://tts.ai/voices/kokoro/?lang=it)) | Não | Apache-2.0 | CPU, CUDA ou ONNX. Precisa de espeak-ng no Windows **[suposição]** |
| Pocket TTS (Kyutai, 100M) | Português listado, voz `rafael`, variante não informada ([repo](https://github.com/kyutai-labs/pocket-tts)) | ~200 ms até o 1º áudio; ~2,3–6x tempo real em CPU **[fonte]** | Sim (WAV) | Código MIT; licença dos pesos não informada | CPU, deixando a GPU livre |
| Chatterbox-Flash | Não (tokenizer inglês) ([card](https://huggingface.co/ResembleAI/chatterbox-flash)) | 103–119 ms TTFP, RTF ~0,1, GPU não informada **[fonte]** | Sim | MIT | Sem pt-BR |
| Chatterbox-Turbo (350M) | Não, focado em inglês | <200 ms (divulgação) **[fonte]** | Sim | MIT | Sem pt-BR |
| Voxtral TTS (Mistral, 4B) | Sim | ~70 ms (divulgação) **[fonte]** ([MarkTechPost](https://www.marktechpost.com/2026/03/28/mistral-ai-releases-voxtral-tts-a-4b-open-weight-streaming-speech-model-for-low-latency-multilingual-voice-generation/)) | Sim | **CC BY-NC** | Licença não comercial |
| XTTS-v2 | Sim | Streaming ~200 ms **[suposição]** | Sim | **CPML, não comercial** | Licença |

Leitura da tabela:

- O **Qwen3-TTS 0.6B Base** é o único candidato que junta pt-BR, streaming,
  clonagem e licença permissiva. Ele preservaria a voz `debora_v2`.
- O **Kokoro** é o plano B, rápido e barato: voz feminina pt-BR pronta, sem
  clonagem, e pode rodar em CPU para deixar a GPU para o STT.
- O **Pocket TTS** é interessante por rodar em CPU, mas a qualidade e a
  variante do português precisam ser ouvidas antes de qualquer decisão.
- **Já implementado no chat:** primeira oração curta (`split_sentences(first=True)`)
  e log `first_audio=` por turno. O ganho estimado de ~1–2 s ainda precisa
  ser medido; isso não resolve o RTF > 1 e o prebuffer continua evitando lacunas.
- **Etapa 2 implementada, benchmark pendente:** seleção por `tts_backend`,
  scripts isolados para Qwen3 0.6B Base e Kokoro, transcrição distribuída da
  `debora_v2` e cliente de benchmark. A API entrega WAVs completos, sem
  streaming HTTP; a velocidade real e a qualidade precisam da medição local.

### 5.2 STT

O STT atual leva 0,29 s no turno inteiro **[medido]** e é a menor parcela da
espera. Streaming só compensa quando o modelo começa a trabalhar durante a
fala, com transcrição parcial e envio especulativo.

| Modelo | pt-BR | Streaming | Licença | Windows |
|---|---|---|---|---|
| faster-whisper `turbo` (atual) | Sim | Não (os rascunhos reprocessam a janela) | MIT | Sim (CUDA) |
| Nemotron 3.5 ASR Streaming 0.6B | Sim, pronto para uso; WER no FLEURS de 5,8% com chunk de 320 ms e 6,3% com 80 ms **[fonte]** ([card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)) | Sim, chunks de 80 a 1120 ms | OpenMDW-1.1 (uso comercial) | Oficialmente só Linux. Existe um runtime C++ (NeMo-Speech.cpp, GGUF) **[suposição: compila no Windows]** |
| Voxtral Realtime (Mistral) | Sim, entre 13 línguas; WER 8,72% com 480 ms de atraso, conforme blog **[fonte]** ([Weesper](https://weesperneonflow.ai/en/blog/2026-03-31-voxtral-whisper-open-source-speech-models-comparison-2026/)) | Sim, atraso configurável de 80 ms a 2,4 s | Verificar no card **[suposição]** | Verificar |
| Parakeet TDT 0.6B v3 | 25 línguas europeias | Só em blocos, sem cache | CC-BY-4.0 **[suposição]** | **Já integrado** ao app (OpenVINO / onnxruntime, `dictation_engine.py`) |
| Kyutai STT | Não (inglês e francês) | Sim | – | – |

### 5.3 Fim de turno, interrupção e eco

| Item | Hoje | Proposta |
|---|---|---|
| Fim de turno | Silero VAD + 0,8 s de silêncio | Silero VAD + **Smart Turn v3.2**, que lê a prosódia ([repo](https://github.com/pipecat-ai/smart-turn)): <ul><li>português entre 23 línguas; BSD-2;</li><li>8M parâmetros, 8 MB em ONNX int8, ~10–65 ms em CPU **[fonte]**;</li><li>janela de até 8 s de áudio 16 kHz.</li></ul>Com ~200 ms de silêncio, se o Smart Turn disser "completo", fecha o turno; senão espera até 0,8–2 s. O `onnxruntime` já está no ambiente. Ganho esperado: ~0,5 s **[suposição]** |
| Eco e interrupção | Filtro de eco por texto (`voice_chat_echo_filter`) e pausa ao detectar fala | Cancelamento acústico de eco (WebRTC AEC), usando o sinal do alto-falante como referência: <ul><li>`pywebrtc-audio` tem wheels para Windows x86_64 e Python 3.10–3.14, com `process(near, far)` e `stream_delay_ms` ([docs](https://strandsagents.com/docs/labs/pywebrtc-audio/));</li><li>`echoff` usa loopback WASAPI ([GitHub](https://github.com/koljab/echoff));</li><li>alternativa: o APM do SDK LiveKit, com quadros de 10 ms.</li></ul>Com fone, o AEC pode ficar desligado |
| Full-duplex | Não | Depende do AEC: só com eco cancelado dá para ouvir enquanto fala sem se auto-interromper |
| Modelos de decisão tipo "Jev" | – | Não recebem áudio; continua valendo `docs/benchmarks/turn-detection-e-classificadores-2026-10-10.md` |

### 5.4 Modelo (LLM)

- Claude Code: primeiro token com mediana de 1,3 s **[medido]**. É um piso
  difícil de baixar. As alavancas são:
  - processo quente, que o harness já mantém;
  - prompt curto;
  - pedir que a primeira oração seja curta, para o TTS começar logo.
- Modelo local: o Gemma 4 E4B int4 na iGPU deu primeiro token em 0,2–0,4 s e
  ~15 tok/s, e segue o prompt em pt-BR melhor que o Qwen3-8B atual
  (`docs/benchmarks/llm-2026-10-09/README.md`) **[medido em 09/10]**.
  - Para o backend local, trocar `llm_model` para o Gemma 4 E4B é a maior
    redução de latência fora o TTS. Isso exige o `VLMPipeline` no
    `llm_server.py`.

## 6. Distribuição no hardware

### Orçamento de VRAM (8 GB) com tudo ativo ao mesmo tempo

| Componente | Dispositivo | VRAM |
|---|---|---|
| STT turbo (faster-whisper) | CUDA | ~1,5 GB **[medido em 14/10]** |
| TTS Qwen3-TTS 0.6B Base | CUDA | ~2–3 GB **[suposição]** |
| TTS Kokoro-82M (alternativa) | CPU ou CUDA | ~0,5–1,5 GB em CUDA; 0 em CPU **[fonte/suposição]** |
| TTS Chatterbox (atual, para comparar) | CUDA | não medido; estimar com `nvidia-smi` durante a síntese |
| Modelo local Gemma 4 E4B int4 | **iGPU (Arc)** | 0 de VRAM; ~6,5 GB de RAM compartilhada |
| Silero VAD + Smart Turn | CPU | 0 |
| AEC WebRTC | CPU | 0 |
| **Total na dGPU** | | **~3,5–4,5 GB de 8 GB**, sem contar o que o Windows e o desktop já usam na dGPU (medir com `nvidia-smi`) |

Regras propostas:

1. **A dGPU fica com STT e TTS**, os dois estágios sensíveis a latência que
   dependem de CUDA.
2. **A iGPU fica com o modelo local**, sem disputar VRAM.
3. **A CPU fica com VAD, fim de turno e AEC**. O 185H tem margem de sobra.
4. **NPU:** não usar no caminho crítico enquanto o `DEVICE_LOST` não estiver
   resolvido: o erro foi reproduzido fora do app, com o driver NPU 32.0.100.5540
   e o OpenVINO 2025.4.1, então não é o cache nem o código do app. O
   tratamento do erro está em `docs/PLANO-DISTRIBUICAO-HARDWARE.md`. Se voltar a ser confiável, é
   candidata a rodar o Smart Turn ou o VAD em baixo consumo, sem ganho real de
   latência.
5. **Disputa na dGPU:** STT e TTS podem rodar ao mesmo tempo na interrupção,
   quando o usuário fala sobre a resposta.
   - O faster-whisper e o torch do TTS ficam em processos separados, com
     contextos CUDA próprios.
   - Medir o aumento de RTF quando os dois rodam juntos (seção 8).
6. **Energia e temperatura:** na tomada, o 4070 Laptop pode reduzir o clock
   quando aquece. `nvidia-smi` não informa `power.limit` neste notebook.
   Monitorar `clocks.sm` e a temperatura durante sessões longas.

### Meta de latência (do fim da fala até o primeiro áudio)

| Etapa | Hoje | Meta com Claude | Meta com modelo local |
|---|---|---|---|
| Fim de turno | 0,8 s | 0,3 s (Smart Turn) | 0,3 s |
| STT | 0,3 s | 0,3 s | 0,3 s |
| 1º token | 1,3 s | 1,2 s | 0,3 s |
| Até a 1ª oração completa | ~1 s | 0,3 s (oração curta) | 0,3 s |
| TTS até o 1º áudio | ~4 s | 0,2 s (streaming) | 0,2 s |
| Espera do prebuffer | 0–3 s | ~0, só com TTS de RTF < 1 | ~0, só com TTS de RTF < 1 |
| Saída de áudio | 0,09 s | 0,02 s (WASAPI) | 0,02 s |
| **Total** | **~7–10 s** | **~2,3 s** | **~1,4 s** |

Ficar abaixo de 800 ms exige começar o modelo durante a fala, com STT em
streaming e envio especulativo. Isso é médio prazo e só vale com o modelo
local, porque cada especulação descartada custa uma chamada.

## 7. Plano em etapas

Cada etapa só avança se o benchmark dela passar.

| Etapa | O que fazer | Critério para seguir |
|---|---|---|
| **0. Medir o que já existe** | <ul><li>Registrar a espera do prebuffer (já logada como `prebuffer held`) junto do tempo total.</li><li>Logar o tempo do fim da fala até o primeiro áudio real (`voice_to_voice`) em cada turno.</li><li>Medir a VRAM do Chatterbox.</li></ul> | Linha de base com p50 e p90 de `voice_to_voice` |
| **1. Ganhos sem trocar modelo** | <ul><li>Primeira oração curta para o TTS.</li><li>Ajustar `voice_chat_prebuffer_max` só depois de medir lacunas versus atraso.</li><li>Saída por WASAPI.</li></ul> | `voice_to_voice` p50 −1 s sem aumentar as lacunas |
| **2. TTS novo — implementada, benchmark pendente** | `tts_backend`: `chatterbox` (padrão), `qwen3` (0.6B Base, clonagem com WAV + transcrição) ou `kokoro` (`pf_dora`, sem clonagem). Scripts uv isolados, mesma API HTTP, um WAV completo por frase; sem streaming por chunks nesta etapa. `debora_v2.txt` acompanha a voz padrão | Medir RTF e tempo até a resposta completa (p50/p90); alvo RTF < 0,3 e avaliação pt-BR às cegas (10 frases de cada). A meta de 1º chunk < 300 ms fica para uma etapa futura com streaming |
| **3. Smart Turn** | Fim de turno em duas fases (seção 5.3) | Menos de 5% de cortes no meio da frase em 50 turnos reais; −0,4 s p50 |
| **4. AEC** | WebRTC AEC com o alto-falante como referência; barge-in sem fone | Sem auto-interrupção em 20 respostas pelo alto-falante |
| **5. Streaming STT + especulação** | Nemotron streaming (se compilar no Windows) ou Voxtral Realtime; começar o modelo local com o texto parcial | `voice_to_voice` < 1 s com o modelo local |
| **6. Acompanhar** | Fala-para-fala pt-BR que caiba em 8 GB | Reavaliar a cada trimestre |

## 8. Benchmarks propostos (comandos exatos)

Os servidores TTS rodam fora do app, cada um em seu ambiente uv isolado.
O cliente `docs/benchmarks/tts-bench/bench_tts.py` usa apenas a biblioteca padrão
e envia as 20 frases de `frases_ptbr.txt` a um servidor já iniciado.
**Não foram executados servidores nem baixados modelos nesta implementação.**
Ao executar os servidores pela primeira vez, uv prepara as dependências e os
backends baixam seus pesos. Comandos completos e instruções do espeak-ng no
[README do benchmark](benchmarks/tts-bench/README.md).

```powershell
# Durante cada benchmark, em outro terminal:
nvidia-smi --query-gpu=memory.used,clocks.sm,temperature.gpu --format=csv -l 1

# Terminal A: escolher UM servidor por vez, com o app parado
uv run --script debora_whisper/tts_server.py --device cuda --voice debora_whisper/voices/debora_v2.wav
uv run --script debora_whisper/tts_server_qwen3.py --device cuda --voice debora_whisper/voices/debora_v2.wav
uv run --script debora_whisper/tts_server_kokoro.py --device cpu
uv run --script debora_whisper/tts_server_kokoro.py --device cuda

# Terminal B: após o servidor responder /health, repetir para cada backend
Invoke-RestMethod http://127.0.0.1:8765/health
uv run --script docs/benchmarks/tts-bench/bench_tts.py --out "$env:TEMP\debora-tts-bench"
```

- Os servidores são scripts uv (PEP 723) com o padrão do `tts_server.py`:
  - versões exatas (`qwen-tts==0.1.1`, `kokoro==0.9.4`, torch/torchaudio 2.6.0);
  - o índice `pytorch-cu124` restrito a `torch`/`torchaudio` via
    `[tool.uv.sources]`, nunca um `--index` global (evita confusão de
    dependências).
- `debora_whisper/voices/debora_v2.txt` é o sidecar UTF-8 distribuído com a
  voz padrão, transcrição confirmada pelo usuário. O Qwen3 exige esse texto
  para clonar. Outras vozes e WAVs que sobrescrevem a voz padrão precisam do
  próprio `<voice>.txt` ao lado do WAV; nunca inferimos a transcrição.

- `bench_tts.py` mede, para cada uma das 20 frases
  de `frases_ptbr.txt`:
  - tempo até receber a resposta HTTP completa;
  - duração do WAV e RTF (tempo de resposta / duração), com p50/p90;
  - CSV por frase e JSON com inventário do servidor, após um aquecimento
    excluído da medição (`--warmup 0` para incluir a primeira requisição fria).
  VRAM no pico é registrada separadamente com `nvidia-smi`. Não mede primeiro
  chunk nem `voice_to_voice`: não há streaming HTTP nesta etapa.
- Os WAVs ficam fora do git (pasta temporária local), só para a avaliação às
  cegas. No repositório entram só os números.
- Smart Turn: rodar o modelo ONNX int8 (8 MB) com o `onnxruntime` do `.venv`
  sobre os WAVs dos turnos salvos (se houver gravações) ou sobre 50 frases
  gravadas na hora.
- Medir a disputa na dGPU: STT e TTS ao mesmo tempo, com o faster-whisper
  transcrevendo em loop enquanto o TTS sintetiza. Comparar o RTF dos dois com
  o RTF de cada um rodando sozinho.

## 9. Riscos

- **Licenças:**
  - Voxtral TTS (CC BY-NC) e XTTS-v2 (CPML) ficam fora.
  - A licença dos pesos do Pocket TTS não está publicada.
  - Clonar voz exige consentimento de quem cedeu a voz. O usuário confirmou
    o áudio fonte e a transcrição da `debora_v2`; essa confirmação não deve
    ser interpretada como uma nova autorização de publicação de áudio gerado.
  - O Chatterbox marca o áudio com a marca d'água PerTh. Qwen3-TTS e Kokoro
    não parecem ter equivalente **[suposição]**: trocar o modelo perde essa
    marca.
- **Windows:**
  - Nemotron e NeMo são oficialmente só Linux.
  - Kokoro em pt usa o espeak-ng, que vem embutido no ambiente do script
    (pacote `espeakng-loader`); não precisa instalar no sistema.
  - O Qwen3-TTS recomenda FlashAttention 2, que é difícil de compilar no
    Windows. Sem ele deve funcionar, com mais memória **[suposição]**.
  - O Smart App Control do Windows já bloqueou o uv nesta máquina (erro de
    SO 4551 ao iniciar o servidor de TTS).
- **Qualidade pt-BR:**
  - Os números de latência das fontes são de divulgação, quase sempre em
    inglês e sem hardware informado.
  - A qualidade em pt-BR só se decide ouvindo (etapa 2).
- **Números de divulgação:** o "97 ms" do Qwen3-TTS e o "~100x" do Kokoro não
  foram reproduzidos em nenhuma fonte independente.
- **Disputa na dGPU:** interromper com STT e TTS ativos pode piorar os dois.
  É preciso medir.

## Fontes

- Qwen3-TTS — https://github.com/QwenLM/Qwen3-TTS
- Chatterbox-Flash — https://huggingface.co/ResembleAI/chatterbox-flash · artigo https://arxiv.org/pdf/2605.30748
- Kokoro-82M, vozes — https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
- Pocket TTS — https://github.com/kyutai-labs/pocket-tts
- Voxtral TTS — https://www.marktechpost.com/2026/03/28/mistral-ai-releases-voxtral-tts-a-4b-open-weight-streaming-speech-model-for-low-latency-multilingual-voice-generation/
- Nemotron 3.5 ASR Streaming — https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- Voxtral vs Whisper (2026) — https://weesperneonflow.ai/en/blog/2026-03-31-voxtral-whisper-open-source-speech-models-comparison-2026/
- Comparação de STT local — https://openwhispr.com/blog/parakeet-vs-whisper-vs-nemotron
- Smart Turn — https://github.com/pipecat-ai/smart-turn · https://huggingface.co/pipecat-ai/smart-turn-v3
- pywebrtc-audio — https://strandsagents.com/docs/labs/pywebrtc-audio/ · echoff — https://github.com/koljab/echoff
- Qwen3-Omni — https://github.com/QwenLM/Qwen3-Omni · tutorial de voz em tempo real — https://arxiv.org/html/2603.05413v2
- Moshi — https://localaimaster.com/blog/moshi-realtime-speech-guide · Human-1 — https://huggingface.co/JoshTalksAI/Human-1/blob/main/README.md
- Roundups de TTS 2026 (só como pista; números não verificados) — https://openvoxai.com/blog/best-free-local-tts-models-2026
