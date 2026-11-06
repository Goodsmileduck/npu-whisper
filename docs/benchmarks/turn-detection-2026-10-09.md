# Detecção de fim de turno: substituir ou complementar o Silero VAD? — 2026-10-09

Pesquisa feita por um subagente; os números são dos autores de cada projeto
e não foram reproduzidos aqui.

**Hoje** (`dictation_engine.py`): Silero v5 em ONNX na CPU, 16 kHz, blocos de
512 amostras (32 ms), limiar 0.5 (0.35 depois que a fala começa). No chat de
voz, o turno termina após `voice_chat_end_silence_seconds` (0.8 s) de
silêncio; o `AdaptiveEndpoint` só vale no ditado.

O pedido original ("teste com o JEV") veio por voz; em 2026-10-10 o JEV e sua
safra de modelos de decisão foram identificados e avaliados em
[turn-detection-e-classificadores-2026-10-10.md](turn-detection-e-classificadores-2026-10-10.md).

| Candidato | O que faz | PT-BR | Formato e licença | Observações |
|---|---|---|---|---|
| [TEN VAD](https://github.com/TEN-framework/ten-vad) | VAD por frame (10/16 ms); troca direta do Silero | acústico, sem teste em PT | lib nativa ~500 KB e ONNX; fora do PyPI; Apache 2.0 **com restrições** (proíbe uso concorrente à Agora) | RTF 0.0086 contra 0.0127 do Silero; diz detectar o fim da fala centenas de ms antes, só em gráficos |
| [Smart Turn v3 (Pipecat)](https://github.com/pipecat-ai/smart-turn) | fim de turno pela prosódia, nos últimos 8 s; complementa o VAD | 95.42% de acerto em PT, 2.79% de cortes cedo ([blog](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)) | ONNX int8 de 8 MB, BSD-2 | 12–95 ms por decisão na CPU; o log-mel precisa ser reescrito em numpy (o original usa `transformers`) |
| [LiveKit turn-detector](https://huggingface.co/livekit/turn-detector) | fim de turno pelo texto (Qwen2.5-0.5B) | TPR 99.4%, TNR 87.4% | ONNX int8, ~500 MB de RAM; licença própria e restritiva | |
| [TEN Turn Detection](https://github.com/TEN-framework/ten-turn-detection) | fim de turno pelo texto (Qwen2.5-7B, torch) | só inglês e chinês | | descartado |

**Recomendação**: o problema é decidir quando o turno acabou, não detectar
fala; testar o **Smart Turn v3** junto do Silero. Depois de ~200–300 ms de
silêncio, roda-se o Smart Turn: "completo" (p > 0.5) fecha o turno; senão,
espera até um teto de 2–3 s. O TEN VAD fica como segundo teste, depois de
avaliar a licença.

**Plano de teste offline**: gravar 40–60 falas em PT-BR a 16 kHz no microfone
de uso (metade com pausas no meio da frase, algumas com ruído), marcar o fim
real, e passar os mesmos WAVs pelo Silero com 0.8 s e pelo Silero mais o Smart
Turn. Medir o tempo do fim da fala até o fim do turno (mediana e p90), os
cortes no meio da frase, a fala perdida e o tempo de CPU por decisão. Critério:
pelo menos 300 ms mais rápido sem mais cortes.
