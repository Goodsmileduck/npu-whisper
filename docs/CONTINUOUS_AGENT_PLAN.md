# Plano para Agente de Voz Contínuo

Este documento descreve a evolução do projeto de ditado para um agente local
que escuta continuamente, identifica quando o usuário está falando com ele e
executa comandos autorizados. Todo o processamento permanece local: nenhum
áudio, transcrição ou validação de licença depende de serviços externos.

> Revisado em 2026-10-06 para refletir o estado do app após o commit
> `711096d` (eventos de hardware, fallback de dispositivo em tempo de execução
> e probe de recuperação da NPU).

## Objetivos

O agente deve:

- manter a captura de áudio ativa sem exigir uma hotkey para cada frase;
- detectar início e fim de fala;
- transcrever segmentos curtos com baixa latência;
- distinguir fala dirigida ao agente de conversa ambiente;
- interpretar intenções de forma determinística;
- executar apenas ações explicitamente permitidas, sem privilégio elevado;
- pedir confirmação para ações sensíveis;
- informar estado, erro e resultado ao usuário;
- manter o modo atual de ditado (push-to-talk) como alternativa e padrão.

O modo agente é um modo **separado**. O fluxo de ditado atual não deve mudar
de comportamento até que o novo fluxo esteja validado.

## Estado atual (baseline)

O que já existe e deve ser reaproveitado — ou levado em conta — pelo agente:

| Área | Comportamento atual | Impacto no agente |
|---|---|---|
| Captura | `AudioRecorder` com stream persistente (`warmup`), lookback circular e telemetria por sessão (`live_frames`, `input_overflows`, gaps de callback/ADC). Durante a gravação, acumula blocos em `self._frames`. | Base da captura contínua, mas o acúmulo em lista precisa virar ring buffer pré-alocado. |
| Recuperação de áudio | `TrayManager` escuta `WM_POWERBROADCAST` (resume) e `WM_DEVICECHANGE` e chama `GUIApp._on_hardware_event`, que fecha e reaquece o stream se não estiver gravando. `_do_start` tenta `close()` + `warmup()` se o stream não estiver pronto. | No modo contínuo o stream está sempre "gravando": o evento precisa de debounce e de uma verificação de que o endpoint de áudio realmente mudou (`DBT_DEVNODES_CHANGED` dispara para qualquer dispositivo USB). |
| Falha de acelerador | `classify_device_failure` distingue GPU/NPU/UNKNOWN. Falhas classificadas geram payload `device_lost`; `GUIApp._update_ui` troca de dispositivo em tempo de execução (NPU → CUDA/GPU; GPU/CUDA → NPU) via `DictationApp.fallback_device`. Falha *latched* (`RestartRequiredError`) ainda exige reinício. | O agente herda esse fallback, mas precisa tratar a janela de recarga (fila pausada) e o caso latched (estado `ERROR`, escuta desligada). |
| Recuperação da NPU | Após fallback, `_schedule_npu_recovery` roda até 2 probes (30 s, 60 s) carregando um modelo NPU de teste e faz hot-swap com `inject_recovered_model`. | O swap precisa ser atômico com relação à fila de transcrição e liberar o modelo anterior (VRAM/NPU). Ver "Pendências do baseline". |
| Espera pelo modelo | `_do_stop` agora bloqueia em `self._model_ready.wait()` sem timeout antes de transcrever. | Na fila contínua, isso trava o consumidor indefinidamente; o agente precisa de timeout e descarte explícito. |
| Alucinação | Filtro por correspondência exata de um conjunto pequeno ("obrigado", "thank you", "obrigado por assistir"…). | Insuficiente para escuta contínua; ver Fase 3. |
| Saída de texto | `type_text(text, auto_enter=config["auto_enter"])` digita na janela em foco e grava no histórico (`MAX_HISTORY`). `app.log` registra `Typed: …`. | No modo agente, `auto_enter` deve ser ignorado e o histórico não pode receber fala ambiente. |
| Logs | `app.log` (eventos, timings, hardware) e `telemetry.log` (linhas com `[Telemetry]`: CPU, RAM do app/sistema, VRAM, saúde do buffer de áudio). | Métricas do agente seguem o mesmo roteamento por prefixo; conteúdo transcrito não vai para telemetria. |
| Backends | OpenVINO (Whisper/Parakeet em NPU/iGPU/CPU) e `faster-whisper` em CUDA (`int8_float16`, `condition_on_previous_text=False`). Parakeet usa `MEL_BUCKETS` de ~2/5/9/16 s e trunca acima de 16 s. | A política de segmentação deve caber nos buckets e não depender de contexto entre segmentos por padrão. |
| Privilégio | O app roda como Administrador por causa das hotkeys globais. | Qualquer processo lançado pelo executor herdaria privilégio elevado. Ver Fase 6. |

### Pendências do baseline (resolver antes ou durante a Fase 1)

1. **`utils.logger` não existe no repositório.** `app.py` importa
   `from utils.logger import log` em `_on_hardware_event`,
   `_schedule_npu_recovery` e `_run_npu_recovery_probe`. Os imports estão fora
   do `try`, então as threads morrem com `ModuleNotFoundError` e a recuperação
   de áudio e da NPU nunca roda. Usar `dictation_engine.log` ou criar o módulo.
2. **Hot-swap sem liberar o modelo anterior.** `inject_recovered_model`
   substitui `self.whisper` sem descarregar o modelo de fallback, mantendo
   VRAM/memória da NPU ocupadas. Também não coordena com uma transcrição em
   andamento que já leu `self.whisper`.
3. **Probe na NPU após `DEVICE_LOST` no mesmo processo.** O design anterior
   recusava novas chamadas porque o driver pode travar. O probe deve rodar com
   timeout e, de preferência, em processo separado (ver Fase 3).
4. **`_model_ready.wait()` sem timeout** em `_do_stop`.
5. **`WM_DEVICECHANGE` sem debounce**: rajadas de eventos reabrem o stream
   várias vezes.

## Arquitetura proposta

```text
Microfone contínuo (stream persistente, reconexão com debounce/backoff)
        |
        v
Ring buffer pré-alocado (NumPy)  <-- callback só copia e sinaliza
        |
        v
VAD em cascata (energia -> Silero/WebRTC)
        |                      \
        v                       +--> wake word (recebe lookback do ring buffer)
Segmentador (início/fim por VAD, corte forçado em fala longa)
        |
        v
Fila de transcrição limitada (backpressure, descarte explícito)
        |
        v
Worker de inferência (instância única; fallback/recuperação de dispositivo)
        |
        v
Filtros de alucinação e normalização
        |
        v
Roteador de intenção (gramática determinística + allowlist)
        |
        +--> resposta informativa (overlay)
        +--> comando permitido --> confirmação --> executor (não elevado)
        +--> texto comum --> ditado (somente se o agente estiver ativo) ou descarte
```

Duas máquinas de estado independentes:

- **Captura**: `STOPPED`, `LISTENING`, `PAUSED`, `RECONNECTING`, `DEVICE_LOST`.
- **Agente**: `INACTIVE`, `ACTIVE`, `CONFIRMING`, `EXECUTING`, `ERROR`.

O estado de cada segmento (`detected`, `queued`, `transcribing`, `done`,
`discarded`) é atributo do segmento, não estado global — fala nova pode chegar
enquanto outro segmento é transcrito. O enum `AppState` existente continua
dirigindo overlay/tray; os novos estados são mapeados para ele em vez de
criar um segundo canal de UI.

## Fases de implementação

### Fase 0 — Intenções sobre o push-to-talk atual

Antes de qualquer escuta contínua, implementar roteador e executor (Fases 5 e
6) usando a hotkey atual como gatilho. Isso entrega valor, testa a camada de
segurança cedo e não introduz o risco de falsos acionamentos.

Critério de saída: comandos da allowlist funcionam via push-to-talk, comandos
fora dela são recusados e cobertos por testes automáticos.

### Fase 1 — Captura contínua

Estender o `AudioRecorder` (não criar outro) com um modo contínuo:

- ring buffer NumPy pré-alocado com capacidade configurável; o callback do
  `sounddevice` apenas copia para o buffer, atualiza contadores e sinaliza um
  `threading.Event` — sem VAD, logging ou alocação de listas no callback;
- um único thread consumidor para VAD/segmentação (não um thread por bloco);
- reaproveitar a telemetria existente (`input_overflows`, gaps) e emiti-la
  periodicamente com prefixo `[Telemetry]`;
- reconexão: integrar com `_on_hardware_event`, aplicando debounce (~1–2 s),
  verificação de que o dispositivo de entrada selecionado mudou/sumiu e
  backoff exponencial limitado; publicar `RECONNECTING`/`DEVICE_LOST`;
- pausar automaticamente ao bloquear a sessão (`WTS_SESSION_LOCK`) e ao
  suspender; retomar só se o usuário tinha a escuta ativa;
- pausar/retomar manual pela tray e pela hotkey;
- shutdown limpo (ver "Ciclo de vida").

### Fase 2 — Detecção de voz e segmentação

VAD em cascata:

1. **Energia**: descarta silêncio absoluto a custo quase zero.
2. **Neural leve (Silero VAD ou WebRTC VAD)**: só nos blocos aprovados pela
   etapa anterior, para separar voz de teclado, ventilador e TV.

O VAD define os segmentos. Regras:

- início após `vad_min_speech_seconds` de fala;
- fim após `vad_end_silence_seconds` de silêncio;
- lookback (`vad_lookback_seconds`) e trailing pad (`vad_trailing_seconds`)
  retirados do ring buffer;
- corte forçado em `segment_max_seconds` (padrão 15 s, cabe no maior bucket do
  Parakeet), com sobreposição só nesse caso.

Comandos típicos (1–3 s) viram um único segmento curto — não há tamanho
mínimo de 5 s.

Pré-processamento (AGC, supressão de ruído) fica **desligado por padrão**:
AGC antes do VAD de energia amplifica ruído e a supressão pode piorar o
Whisper, treinado com áudio ruidoso. Só habilitar se o benchmark mostrar
ganho de WER ou de falsos acionamentos.

Eco: o `ChimePlayer` (e qualquer saída sonora futura) volta pelo microfone.
Marcar no ring buffer os intervalos em que o app está tocando som e ignorá-los
no VAD/wake word.

Cada segmento registra: início, fim, duração, motivo de encerramento
(silêncio, corte forçado, pausa, shutdown), nível de ruído e destino
(transcrito, descartado, interrompido). Esses metadados vão para
`telemetry.log`; o texto, não.

### Fase 3 — Transcrição

- uma única instância de modelo, compartilhada com o modo ditado;
- fila limitada (`agent_max_queue`); ao encher, descartar o segmento mais
  antigo **e registrar** o descarte;
- o consumidor espera o modelo com timeout (não `wait()` infinito); durante
  `fallback_device`/recarga, a fila pausa e o estado mostra "recarregando";
- falha latched (`RestartRequiredError`): agente vai para `ERROR`, captura
  vai para `PAUSED`, a UI informa — nunca continuar escutando sem conseguir
  transcrever;
- hot-swap (`inject_recovered_model`) só entre segmentos, sob o mesmo lock
  usado pelo consumidor, liberando o modelo anterior;
- medir latência e RTF por segmento (já existe para `WhisperNPU`; estender
  para `FasterWhisperCUDA` e Parakeet).

**Isolamento de processo (recomendado antes de habilitar por padrão):** rodar
a inferência em um processo worker. Um `DEVICE_LOST` ou travamento de driver
mata só o worker, que é reiniciado em outro dispositivo; o probe de
recuperação da NPU também roda isolado e com timeout. Isso substitui o latch
"reinicie o app" por recuperação real em um serviço de longa duração.

**Contexto entre segmentos:** `initial_prompt` com o texto anterior fica
**desligado por padrão**. O backend CUDA usa
`condition_on_previous_text=False` justamente para evitar loops de
alucinação. Avaliar apenas via benchmark, e só para fala longa cortada à
força.

**Junção de sobreposição:** quando houver corte forçado, a deduplicação do
trecho sobreposto exige timestamps de palavra (hoje desligados:
`return_timestamps=False` / `without_timestamps=True`) ou alinhamento de
texto. Definir e testar antes de usar overlap.

**Filtros de alucinação** (obrigatórios em escuta contínua):

- descartar quando `no_speech_prob` alto e `avg_logprob` baixo (CUDA expõe
  ambos; verificar o equivalente no OpenVINO GenAI);
- descartar por `compression_ratio` alto (repetição);
- lista de frases bloqueadas com normalização (caixa, pontuação, acentos),
  estendendo o conjunto atual com "legendas pela comunidade Amara.org",
  "inscreva-se", "legendado por" etc.;
- descartar texto cujo segmento tem duração incompatível com o número de
  palavras.

Modelos:

| Cenário | Modelo |
|---|---|
| Menor latência em NVIDIA | Whisper `turbo` (CUDA) |
| NPU equilibrado | Whisper `small` ou `base` |
| Baixo consumo / fallback | Whisper `tiny` ou `base` |
| Comandos curtos em NPU | Parakeet (segmentos ≤ 16 s, bucket de ~2 s) |
| Avaliação futura | Qwen3-ASR-0.6B |

`turbo`, `medium` e `large` **não** devem ser usados na NPU: o próprio loader
avisa que causam instabilidade (`DEVICE_LOST`).

Idioma: fixar o idioma configurado (`language`) em vez de `auto` por
segmento; detecção automática em trechos curtos é instável e mais lenta.

### Fase 4 — Ativação do agente

Ordem de avaliação:

1. **Push-to-talk** com a hotkey atual (Fase 0).
2. **Modo ativo temporário**: hotkey ativa o agente; ele escuta comandos até
   `agent_active_timeout_seconds` de silêncio.
3. **Wake word local**. Restrições:
   - Porcupine exige AccessKey com validação online — conflita com o
     requisito de processamento 100% local; só considerar se isso for
     aceito explicitamente;
   - openWakeWord não tem modelos prontos em PT-BR; exige treino de uma
     palavra própria;
   - o detector deve receber o lookback do ring buffer, senão o início da
     palavra é cortado pelo atraso do VAD. Detectores dedicados já são
     baratos; condicioná-los ao VAD economiza pouco e aumenta perdas —
     medir antes de decidir.
4. Desativação automática após silêncio.

Detectar a wake word pela transcrição não substitui um detector acústico:
é menos robusto e mantém o modelo grande rodando o tempo todo.

Indicadores no overlay/tray (reaproveitando `STATE_ICONS`): escutando,
agente ativo, processando, executando, aguardando confirmação, recarregando
modelo, erro/microfone indisponível. Escuta contínua sempre com indicador
visível.

### Fase 5 — Roteamento de intenções

Separar transcrição de interpretação. Representação:

```json
{
  "intent": "open_application",
  "arguments": { "name": "notepad" },
  "matched_rule": "abrir <app>",
  "requires_confirmation": false
}
```

O roteador começa com gramática determinística (padrões por idioma, PT-BR e
inglês) e correspondência aproximada **somente** contra os nomes da
allowlist. Não há score de confiança de modelo nesta fase; um LLM com saída
restrita a schema pode ser avaliado depois.

Intenções iniciais:

- abrir aplicativo permitido;
- abrir URL permitida;
- pesquisar texto (URL montada com encoding, domínio fixo);
- controlar o agente (pausar, ativar, cancelar);
- iniciar/parar ditado;
- responder com informação local (hora, estado do agente/dispositivo).

Respostas informativas aparecem no overlay. TTS fica fora do escopo inicial
(introduz eco; ver Fase 2).

A transcrição é entrada não confiável — inclusive áudio de TV, vídeos e
chamadas pode conter "comandos".

### Fase 6 — Executor seguro

- catálogo de comandos permitidos com argumentos tipados e validados;
- `subprocess` com lista de argumentos (`shell=False`), nunca concatenação;
- **processos lançados sem elevação**: como o app roda como Administrador,
  o executor deve iniciar processos com o token do usuário não elevado (por
  exemplo, via token do `explorer.exe` / `CreateProcessWithTokenW`, ou
  delegando ao shell do usuário). Executor que herda elevação não passa no
  critério de prontidão;
- timeout, cancelamento, captura de stdout/stderr e código de saída;
- limite de processos simultâneos;
- logs estruturados em `app.log` com intenção e resultado (sem o texto bruto
  fora do opt-in).

Bloquear ou exigir confirmação para: apagar/sobrescrever arquivos; executar
PowerShell, shell ou scripts; instalar pacotes; alterar configurações do
sistema; enviar mensagens; caminhos não validados; qualquer ação fora da
allowlist.

**Digitação também é execução.** No modo agente:

- `auto_enter` é sempre ignorado (nunca enviar `Enter`/quebra de linha);
- recusar digitação quando a janela em foco for terminal/shell
  (Windows Terminal, `conhost`, PowerShell, `cmd`, terminais de IDE) ou
  janela elevada, a menos que o usuário confirme.

Confirmação:

- mostra a ação interpretada, não a transcrição
  (`Vou abrir o Notepad. Confirmar?`);
- ações sensíveis confirmam por clique ou hotkey, não por voz — "sim" dito
  por terceiros ou pela TV não pode autorizar nada;
- timeout cancela por padrão.

Nunca usar `Invoke-Expression`, `eval`, `shell=True` ou execução direta da
saída de um modelo.

### Fase 7 — Memória e contexto

Sem memória persistente no início. O agente mantém apenas o necessário para a
intenção atual (último comando, confirmação pendente).

Histórico e logs:

- somente texto dirigido ao agente (ativado) entra no histórico
  (`self._history`) e no `Typed:` do `app.log`;
- fala ambiente é descartada em memória e nunca persistida;
- áudio nunca é salvo por padrão.

Se memória for necessária depois: armazenar só dados autorizados, separar
transcrição e comandos, oferecer limpeza pela interface e registrar retenção.

## Configuração proposta

Novas chaves, com padrões seguros, validadas em `validate_config` sem quebrar
arquivos existentes:

```json
{
  "agent_mode": false,
  "continuous_listening": false,
  "activation_mode": "push_to_talk",
  "agent_active_timeout_seconds": 8,
  "wake_word": null,
  "wake_word_sensitivity": 0.5,
  "vad_enabled": true,
  "vad_energy_threshold": 0.01,
  "vad_speech_threshold": 0.5,
  "vad_min_speech_seconds": 0.25,
  "vad_end_silence_seconds": 0.8,
  "vad_lookback_seconds": 0.5,
  "vad_trailing_seconds": 0.3,
  "segment_max_seconds": 15,
  "segment_overlap_seconds": 0.75,
  "ring_buffer_seconds": 30,
  "agent_max_queue": 2,
  "agent_model_wait_seconds": 10,
  "use_previous_text_prompt": false,
  "hallucination_no_speech_threshold": 0.6,
  "hallucination_logprob_threshold": -1.0,
  "hallucination_compression_ratio": 2.4,
  "preprocess_agc": false,
  "preprocess_noise_suppression": false,
  "pause_on_session_lock": true,
  "command_confirmation": "always",
  "allowed_commands": []
}
```

Padrões: agente desligado, push-to-talk, sem wake word, confirmação sempre,
allowlist vazia. Os limiares acima são pontos de partida, não valores medidos.

## Interface e ciclo de vida

Controles (tray e Settings): ativar/desativar agente; pausar escuta;
selecionar microfone, modelo e dispositivo; configurar ativação; ver comando
interpretado; confirmar/cancelar; ver último erro e dispositivo ativo
(incluindo fallback/recuperação em curso); limpar histórico.

Desligamento:

1. impedir novos segmentos;
2. cancelar probes de recuperação da NPU e timers;
3. aguardar (com timeout) ou cancelar a transcrição atual;
4. esvaziar e encerrar a fila;
5. parar o stream e o `ChimePlayer`;
6. liberar o modelo (e encerrar o worker, se houver);
7. remover hotkeys e handlers de mensagens da tray;
8. registrar no log que o agente terminou.

## Testes e benchmark

### Testes unitários

- transições das duas máquinas de estado;
- ring buffer (wrap-around, capacidade, lookback);
- VAD com silêncio, fala, ruído e áudio de chime;
- segmentação: fim por silêncio, corte forçado, sobreposição e junção;
- fila: limite, descarte registrado, timeout de espera do modelo;
- filtros de alucinação;
- roteamento de intenções em PT-BR e inglês;
- validação de argumentos, allowlist, confirmação e timeout;
- `auto_enter` ignorado e bloqueio de digitação em terminal;
- executor não herda elevação.

### Testes de integração

- captura real com dispositivo selecionado por horas;
- desconexão/reconexão de microfone (USB e Bluetooth) e resume de suspensão,
  incluindo rajadas de `WM_DEVICECHANGE`;
- `DEVICE_LOST` simulado durante transcrição: fallback, fila pausada e
  retomada; probe de recuperação e hot-swap entre segmentos;
- falha latched: agente em `ERROR` e escuta pausada;
- encerramento durante transcrição e durante probe;
- comando permitido e comando bloqueado.

### Métricas

Emitidas em `telemetry.log` com prefixo `[Telemetry]`:

- fim da fala → texto (p50/p95);
- fim da fala → intenção (p50/p95);
- RTF por backend/dispositivo;
- tamanho e atraso da fila, descartes;
- falsos acionamentos por hora;
- alucinações filtradas por hora;
- comandos não reconhecidos;
- WER separado em PT-BR e inglês;
- CPU, RAM do app, VRAM e uso de NPU em escuta ociosa e ativa
  (reaproveitar o loop de telemetria existente);
- reconexões e fallbacks por hora de escuta;
- execuções incorretas — deve ser zero nos testes de segurança.

Cenários: fala curta, fala longa, ruído, teclado, televisão, vídeo com fala,
múltiplos locutores e pausas naturais. Não usar apenas TTS para validar
ativação ou comandos.

## Segurança e privacidade

- escuta contínua sempre com indicador visível e pausa em um clique;
- pausa automática ao bloquear a sessão;
- áudio nunca salvo por padrão;
- fala ambiente nunca persistida em histórico ou log;
- comandos só da allowlist, executados sem elevação;
- digitação no modo agente sem `Enter` e bloqueada em terminais;
- ações sensíveis com confirmação fora do canal de voz;
- wake word e modo agente desligáveis;
- nenhuma dependência de validação online;
- erros sempre visíveis; nunca simular sucesso.

## Ordem recomendada

1. Resolver as pendências do baseline (`utils.logger`, liberação de modelo no
   hot-swap, timeout no `_model_ready.wait()`, debounce de eventos de
   hardware).
2. Fase 0: roteador determinístico + executor não elevado sobre push-to-talk,
   com testes de segurança.
3. Fase 1: ring buffer e captura contínua com reconexão e pausa por bloqueio
   de sessão.
4. Fase 2: VAD em cascata, segmentação e métricas por segmento.
5. Fase 3: fila, filtros de alucinação, integração com fallback/recuperação;
   benchmark de latência e falsos acionamentos.
6. Modo ativo temporário e indicadores de estado.
7. Worker de inferência em processo separado.
8. Avaliar wake word local.
9. Avaliar Qwen3-ASR, prompt com contexto e junção por timestamps.
10. Só depois: memória, contexto prolongado e comandos mais amplos.

## Critério de prontidão

O modo agente só pode ser habilitado por padrão quando:

- não houver execução arbitrária de texto;
- o executor comprovadamente não herdar privilégio de Administrador;
- ações sensíveis exigirem confirmação fora do canal de voz;
- a escuta puder ser pausada claramente e pausar ao bloquear a sessão;
- o shutdown for limpo, inclusive durante fallback/probe;
- a fila for limitada e os descartes registrados;
- `DEVICE_LOST` não deixar o agente escutando sem conseguir transcrever;
- p95 de fim da fala → intenção ≤ 1,5 s no dispositivo padrão;
- falsos acionamentos ≤ 1 a cada 8 h de escuta em ambiente doméstico com TV;
- comandos bloqueados forem testados automaticamente no CI;
- PT-BR e inglês tiverem resultados separados.
