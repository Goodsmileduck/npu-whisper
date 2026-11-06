# Plano de distribuição de hardware

Data: 2026-10-10

Status: proposta

## 1. Objetivo e estado atual

Planejar a execução a cada início da Débora, distribuindo STT, LLM local e TTS
entre os dispositivos compatíveis. Cada componente recebe um destino e uma
cadeia de fallback própria. Usar todos os aceleradores não é um objetivo em si:
importam latência, memória e estabilidade. Este documento não altera código.

As referências abaixo são relativas à raiz do repositório. `GPU` significa o
dispositivo OpenVINO, normalmente a GPU Intel; `CUDA` significa NVIDIA. O plano
deve guardar também a identidade física e o índice, sem confundir os dois.

| Componente | Onde roda hoje | Lacuna |
|---|---|---|
| STT Whisper | `debora_whisper/dictation_engine.py:select_device` escolhe o primeiro disponível em `device_priority = [CUDA, NPU, GPU, CPU]`. `create_model` cria `FasterWhisperCUDA` em CUDA ou `WhisperNPU` com OpenVINO GenAI em NPU/GPU/CPU. | A prioridade não considera outros componentes, memória livre ou latência medida. |
| STT Parakeet | `debora_whisper/dictation_engine.py:ParakeetNPU._load_pipeline` compila os buckets do encoder no dispositivo escolhido; decoder tenta GPU e depois CPU. Pré-processamento em CPU. | O decoder tenta GPU até quando `device=CPU`. Falha pode provocar recarga de todo o STT. Não existe backend CUDA para Parakeet. |
| LLM local | `debora_whisper/voice_chat.py:start_llm` inicia processo separado; `debora_whisper/llm_server.py:load` usa OpenVINO GenAI em `llm_device` (padrão GPU), depois CPU. | Fallback de carga próprio, sem reserva compartilhada; `generate` informa erro de inferência sem migrar o modelo. Não há backend CUDA de LLM implementado. |
| TTS | `debora_whisper/voice_chat.py:ensure_tts_server` gerencia servidor HTTP local, quando aplicável. `debora_whisper/tts_server.py:load_model` carrega Chatterbox em CUDA, ou CPU se CUDA não estiver disponível no PyTorch desse servidor. | O ambiente do TTS é separado do STT. CUDA disponível para faster-whisper não prova CUDA utilizável pelo TTS; OOM e perda em execução não têm cadeia coordenada. |
| Claude | `debora_whisper/voice_chat.py:VoiceChat._generate` chama `debora_whisper/harness.py:start_harness`. | Não há inferência de LLM local para distribuir; o processo do harness e suas ferramentas ainda consomem CPU/RAM. |

Com RTX + NPU, Whisper tende a ir para a RTX e deixar a NPU ociosa. No chat,
STT e TTS podem disputar a RTX; **o LLM atual usa a GPU Intel, não a RTX**.
LLM na NVIDIA seria outro projeto. CPU, iGPU e NPU também disputam RAM,
largura de banda e energia do sistema.

`MODEL_REGISTRY.preferred_device` é apenas o selo apresentado por
`debora_whisper/ui/settings.py:SettingsWindow._build_model_list`; não comprova
suporte nem decide a alocação. `device_supports_model` só exclui CUDA para
Parakeet. `detect_devices` verifica OpenVINO e faster-whisper, mas não monta
uma matriz de capacidades por componente.

## 2. Quando cada carga fica ativa

| Modo | STT | LLM | TTS | Prioridade |
|---|---|---|---|---|
| Ditado | Rascunhos e transcrição final | Inativo | Inativo | Tempo até a primeira palavra/rascunho útil, sem prejudicar o texto final. |
| Chat local | Escuta, transcrição e interrupção da resposta | Gera resposta local | Sintetiza frases enquanto o LLM continua | Tempo até o primeiro áudio; preservar STT para interrupção. |
| Chat Claude | Igual ao chat local | Serviço Claude; sem reserva de acelerador local | Continua local, salvo URL externa | Primeiro áudio e STT; espera de rede não é falha de GPU. |

Hoje `debora_whisper/dictation_engine.py:DictationApp.set_voice_chat` mantém LLM
e TTS carregados ao desligar o chat. O plano deve contar memória residente mesmo
sem inferência e liberar componentes ociosos, em limite seguro de turno, quando
necessário. Ao mudar modo/backend, recalcular reservas; no início em ditado,
preparar apenas um plano latente para o chat, sem carregar seus modelos.

## 3. Sondagem rápida em cada início

Orçamento proposto: **no máximo 750 ms adicionais no caminho até o primeiro
ditado**, incluindo leitura do cache e espera pela sondagem. Não é promessa de
inicialização total: download e carga do modelo escolhido têm custos próprios.
Não compilar modelos nem executar benchmarks nesse caminho.

Partir de `debora_whisper/dictation_engine.py:detect_devices` e `has_nvidia_gpu`,
mas colocar consultas potencialmente bloqueantes em processo auxiliar, com
prazo comum contado desde antes de iniciá-lo. Uma thread não protege contra
driver que trava segurando o GIL. Ao vencer o prazo, usar resultados parciais e
um plano conservador; não esperar o encerramento de um driver travado.

- CPU e RAM disponível via `psutil`; dispositivos e nomes via OpenVINO.
- Uma consulta `nvidia-smi` para identidade/UUID, driver e VRAM total/livre por
  placa. Ausência, saída inválida ou timeout significam informação desconhecida.
- NPU presente, propriedades de driver expostas pelo OpenVINO e histórico de
  falhas. Enumerar a NPU não prova que consegue inferir: distinguir presente,
  verificada e em quarentena. Sem validação compatível anterior, sua utilização
  depende de confirmação em segundo plano; CPU/iGPU/CUDA atendem primeiro.
- Compatibilidade por componente, backend, modelo/export e precisão; arquivos
  locais completos e dependências já instaladas. Consultar capacidades do
  ambiente real do TTS, sem importá-lo no processo da interface. Separar
  “suportado” de “pronto para carregar”; fallback imediato exige artefatos locais.

Persistir inventário estável, capacidades, picos de memória e medições em cache
versionado sob `~/.debora`. Chave: identidade dos dispositivos, versões dos
drivers disponíveis, OpenVINO/GenAI, CTranslate2/faster-whisper, PyTorch/TTS,
modelo/export/revisão/precisão e versão da política. Usar metadados de artefatos
já conhecidos; não calcular hash de gigabytes no início. Driver sem versão
consultável fica como desconhecido: cache curto (24 h) e nova validação isolada,
sem tratá-lo como prova permanente de saúde. Cache corrompido é descartável.

Atualizar presença e memória livre a cada início e antes de cada carga; nunca
reutilizar VRAM livre antiga como autorização. Mudança de versões invalida
compatibilidade e medições correspondentes. Histórico de perda tem precedência
sobre cache positivo, aproveitando `remember_npu_loss`/`avoid_lost_npu`.

Opcionalmente medir um pequeno trecho de fala de teste, sem dados do usuário,
depois que o STT estiver pronto e o app ocioso. Limitar a 5 s de trabalho ativo,
em processo descartável; parar diante de demanda de áudio. Sem compilação fria
ou download para benchmark. `debora_whisper/npu_probe.py:probe_npu` já oferece
isolamento e timeout, mas seu limite atual de 300 s pertence à recuperação,
nunca à sondagem de início. Silêncio confirma execução, não desempenho nem
estabilidade em fala real.

## 4. Regras de distribuição

Entradas: modo, backend do chat, escolhas explícitas, modelo/idioma, matriz de
suporte, memória livre e reservada, histórico de falhas, estimativas de latência
e custo de recarga. Implementar primeiro uma função pura e determinística;
medições refinam as regras, sem exigir um otimizador complexo.

1. Filtrar destinos ausentes, incompatíveis, sem dependências/artefatos ou em
   quarentena. Whisper: CUDA/NPU/GPU/CPU. Parakeet: encoder NPU/GPU/CPU e
   decoder GPU/CPU. LLM: GPU/CPU inicialmente; NPU só com export e combinação
   de versões validados. TTS: CUDA/CPU. Claude: LLM local desativado.
2. Aplicar escolhas explícitas compatíveis. Para cada destino, estimar memória
   de pesos, execução e pico temporário; incluir KV cache/contexto do LLM e todos
   os buckets residentes do Parakeet. Exigir que **novas reservas caibam na
   memória livre menos margem de `max(512 MiB, 15% do total)`**. São valores
   iniciais ajustáveis. Não descontar duas vezes modelos já residentes. Memória
   desconhecida impede admitir várias cargas grandes no mesmo acelerador.
3. Em ditado, escolher o STT com menor latência esperada até texto útil; sem
   medição comparável, usar `device_priority` como desempate. Não deslocar STT
   rápido para NPU apenas para ocupá-la. Manter LLM/TTS inativos.
4. Em chat, reservar CUDA para TTS quando utilizável e priorizar STT na NPU
   validada; LLM local na iGPU, com CPU como alternativa. Se a NPU não atender,
   comparar STT em iGPU com compartilhar CUDA: permitir compartilhamento só
   com memória suficiente e latência aceitável. Sem dados, preferir separação.
   Parakeet exige contabilizar também a disputa entre decoder e LLM na iGPU.
5. Para empates, preferir destino já carregado, menor contenção e ordem do
   usuário. CPU é o último destino compatível, não uma garantia de RAM infinita.
   Se nada couber, manter componentes saudáveis e informar qual está indisponível;
   não trocar modelo ou quantização silenciosamente.

Saída proposta: plano com geração/id, modo e, por componente, backend/modelo,
dispositivo físico preferido, dispositivo efetivo, fallbacks ordenados, reserva,
motivo e confiança da estimativa. Parakeet terá entradas para encoder e decoder;
WhisperPipeline e faster-whisper continuam unidades indivisíveis. Não há API
existente aqui para repartir as camadas do Whisper entre aceleradores.

As cadeias passam pelos mesmos filtros e são revalidadas antes de cada uso.
Não reservar memória para todas as alternativas ao mesmo tempo. Uma troca
recalcula a reserva do componente afetado; não redistribui os demais por efeito
colateral. Pressão de memória pode adiar sua carga ou escolher outro fallback.

### Exemplos de planos

As setas mostram preferido e fallbacks; pressupõem modelos locais, suporte
validado e memória suficiente. São escolhas iniciais, não resultados de benchmark.

| Máquina e modo | STT | LLM local | TTS |
|---|---|---|---|
| RTX + NPU + iGPU, ditado Whisper | CUDA → NPU → GPU → CPU | Inativo | Inativo |
| RTX + NPU + iGPU, chat local Whisper | NPU → GPU → CUDA, se houver folga → CPU | GPU → CPU | CUDA → CPU |
| RTX + NPU + iGPU, chat Claude | NPU → GPU → CUDA, se houver folga → CPU | Desativado; Claude externo | CUDA → CPU |
| NPU + iGPU, ditado Whisper | NPU → GPU → CPU | Inativo | Inativo |
| NPU + iGPU, chat local | NPU → GPU → CPU | GPU → CPU | CPU; limitar concorrência |
| NPU + iGPU, chat Claude | NPU → GPU → CPU | Desativado | CPU; iGPU disponível ao STT |
| Somente CPU, ditado | CPU | Inativo | Inativo |
| Somente CPU, chat local / Claude | CPU | CPU no local; desativado no Claude | CPU; geração/síntese com fila limitada |

Para Parakeet na máquina completa: encoder NPU → GPU → CPU; decoder GPU → CPU;
TTS CUDA → CPU. No chat local, preferir decoder CPU se o LLM ocupar a iGPU e a
medição confirmar latência aceitável. Esse controle separado ainda precisa ser
implementado. Na máquina CPU, encoder **e** decoder devem obedecer CPU.

Manter captura/VAD responsivos e filas limitadas em todos os casos. A proteção
`debora_whisper/dictation_engine.py:_inference_gate`, usada em
`DictationApp._finish_recording` e `DictationApp._load_voice_chat`, evita sobrepor
a compilação do LLM na iGPU durante o warm-up e a inferência na NPU, combinação
já associada a perda de dispositivo. Essa proteção não cobre todos os caminhos:
`debora_whisper/voice_chat.py:generate_reply` chama `load_llm` diretamente e,
se o processo tiver encerrado, pode reiniciá-lo e compilar via `start_llm` sem
adquirir o gate, enquanto o STT transcreve novo áudio. Preservar o gate atual e
implementar agendamento coordenado em **todos** os caminhos de carga/reinício
do LLM antes de contar com essa exclusão como garantia geral. Validar qualquer
substituição do gate pelo supervisor. Adiar compilação do LLM para depois da
prontidão do STT e intervalo ocioso; não presumir que processos separados
eliminam conflito de driver.

## 5. Fallback por componente

Hoje `WhisperNPU._load_pipeline` cai diretamente para CPU em erro benigno;
Parakeet já tem alternativas distintas na carga do encoder/decoder. Em execução,
`debora_whisper/dictation_engine.py:DictationApp._error_payload` põe o modelo
perdido em quarentena e `debora_whisper/app.py:GUIApp._update_ui` escolhe outro
dispositivo, recarregando todo o STT via `DictationApp.fallback_device`.
`tests/test_device_failure.py:test_gpu_failure_during_transcription_falls_back`
confirma que esse caminho já não fixa a falha global de GPU. Ainda existem
`record_device_failure`, `ensure_devices_usable` e tratamento de
`RestartRequiredError`; não confundir seus comentários com o fluxo ativo.

Propor um supervisor com estado por componente: pronto, carregando, degradado,
em quarentena ou indisponível. Reutilizar o isolamento do LLM/TTS e introduzir
processo de STT; para preservar encoder quando o decoder falhar fatalmente,
separar também seus contextos/processos. Enquanto compartilham contexto, não
prometer recuperação independente segura: reiniciar a unidade STT afetada,
preservando LLM e TTS. Não remover proteções de driver antes desse isolamento.

| Gatilho | Ação |
|---|---|
| Modelo/operação incompatível, falha de carga | Marcar a combinação componente/modelo/versões/dispositivo como incompatível e seguir a cadeia; não repetir a cada turno. Arquivo corrompido é erro de artefato, não do hardware. |
| OOM recuperável | Liberar somente recursos ociosos do componente, atualizar memória e tentar uma vez; persistindo, seguir a cadeia. Não reduzir modelo/contexto contratado silenciosamente. |
| `DEVICE_LOST`, contexto CUDA/OpenCL fatal | Quarentenar imediatamente o contexto; não chamar inferência nem destrutores nativos nele. Encerrar trabalhador isolado e carregar a alternativa em contexto novo. |
| Dois timeouts consecutivos | Reiniciar trabalhador do componente e seguir a cadeia. Usar limites separados de carga e inferência, escalados por duração de áudio/tokens e medições; erro de rede/HTTP não prova falha de hardware. |

Uma falha do TTS não muda STT/LLM; falha do decoder não muda o encoder saudável
quando isolado. Se houver perda comprovada de um dispositivo físico compartilhado,
cada componente que o utiliza precisa de verificação própria; não migrar todos
por simples associação. Falha `UNKNOWN` bloqueia os contextos candidatos da
operação afetada, sem escolher novamente o mesmo dispositivo às cegas.

Reavaliar retorno após 30, 60, 300 e 900 s, aproveitando
`debora_whisper/app.py:GUIApp._schedule_npu_recovery`. Um probe por vez, somente
ocioso, fora do processo da interface. Quatro falhas suspendem tentativas na
sessão; novo início consulta o histórico antes de tentar. Para OOM, exigir folga
em duas amostras separadas por 5 s. Incompatibilidade só volta a ser candidata
após mudança de versão/modelo ou solicitação explícita.

Voltar ao preferido apenas após carga e inferência no dispositivo real, sem
fallback silencioso, e em limite de turno com reserva para a troca. Aplicar pelo
menos 60 s de estabilidade antes de outra promoção. Reutilizar as verificações
de atividade e encerramento de `DictationApp.inject_recovered_model`, que hoje
não recebe snapshot da configuração nem identificador de geração. A guarda em
`GUIApp._swap_to_recovered_npu` compara apenas a identidade do engine, capturada
por `GUIApp._run_npu_recovery_probe` **depois** do probe; uma troca em Settings
durante a recuperação ainda pode instalar um modelo antigo no engine novo.
Implementar como trabalho novo a captura da identidade do engine e de um
snapshot com geração do plano/configuração **antes** do probe. Validar ambos
atomicamente com a instalação, inclusive nas tentativas adiadas, rejeitando
resultados obsoletos e impedindo troca de engine/configuração entre validação
e instalação. Se não houver memória para duas instâncias, permanecer no
fallback até uma recarga ociosa planejada.

Preservar áudio pendente em fila limitada e tentar o segmento STT uma única vez
na alternativa, sem repetir texto já inserido. LLM não repete trechos já enviados;
TTS retoma apenas frases ainda não reproduzidas. Não reenviar automaticamente
uma ação ao harness Claude por causa de uma falha local de síntese.

### O que o usuário vê

Registrar decisão e transição em `app.log`, por exemplo:
`component=tts device=CUDA fallback=CPU reason=OOM plan=7`.
Mostrar “Voz em CPU; ditado na NPU” pelo aviso existente
`DictationApp._notice`, callbacks de `GUIApp._update_ui` e informações de
`debora_whisper/ui/tray.py:TrayManager.update_info`. Um componente degradado não
transforma todo o app em erro. Sem novo mecanismo nativo de notificações.

`debora_whisper/dictation_engine.py:DictationApp._monitor_resources` hoje coleta
CPU/RAM e consulta VRAM por `nvidia-smi` **somente se `config.device == CUDA`**.
Nesta branch não há consulta de memória NPU nessa função. Ampliar para todos os
dispositivos usados, inclusive TTS em CUDA com STT na NPU: total/livre, reservas,
estado por componente, latência, motivo da troca e próxima tentativa em
`telemetry.log`. Consultar memória NPU via propriedades OpenVINO apenas quando
expostas; caso contrário, registrar indisponível. Memória de NPU/iGPU pode ser
compartilhada com RAM: não somar como se fossem bancos independentes. VRAM global
não identifica consumo por componente; reservas continuam sendo estimativas.

## 6. Configuração e limites

Adotar `device="auto"` e `llm_device="auto"` como padrões novos; adicionar
`tts_device="auto"` para servidor gerenciado. `auto` é política da Débora, não
o plugin `AUTO` do OpenVINO. Separar preferência persistida de dispositivo efetivo
para que um fallback não sobrescreva a intenção do usuário.

`--device` prevalece sobre `device` salvo; uma escolha concreta fixa a preferência
do STT, inclusive encoder/decoder em CPU quando solicitado. `llm_device` só afeta
LLM local. Falhas ainda permitem fallback compatível, informado na interface;
configuração impossível, como Parakeet/CUDA ou LLM/CUDA atual, recebe erro claro.
Modelo e idioma permanecem escolhas do usuário. `device_priority` vira desempate
e ordem residual das alternativas em auto; escolha explícita tem precedência.

Migrar com versão de configuração: preservar valores explícitos existentes.
Hoje `debora_whisper/app.py:main` chama `apply_device_priority` na ausência de
`--device`, e `config.device` também guarda estado efetivo; arquivos antigos não
identificam com segurança uma preferência manual. Preservar o valor e explicar
a opção Auto, em vez de inferir intenção pelo valor padrão. Aplicar a mesma
semântica na CLI e em `SettingsWindow._get_new_config`.

URL TTS externa ou `tts_server_command` personalizado fica sob controle externo;
não encerrar nem reposicionar esse serviço. Só usar destino/reserva informados
por ele, ou marcar desconhecido. Falta de pacote significa capacidade ausente:
**nunca executar pip nem instalar dependências em runtime**. O comando atual em
`voice_chat.py:tts_command` usa `uv run --script`; a integração do plano deve
exigir ambiente TTS previamente preparado, sem resolver/instalar pacotes durante
sondagem ou fallback.

Não adicionar `ctypes.windll`, DWM, registro ou APIs win32. Usar subprocessos,
OpenVINO, `psutil` e callbacks já existentes. A marca de boot atual em
`dictation_engine.py:_boot_time` usa API Windows; reaproveitar seu contrato sem
ampliar essa dependência, com estado desconhecido conservador onde não existir.

## 7. Entrega em PRs pequenos

PR 2 entregou o planejador em modo apenas registrado (log-only); a integração da configuração Auto fica para o PR 3.

| PR | Escopo | Testes focados |
|---|---|---|
| 1 | Inventário com prazo, cache versionado e matriz por backend; ainda sem mudar a escolha. | `tests/test_device_priority.py`: consulta travada, múltiplas GPUs, driver desconhecido, invalidação/cache corrompido, ambientes STT/TTS diferentes e nenhuma instalação. |
| 2 | Planejador puro, reservas, exemplos acima e configuração Auto. | Tabelas de casos para cada máquina/modo, VRAM insuficiente, RAM compartilhada, override, migração legada, Parakeet sem CUDA e Claude sem LLM local. |
| 3 | Aplicar plano no STT e tornar encoder/decoder configuráveis em carga; manter proteções existentes. | `tests/test_device_failure.py`: dispositivo efetivo, cadeias esgotadas, CPU obedecida por ambos, erro atribuído ao decoder e quarentena sem nova chamada ao objeto perdido. |
| 4 | Integrar LLM/TTS e reservas por modo usando processos já existentes. | `tests/test_llm_process.py` e `tests/test_voice_chat.py`: OOM/timeout de um serviço preserva os outros, memória residente, serviço externo, ausência de CUDA no TTS e Claude sem recarga/reenvio. |
| 5 | Isolar STT, depois encoder/decoder; supervisor de falhas e retorno com back-off. | `tests/test_device_failure.py`: trabalhador travado, recuperação falsa em CPU, troca durante transcrição, geração obsoleta, preservação do encoder e ausência de texto/áudio duplicado. Dividir em PRs menores se necessário. |
| 6 | Telemetria, resumo no tray/overlay e ativação gradual do padrão Auto. | Callbacks sem desktop real, VRAM com STT fora de CUDA, memória NPU indisponível, teto de 750 ms e sessões reais nas três classes de máquina. |

Validar testes focados em cada etapa e suíte completa nas mudanças de integração.
Iniciar com plano apenas registrado para comparar decisões; depois habilitar Auto
por opção, e torná-lo padrão para novas configurações após validar latência e
recuperação. Manter possibilidade de voltar à política anterior durante rollout.

## 8. Riscos e perguntas abertas

- Turbo/NPU: o incidente documentado em
  [npu-device-lost-2026-10-06.md](diagnostics/npu-device-lost-2026-10-06.md)
  isolou falhas no export FluidInference int4; o registro atual usa OpenVINO
  int8, com 80/80 inferências no ensaio citado. Não proibir todo turbo/NPU nem
  tratar esse ensaio como garantia universal. Compatibilidade precisa incluir
  export e driver; silêncio não reproduziu o problema original.
- Quais picos de memória e metas de primeira palavra/primeiro áudio são
  aceitáveis para cada modelo? Medir uso real antes de ajustar margens e regras;
  CPU-only pode funcionar com chat lento, sem promessa de tempo real.
- Qual o custo de transportar tensores encoder/decoder entre processos? Medir
  antes de habilitar essa separação por padrão. Timeout de processo limita a
  espera da interface, mas não garante recuperação de um driver travado.
- A compilação LLM/iGPU pode bloquear STT/NPU mesmo após isolamento. Até validar
  cancelamento e agendamento seguros, preservar a exclusão no warm-up, estendê-la
  a todos os caminhos de carga/reinício e mostrar “preparando chat”.
- Fallback CUDA → OpenVINO exige outro export local do mesmo modelo; como
  preparar esses artefatos sem download no momento da falha e sem duplicação
  excessiva em disco? Tornar essa preparação explícita no setup.
- Propriedades de memória/driver NPU variam; dados desconhecidos precisam de
  tratamento conservador. Concorrência externa pode invalidar qualquer reserva
  entre sondagem e carga, portanto OOM continua sendo caminho normal de fallback.
