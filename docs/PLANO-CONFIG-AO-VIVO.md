# Plano: configurações ao vivo pelo chat de voz

Data: 2026-10-13

Status: proposta revisada; recuperação do plano original ainda parcial

Objetivo: pedir à Débora dentro do Claude Code que altere a Débora local —
por exemplo, desligar `inline_drafts`, trocar o modelo ou escolher outra voz —
sem abrir Settings nem reiniciar o aplicativo nos casos suportados. A resposta
deve distinguir mudança aplicada, aguardando um ponto seguro e impossível sem
reinício. Esta entrega contém apenas documentação.

## Recuperação: o que foi encontrado

**Não foi localizado um plano completo específico de configuração ao vivo.**
O antecedente mais concreto recuperado é a proposta de **auto fix de 09/10**,
na sessão Claude Code `deab5da1-1aa4-4480-afc5-7c01172eb825`, posteriormente
incorporada ao código. Ela é a base histórica desta revisão, sem atribuir-lhe
uma arquitetura MCP ou de recarga que o registro não contém.

| Fonte | Evidência e uso nesta revisão |
|---|---|
| Sessão `deab5da1-1aa4-4480-afc5-7c01172eb825` | No arquivo `.jsonl`, linha 179, o usuário aceita acesso a `~/.debora` para auto fix. Linhas 189 e 247: localizar configuração/logs, explicar a alteração, guardar backup e avisar que aplicar/reiniciar ainda seria necessário. |
| [Commit `75b8119`](https://github.com/alexandre-machado/debora-whisper/commit/75b8119dc2f6aca19c9c50d71743b0c34e341c94), 09/10 | `Add Claude harness self-repair context and deduplicate permission denials`: implementa `_self_prompt`, acesso adicional ao diretório e as instruções de auto fix em `harness_prompt.md`. É a evidência versionada mais direta. |
| Memória `memory/harness-debora-dir-access.md` | Origem na mesma sessão; registra que o usuário aceita leitura/edição de `~/.debora`. O simples acesso a essa pasta não deve reaparecer como impedimento ou nova pergunta de autorização. |
| [VOICE_HARNESS_IDEA.md](VOICE_HARNESS_IDEA.md), revisão no [commit `b1072bd`](https://github.com/alexandre-machado/debora-whisper/commit/b1072bd) | Candidato arquitetural mais completo: Claude como inteligência, stream-json persistente, voz na Débora e fase de permissões. A troca de configuração reiniciar o harness no próximo turno não equivale a detectar edições externas de `config.json`. |
| Versão anterior no [commit `8d21599`](https://github.com/alexandre-machado/debora-whisper/commit/8d21599), também preservada em branches antigas | Propunha Qwen como roteador e um **cliente** MCP genérico. Não propunha um servidor de configuração da Débora. Foi substituída pelo desenho de 09/10. |
| [CONTINUOUS_AGENT_PLAN.md](CONTINUOUS_AGENT_PLAN.md), [PLANO-PERMISSOES-CHAT-VOZ.md](PLANO-PERMISSOES-CHAT-VOZ.md) e [PLANO-TELEMETRIA-NA-INTERFACE.md](PLANO-TELEMETRIA-NA-INTERFACE.md) | Outros candidatos examinados: fornecem estados de escuta, política de ações sensíveis, logs/UI e regras de threads; nenhum descreve o controle ao vivo pedido aqui. |

As sessões e memórias acima ficam em
`C:/Users/alexandre-machado/.claude/projects/D--repos-alexandre-machado-debora-whisper/`.
Identificação exata da autorização: sessão `deab5da1-1aa4-4480-afc5-7c01172eb825`,
2026-10-09T17:04:55.981Z, linha 179. A proposta aparece às 17:05:07.136Z,
linha 189; o relato da implementação, às 17:10:46.066Z, linha 247.
Essas referências locais não dependem de publicar os transcripts privados.

A busca percorreu `docs/`, inclusive `docs/notes`, nas 51 referências locais
enumeradas por `git for-each-ref refs/remotes/origin refs/heads`, além do
histórico de commits, sessões `.jsonl` (inclusive subagentes) e `memory/`.
Foram procurados termos de configuração, tempo real, MCP, reload, watcher,
settings e auto fix. `main`, `origin/main` e este worktree apontavam para
[`d5015e1`](https://github.com/alexandre-machado/debora-whisper/commit/d5015e131b9b7268f063921b201ddad4db3fd2d5).
Referências remotas aqui significam as cópias disponíveis localmente, sem fetch.

**Limitação da recuperação:** `gh issue list --state all` e
`gh pr list --state all` falharam no proxy local (`127.0.0.1:9`); a tentativa
pela web também não recuperou as páginas. Corpos e comentários de issues/PRs
não puderam ser revisados nesta execução. A sessão
`60e29727-87b0-4878-b202-00d714271ca9`, mensagem de 13/10 às 15:13:11.432Z,
também relata não ter encontrado o plano, mas esse relato não substitui uma
consulta atual ao GitHub. Confirmar a fonte específica continua em aberto;
as decisões novas abaixo estão identificadas como propostas desta revisão.

## Como funciona hoje

Revisão baseada no código de `d5015e1`:

- [dictation_engine.py](../debora_whisper/dictation_engine.py): `load_config`
  mescla JSON com `DEFAULT_CONFIG` na inicialização. Não há watcher nem comando
  de recarga. `save_config` grava o arquivo inteiro diretamente, sem substituição
  atômica, controle de versão ou rollback. Uma edição externa pode ser perdida
  no próximo salvamento do app.
- `validate_config` cobre parte das chaves, mas não é um contrato de entrada
  para ferramentas: não rejeita chaves desconhecidas, deixa campos sem tipo/faixa
  e admite números inadequados em alguns caminhos. A validação de
  `sample_rate`, por exemplo, aceita `True` e infinito; não basta reutilizá-la
  sem ampliar o esquema.
- [app.py](../debora_whisper/app.py), `_on_settings_apply`: valida, calcula
  diferenças, usa `stop_if_idle` antes de reconstruir, atualiza o dicionário
  compartilhado e salva. Atualiza o overlay e, quando necessário, recria o
  engine em segundo plano. Não retorna um resultado estruturado; não restaura
  automaticamente a configuração anterior se salvar/carregar falhar.
- `_REBUILD_KEYS` contém exatamente `model_size`, `device`, `hotkey`,
  `voice_chat_hotkey`, `beep_on_start`, `sample_rate` e `max_record_seconds`.
  Reconstruir o engine normalmente não exige reiniciar o processo. Com
  `device_failure()` latched, o app conserva configurações para a próxima
  inicialização e evita recarregar no contexto de hardware danificado.
- `DictationApp`, `AudioRecorder` e [VoiceChat](../debora_whisper/voice_chat.py)
  recebem o mesmo dicionário; algumas opções são lidas durante o uso, outras
  ficam em atributos ou processos. `AudioRecorder` captura limiares de VAD no
  construtor; `_continuous` não acompanha uma simples atribuição no dicionário.
  `VoiceChat.respond` combina leituras ao longo do turno e cópias por síntese;
  hoje é possível mudar comportamento no meio de uma resposta.
- `_set_voice_chat` e `DictationApp.set_voice_chat` já aplicam troca de modo,
  aquecimento, interrupção e sincronização visual. É preciso preservar
  `start_in_dictation`, `_saved_voice_chat` e `set_voice_chat_config`: preferência
  salva e modo efetivo ao iniciar não são sempre iguais.
- [harness.py](../debora_whisper/harness.py): `harness_key` determina reutilização
  do subprocesso. Inclui idioma, pasta, modelo, permissões, memória, ferramentas
  e identificação de sessão. A lista `harness_changed` em `app.py` é menor:
  não inclui todos esses campos. `_turn_lock` serializa o protocolo; o handler
  atual de permissão é síncrono e não está ligado a confirmação por voz.
- [ui/settings.py](../debora_whisper/ui/settings.py): a janela copia a configuração
  ao abrir e `_get_new_config` devolve um snapshot inteiro. `_apply` atualiza sua
  cópia antes de saber se o app aceitou a operação. Só há sincronização externa
  específica de `voice_chat`. Uma janela antiga pode desfazer mudanças por voz.
  O dispositivo é recalculado por `select_device`; o seletor de microfone não
  persiste uma chave de entrada em `_get_new_config`.
- Posição/largura do balão e troca de modo também chamam `save_config`.
  Centralizar só o botão Apply deixaria escritores concorrentes de fora.

O contexto de auto fix continua dizendo que se pode editar `config.json`, mas
`harness_command` adiciona `--disallowedTools Edit(...)` para configuração e
backups. O próprio comentário reconhece que shell pode contornar essa defesa.
É necessário reconciliar o prompt com a API proposta, sem apresentar essa regra
de arquivo como isolamento completo.

## Mecanismo recomendado

**Um serviço de configuração no processo da Débora, exposto ao harness por
ferramentas MCP locais.** Settings, bandeja, hotkeys e geometria usam o mesmo
serviço. A Débora decide e aplica; o modelo só apresenta pedidos tipados.

| Alternativa | Avaliação |
|---|---|
| Observar `config.json` e aplicar qualquer alteração | Não recomendado como canal de comando: não identifica intenção/sessão, não confirma valores, encontra JSON parcial e disputa salvamentos com a UI. |
| IPC próprio mais comando de shell | Permite um núcleo pequeno, mas exige que o harness use shell, aumenta a superfície de permissões e perde descoberta/esquema nativos das ferramentas. Pode ser adaptador futuro do mesmo serviço. |
| MCP por stdio, com ponte para o app | Viável, mas o processo MCP iniciado pelo Claude precisaria de um segundo IPC para alcançar o engine já aberto. Não confundir seu stdio com o stream-json do harness. |
| MCP via Streamable HTTP em loopback, hospedado pelo app | Recomendado para a prova inicial: alcança a instância existente diretamente e elimina a ponte. Exige autenticação, limites e ciclo de vida explícitos. |

O Claude documenta servidores locais e `--mcp-config`; o transporte deve ser
registrado apenas no subprocesso iniciado pela Débora, sem editar configurações
globais ou `.mcp.json` do projeto. Não usar `--strict-mcp-config` de modo a
eliminar conectores existentes. A compatibilidade precisa ser demonstrada com
o CLI instalado, incluindo modo headless, autenticação e confirmação.
Referências: [MCP no Claude Code](https://code.claude.com/docs/en/mcp) e
[referência do CLI](https://code.claude.com/docs/en/cli-reference).

O endpoint escuta somente em `127.0.0.1`, em porta efêmera, com capacidade/token
aleatório por execução e vínculo do host à sessão do harness. Autenticar também
leituras; validar Host/Origin, rejeitar origens de navegador não autorizadas,
limitar corpo, frequência e duração. O token não aparece em argumentos do
modelo, respostas, URL, logs ou `config.json`; provar passagem por ambiente ou
canal privado ao processo. Metadados temporários têm acesso restrito ao usuário
e limpeza na saída. Não há endpoint de shell, leitura arbitrária de arquivos
ou carregamento de código. Essas exigências seguem o
[transporte MCP](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports).

Loopback e token evitam clientes casuais/remotos; não isolam um processo hostil
do mesmo usuário com acesso a arquivos e ambiente. A confirmação dentro do app
continua necessária, inclusive com `acceptEdits` ou `harness_permission_response`
permissivos. A capacidade de chamar uma ferramenta não autoriza sua alteração.
Uma regra organizacional que bloqueie a ferramenta continua prevalecendo.

O servidor vive fora do engine e sobrevive à reconstrução dele. Uma fila leva
pedidos à coordenação do app; a thread Tk consome eventos via `root.after`.
Carregamento, persistência e espera não bloqueiam Tk nem callbacks de áudio.
Preferir biblioteca MCP mantida, declarada em `pyproject.toml`; nada de instalar
dependências em runtime. Sem novas APIs Windows no núcleo de configuração.

### Contrato proposto das ferramentas

Nomes abaixo são novos, não ferramentas já implementadas:

| Ferramenta | Resultado/efeito |
|---|---|
| `get_settings(keys)` | Apenas chaves autorizadas: valores persistidos, efetivos, pendentes, revisão e restrição de aplicação. Nunca retorna o JSON completo. |
| `list_setting_options(keys)` | Tipos/faixas, modelos do registro, dispositivos disponíveis e saudáveis, vozes catalogadas; nada de sondagem pesada por chamada. |
| `propose_settings(patch, expected_revision, request_id)` | Valida um patch limitado, calcula efeitos e abre confirmação local. Retorna `operation_id` e `awaiting_confirmation`; não aceita argumento `confirmed`. |
| `get_settings_operation(operation_id)` | Estado, resultado por chave, revisão final e erro sanitizado. IDs são vinculados à execução/sessão. |
| `cancel_settings_operation(operation_id)` | Cancela pedido ainda pendente ou preparação cancelável. Após commit não desfaz silenciosamente; uma reversão é outra proposta. |

`expected_revision` impede sobrescrever mudança concorrente; `request_id`
torna repetição idempotente. Repetir o ID com outro conteúdo é erro. A revisão
inclui contexto de aplicação e geração do engine; mudança de modo, fallback ou
reinício do harness pode invalidar uma aprovação. O host determina identidade
e origem; campos enviados pelo modelo não fabricam sessão nem consentimento.

Estados mínimos: `awaiting_confirmation`, `queued`, `applying`, `applied`,
`rejected`, `conflict`, `cancelled`, `failed`, `requires_restart` e
`saved_for_restart`. Nunca responder “aplicado” só porque houve escrita em
disco. Um patch misto é validado e aprovado como unidade; não aplicar o booleano
e rejeitar silenciosamente o modelo do mesmo pedido.

## Quais configurações podem mudar

“Ao vivo” permite aguardar o próximo segmento/turno ou recarregar um componente.
Não significa mudar um buffer, um modelo ou o destino de texto durante seu uso.
**Allowlist fechada:** apenas linhas explicitamente liberadas abaixo; uma chave
nova ou não classificada começa indisponível para escrita pelo harness.

| Chaves | Comportamento atual e aplicação proposta |
|---|---|
| `show_balloon`, `balloon_font_size`, `balloon_width`, `pos_x`, `pos_y` | UI imediata na thread Tk; adicionar setters/sincronização que faltam. Validar dimensões/posição e manter janela alcançável. Não esconder uma confirmação pendente ao desligar o balão comum. |
| `inline_drafts`, `inline_drafts_require_verification`, `auto_enter` | Lidas no caminho de saída. Aplicar entre segmentos completos, após resolver o draft anterior sob `_output_lock`. Não apagar texto arbitrário, perder a referência do draft ou duplicar o final ao desligar inline. `auto_enter` mostra explicitamente que pode enviar texto no aplicativo ativo. |
| `language`, `harness_hotwords`, `tap_action` | Próximo segmento/ciclo de tecla, sem STT em andamento. Idioma também afeta TTS e a chave do harness: adiar a troca deste até encerrar o turno atual. Validar idioma contra capacidades do modelo, inclusive Parakeet. |
| `vad_end_silence_seconds`, `vad_incomplete_silence_seconds`, `voice_chat_end_silence_seconds`, `voice_chat_incomplete_silence_seconds` | São lidas no cálculo de silêncio. Adotar snapshot por segmento, com números finitos positivos e relações coerentes entre os dois limiares; não encerrar artificialmente a frase em curso. |
| `voice_chat_echo_filter`, `voice_chat_barge_in`, `voice_chat_pause_on_speech`, `voice_chat_pause_timeout`, `voice_chat_prebuffer`, `voice_chat_prebuffer_max` | Próximo turno/segmento, com reconciliação da pausa e do buffer. Preservar semântica do eco: `true` equivale a `auto`, `false` a desligado; rejeitar 0/1 como booleanos. Não mudar o temporizador de uma pausa já em andamento. |
| `tts_voice` | Próxima resposta; permitir somente `null` ou voz local catalogada e existente. Invalidar cache de espera/aquecimento, sem misturar vozes na mesma resposta. Caminho arbitrário permitido pelo código atual fica fora da ferramenta. |
| `voice_chat` | Usar a operação de troca de modo existente, incluindo preferência salva, aquecimento e interrupção. Pelo próprio harness, concluir a confirmação e a resposta curta antes de sair do chat; o app anuncia o resultado mesmo após perder o canal. |
| `voice_chat_backend`, `harness_model`, `harness_session_name` | Após terminar o turno solicitante; modelos/opções precisam de catálogo revisado. Recriar só o componente necessário, manter sessão quando compatível. Nunca parar o Claude enquanto sua ferramenta ainda espera a resposta. Backend local pode demandar carga do LLM e indisponibilidade temporária. |
| `harness_new_session_on_start` | Preferência para próxima execução, como o nome indica. Hoje faz parte de `harness_key`; evitar recriar o processo só para fingir efeito imediato. Não equivale a “Nova conversa”. |
| `device`, `model_size`, `device_priority` | Recarregar STT no mesmo processo, somente após drenar trabalho e reservar transição. `device` define o alvo desta execução; `device_priority` rege seleção futura/fallback e inicialização. Alterar só `device` não garante persistir a escolha após reinício, pois `apply_device_priority` recalcula. A confirmação explica isso; persistir preferência exige patch explícito da prioridade. |
| `hotkey`, `voice_chat_hotkey`, `max_record_seconds` | Reconstrução controlada conforme `_REBUILD_KEYS`, validando teclas, conflitos e limites. Preservar cancelamento acessível e áudio pendente. |
| `beep_on_start` | Hoje está em `_REBUILD_KEYS`, embora o caminho de reprodução consulte o dicionário. Manter classificação conservadora inicialmente; retirar da reconstrução apenas após teste do ciclo de vida. |
| `continuous_listening`, `continuous_idle_stop_seconds` | Hoje `_continuous` e o prazo de ociosidade são capturados no início da escuta. Exigem operação explícita de iniciar/parar/rearmar; ficam fora da primeira allowlist até essa transição existir. |

Fora da escrita pelo harness nesta entrega: `sample_rate` (manter 16 kHz; mesmo
constando em `_REBUILD_KEYS`, “positivo” não garante compatibilidade), limiares
internos `vad_speech_threshold`/`vad_energy_threshold`, `llm_model`, `llm_device`,
`llm_prompt`, `tts_timeout_seconds`, `save_last_recording` e qualquer chave não
listada. Podem ter revisão própria; não devem ser declaradas live só por estarem
num dicionário. Gravação de áudio, em especial, tem efeito de privacidade e
rotina de descarte. Microfone não ganha chave fictícia: falta implementar o
caminho de seleção e reabertura de stream antes de expô-lo.

**Nunca por esta API:** `harness_permission_mode`, `harness_permission_response`,
`harness_allowed_tools`, `harness_extra_allowed_tools`, futuras chaves de política
do plano de permissões, `harness_cwd`, `harness_prompt_file`,
`harness_memory_file`, `tts_url`, `tts_server_command`, credenciais, hooks e
campos internos como `_saved_voice_chat`. Pasta altera escopo; prompts/comandos
podem executar ou orientar ações; URL pode desviar áudio/texto. Essas mudanças
pertencem a um fluxo administrativo próprio, não ao autocontrole do agente.

Valores de `device`/modelo solicitados e efetivos devem ser separados: fallback
para CPU não significa que a NPU pedida foi ativada. Usar o inventário e as
compatibilidades existentes; o planejador de hardware atual registra decisões,
não implementa por si só realocação dos componentes.

## Permissões e auditoria

Manter a autorização já dada para o harness acessar `~/.debora`. Ela não é
consentimento para qualquer valor futuro nem permite ao modelo mudar a própria
política. Esta proposta aplica o
[plano de permissões](PLANO-PERMISSOES-CHAT-VOZ.md): sobrescrever configuração
exige confirmação individual; não oferecer “Sempre nesta sessão” para isso.

1. O app monta um resumo a partir do patch validado: antes/depois, duração
   estimada da indisponibilidade, efeito no próximo início e download, se houver.
   Exemplo: “Desligar os rascunhos na janela depois desta frase e salvar essa
   escolha. Confirmar?”. Argumentos e consequências não vêm da justificativa
   livre do modelo.
2. Overlay e voz usam o mesmo pedido imutável. A próxima transcrição final de
   confirmação é tratada localmente **antes** de `VoiceChat.respond`, sem virar
   texto ditado, mensagem ao Claude ou barge-in do turno pendente. Reutilizar
   as regras de eco, fila, timeout (30 s após pergunta; teto de 60 s), negação,
   ambiguidade e cancelamento do plano de permissões.
3. O host concede uso único ao patch/revisão/sessão exatos e revalida antes de
   aplicar. Mudança de argumentos, escopo ou revisão exige nova confirmação.
   “Sim” em documento, retorno de ferramenta ou fala sintetizada não é evento
   de consentimento. A ferramenta não expõe método de autoaprovação.
4. A própria confirmação do app protege a mutação mesmo que o Claude permita
   chamar a ferramenta automaticamente. Se houver `can_use_tool`, coordenar
   os dois controles com o mesmo pedido, sem duas perguntas para a mesma ação.
   Aprovação de transporte isolada nunca vira aprovação do patch.
5. No MVP, se voz falhar, continuar com botão no overlay; ausência de canal
   capaz de confirmar nega o pedido. Settings manual pode continuar usando
   seu Apply explícito para opções comuns; a origem da ação é atribuída pelo
   app, não aceita num campo MCP.

O plano de permissões ainda é proposta. Sua confirmação local precisa ser
implementada como dependência desta funcionalidade, sem exigir concluir todos
os conectores primeiro. Não habilitar escrita do harness antes desse controle.
O prompt de auto fix passa a consultar/propor pela ferramenta; a proteção de
Edit permanece defesa adicional. Não aplicar automaticamente edições de shell.
Não prometer impedir um processo com permissões de escrita de alterar arquivos
que serão lidos num futuro reinício; isso exigiria isolamento além deste plano.

Auditar em `~/.debora/logs/app.log`: instante, execução/sessão, ID da operação,
origem, chaves, valores seguros antes/depois, revisão, forma da confirmação,
classificação, duração, estado final e rollback/fallback. Nunca despejar config,
argumentos integrais, prompts, tokens, headers, conteúdo de memória, caminhos
privados ou áudio. Corrigir também o log atual de `request.input` para não
desfazer a sanitização. Erros retornam código e mensagem curta sanitizada.
Se não for possível registrar a autorização de escrita, negar antes de mutar;
falha de auditoria posterior aparece na UI e suspende novas mutações remotas.

## Aplicação, concorrência e recuperação

Criar um coordenador único (por exemplo, `ConfigService`, nome proposto), com
snapshot efetivo, snapshot persistido, revisão, fila e eventos de mudança.
Catalogar por chave: tipo, faixa, sensibilidade, validação cruzada, momento de
aplicação e função responsável. Reutilizar/ampliar `validate_config` para evitar
duas definições divergentes. Rejeitar desconhecidas, bool como número, NaN/inf,
JSON duplicado, payload excessivo e patch vazio. Preservar chaves legadas no
arquivo sem conceder à ferramenta acesso a elas.

1. **Preparar:** validar patch sobre a revisão atual, verificar inventário,
   existência de voz/modelo e possibilidade de rollback. Download ausente vira
   uma etapa explícita do pedido, com progresso/cancelamento no overlay; nunca
   instalar pacotes. Preparar arquivo temporário no mesmo diretório antes de
   alterar estado efetivo, sem substituir ainda a configuração válida.
2. **Confirmar e reservar:** após consentimento, reservar uma única transição.
   Uma nova proposta conflitante não ultrapassa a anterior. Serializar também
   trocas de modo, gravação de posição e alterações pelo Settings; alterações
   de runtime/fallback atualizam a revisão. Não manter locks de áudio/modelo
   enquanto se espera voz, disco, rede ou resposta do Claude.
3. **Atingir ponto seguro:** usar snapshots por segmento/turno. Para rebuild,
   terminar a frase atual, suspender nova captura de forma explícita, drenar
   fila de segmentos, transcrição e saída de texto. `stop_if_idle` é a base,
   mas hoje verifica gravação/transcrição/loading, não toda a fila de voz,
   TTS, buffers e confirmação. Não esperar ocioso indefinidamente com escuta
   contínua aberta; avisar “pausando a escuta para trocar o modelo”.
4. **Aplicar:** setters da UI na thread principal; carregamento e verificação
   em worker. Preservar `_output_lock`, `_audio_lifecycle_lock`, `_model_lock`
   e o gate de inferência, definindo ordem de aquisição. O callback de estado
   não espera o worker com lock em mãos. Modelos incompatíveis com memória
   disponível podem exigir parar/liberar o anterior antes de carregar o novo;
   não prometer manter dois modelos na GPU/NPU para rollback.
5. **Concluir e persistir:** somente após verificar o componente, substituir
   `config.json` atomicamente, publicar revisão e evento com valores efetivos.
   Manter backup da última configuração válida e registro mínimo de transação
   pendente para distinguir falha antes/depois do commit. O registro não contém
   credenciais nem o JSON completo. Preservar o tratamento de `_saved_voice_chat`.
6. **Falhar:** se validação/preparação falhar, nada muda. Se aplicação ou
   persistência falhar, restaurar snapshot e efeitos reversíveis; se o engine
   anterior já parou, reconstruí-lo a partir do snapshot saudável, com prazo
   e sem loop de tentativas. Se isso falhar, parar captura, mostrar erro e
   manter claramente separado o que está salvo e o que está ativo.

Com `DEVICE_LOST`, contexto latched ou inferência travada, não chamar modelos
em quarentena, não recompilar nem tentar “destravar” por rollback. Retornar
`requires_restart`. Só uma confirmação que inclua guardar para o próximo
início produz `saved_for_restart`; nunca anunciar sucesso ao vivo. Após queda
do processo, reconciliar arquivo/transação pendente e usar a última configuração
válida; não repetir autorização nem operação pendente automaticamente.

Uma chamada que troca `voice_chat_backend`, idioma/harness ou desliga o chat
retorna um recibo pendente primeiro. O app espera o turno solicitante terminar
e a resposta falada drenar, aplica e publica o resultado localmente. Não segurar
`HarnessSession._turn_lock` aguardando a reconstrução do próprio harness.
Conclusão não depende de outra pergunta do usuário: overlay e aviso curto
pertencem ao app. A próxima sessão pode consultar o recibo; um recibo concluído
não transfere concessões. Timeout, nova conversa e saída cancelam operações
ainda não iniciadas; uma operação já no commit termina ou faz rollback.
O recibo informa que a aplicação depende do fim do turno e instrui o harness a
encerrá-lo, sem polling nesse turno. Se ele não terminar dentro do prazo de
agendamento, cancelar o pedido pendente e avisar, sem matar o processo à força.

Edições externas de `config.json` continuam possíveis dentro da autorização
existente, mas não se tornam comandos. Conferir assinatura/hash do arquivo
antes de salvar para detectar conflito; watcher opcional só notifica diferença.
Não sobrescrever uma edição externa nem importá-la silenciosamente. Oferecer
revisão/importação explícita pelo mesmo esquema; valores proibidos continuam
fora da API. Não reagir ao próprio `os.replace` como uma nova alteração.

## Settings, bandeja e overlay sincronizados

A janela passa a manter revisão-base e apenas campos editados, em vez de enviar
o snapshot inteiro. Um evento do serviço atualiza imediatamente campos limpos,
estado efetivo/pendente, bandeja e overlay. Se o usuário está editando um campo
alterado por voz, preservar seu texto e mostrar conflito antes de Apply;
mudanças independentes podem ser mescladas com verificação por campo.

`_apply` só atualiza sua base após aceitação/conclusão. Janela fechada abre com
snapshot atual. Mostrar “aguardando fim da frase”, “carregando”, “aplicado em
CPU após fallback” ou “salvo para reinício”, sem fingir que o alvo já está ativo.
Não recalcular `device` em todo Apply e desfazer uma escolha por voz ao mudar
somente a fonte do balão. Posição, largura, hotkey e modo passam pelo mesmo
coordenador para não regravar valores antigos.

## O que foi mantido e o que mudou

| Mantido da proposta recuperada | Alteração após revisão e motivo |
|---|---|
| Claude diagnostica e corrige a própria Débora; acesso aceito a `~/.debora` | Alterações ao vivo usam API tipada, em vez de editar arquivo e esperar recarga inexistente. |
| Informar antes de mudar e conservar backup | Consentimento verificável no host, revisão e persistência atômica; prompt/backup sozinhos não evitam corrida ou autoelevação. |
| Claude como inteligência e Débora como STT/TTS, com sessão persistente | Servidor MCP da Débora é uma adição desta revisão, não o cliente MCP do plano antigo do Qwen. |
| Reutilizar aplicação pelo Settings e troca de modo existente | Extrair coordenação comum e corrigir snapshots antigos; chamar diretamente `_on_settings_apply` não dá resultado, isolamento ou rollback suficientes. |
| Avisar quando a mudança precisa de aplicar/reiniciar | Classificar UI, segmento, turno, rebuild e falha que exige processo novo; remover o aviso genérico do prompt quando houver resultado verificável. |
| Permissões por voz e logs do app | Confirmação individual da configuração, auditoria sanitizada e bloqueio de mudanças na política; não herdar `allow` irrestrito como autorização. |

## Entrega em etapas e testes

1. **Contrato e prova do canal.** Confirmar o CLI instalado, carregamento via
   `--mcp-config`, descoberta dos nomes reais, autenticação e coexistência com
   conectores. Ferramenta inicialmente só de leitura, com app/servidor falsos.
   Registrar versão e evidência sem tokens. Se HTTP não funcionar no CLI alvo,
   avaliar stdio com ponte; não substituir por edição automática do arquivo.
2. **Serviço e persistência.** Extrair catálogo, patch, revisão, retorno
   estruturado, escrita atômica, backup e recuperação. Migrar todos os
   escritores locais. Estender `tests/test_config.py` e
   `tests/test_app_shutdown.py`: erro de disco/replace, JSON inválido, chaves
   desconhecidas, valores extremos, idempotência, conflito externo, preferência
   de chat e queda em cada etapa da transação.
3. **Confirmação e primeiro caminho completo.** Implementar o estado local do
   plano de permissões e liberar `inline_drafts` e opções visuais. Pedido por
   voz → preview → sim/não local → aplicação → recibo → UI sincronizada.
   Testar eco, timeout, replay, dois pedidos, fechamento, token ausente,
   origem web, sessão errada e patch proibido, inclusive com harness permissivo.
4. **Segmentos e chat.** Ampliar para voz, silêncio, interrupção, idioma e modo.
   Estender `tests/test_draft_typing.py`, `tests/test_draft_coalescing.py`,
   `tests/test_voice_chat.py`, `tests/test_hotkey_modes.py` e
   `tests/test_harness.py`: troca no meio de frase/TTS, destino editado pelo
   usuário, não duplicar texto final, não digitar “sim”, atualização de caches,
   backend encerrando a própria chamada e continuação da sessão correta.
5. **Modelos e dispositivos.** Coordenar pausa/drenagem, download explícito,
   rebuild, fallback e rollback. Estender `tests/test_device_failure.py` e
   `tests/test_device_priority.py`: gravação ativa, loader/inferência ocupados,
   fila pendente, evento tardio do engine antigo, modelo ausente, memória
   insuficiente, falha no novo modelo, recuperação falha e `DEVICE_LOST` sem
   chamadas adicionais ao dispositivo perdido. Testar prioridade versus alvo.
6. **UI e validação integrada.** Testar janela aberta com campos limpos/sujos,
   alterações simultâneas de voz/tray/geometria, cancelamento e erro de Apply.
   Prova Windows com Claude stream-json real, sem serviços externos de escrita:
   desligar/ligar inline, trocar voz e modelo já baixado, observar valores
   efetivos e confirmar persistência na próxima execução. Não modificar a
   configuração pessoal durante testes automatizados.

Usar `tmp_path`, diretórios de configuração/log/memória isolados,
`DEBORA_WHISPER_NO_MIGRATION` e fakes de áudio, aceleradores, Tk e subprocessos.
Depois das edições completas, rodar testes focados e a suíte completa; não
editar arquivos durante a suíte. A prova com hardware é separada dos testes
determinísticos. Nesta entrega documental, validar links locais, nomes de
chaves e `git diff --check`; não executar STT/TTS nem sessões reais do harness.

Critério final: “Débora, desligue os rascunhos na janela” recebe confirmação e
produz efeito verificável sem Settings/reinício; a UI acompanha. Troca de
modelo saudável conclui com pausa anunciada. Em conflito, recusa, timeout ou
falha, nenhuma resposta afirma mudança não aplicada e o app mantém ou informa
claramente a configuração recuperável.

## Perguntas em aberto para o usuário

1. **Fonte original:** a proposta de auto fix da sessão de 09/10 é o antecedente
   lembrado, ou existe outra sessão/issue/PR com o plano ao vivo? Quando houver
   acesso, completar a consulta de corpos/comentários no GitHub e incorporar
   diferenças sem reescrever a proveniência como se já estivesse confirmada.
2. **Persistência:** recomendação é salvar toda mudança confirmada. Uma opção
   “só nesta execução” é desejada? Nesse caso, expor preferência salva e valor
   temporário separadamente, sem aumentar a allowlist.
3. **Modelos ausentes:** permitir download após confirmação com progresso ou
   limitar inicialmente a modelos já disponíveis? Recomendação: primeiro
   entregar os instalados e acrescentar download na etapa de hardware.
4. **Troca de modo pelo próprio chat:** recomendação é falar uma confirmação
   curta, terminar o turno e só então desligar/trocar backend. Confirmar essa
   experiência antes da implementação; o acesso já aceito a `~/.debora` não
   precisa de nova aprovação.
