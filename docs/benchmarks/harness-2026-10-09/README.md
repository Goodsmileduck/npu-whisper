# Claude Code por stdio — 2026-10-09

Executado no Windows com `claude.exe` 2.1.295, modelo padrão informado pelo
CLI: `claude-opus-5-5`. Cada execução usa uma pasta temporária, sem escrever
no repositório. O script imprime JSONL; os exemplos abaixo omitem metadados
de uso, UUIDs de eventos e blocos grandes de inicialização.

```powershell
.venv/Scripts/python.exe docs/benchmarks/harness-2026-10-09/probe_claude_stream.py
.venv/Scripts/python.exe docs/benchmarks/harness-2026-10-09/probe_claude_stream.py --stdio
```

## Turnos e latência

Comando base:

```text
claude -p --input-format stream-json --output-format stream-json --verbose
  --include-partial-messages --session-id <uuid>
  --permission-mode acceptEdits --permission-prompts host
  --append-system-prompt-file <arquivo-temporário>
  --system-prompt-snapshot off
```

Uma mensagem por linha no stdin, mantendo-o aberto entre turnos:

```json
{"type":"user","message":{"role":"user","content":"Lembre da palavra jabuticaba. Diga só que lembrou."}}
{"type":"user","message":{"role":"user","content":"Qual palavra pedi para lembrar? Responda brevemente."}}
```

O segundo turno lembrou `jabuticaba`, no mesmo processo e sessão. Tempos do
envio até o primeiro `text_delta`, medidos com `time.monotonic()`:

| Execução | Turno 1, frio | Turno 2, quente |
|---|---:|---:|
| `host` sozinho | 3,781 s | 1,766 s |
| `host` + `--permission-prompt-tool stdio` | 2,578 s | 0,719 s |
| Repetição com retomada e snapshot desativado | 3,031 s | 1,453 s |

São amostras isoladas, não médias. O primeiro comando foi executado antes
de acrescentar `--system-prompt-snapshot off` ao script. Processo quente
reduz a espera, mas o tempo também depende da API e do conteúdo.

Tipos vistos: `system`, `stream_event`, `assistant`, `user` (resultados de
ferramentas/interrupção), `result`, `rate_limit_event`, `control_request` e
`control_response`. Subtipos de `system`: `init`, `hook_started`,
`hook_response`, `commands_changed`, `status`, `thinking_tokens`.

Recortes do stdout:

```json
{"type":"system","subtype":"init","session_id":"6c56f122-2d9c-472f-b7e2-dda14fe8ba9f","model":"claude-opus-5-5"}
{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"VOZCONFI"}}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"VOZCONFIRMADA\n\nJabuticaba."}]}}
{"type":"result","subtype":"success","is_error":false,"terminal_reason":"completed","result":"VOZCONFIRMADA\n\nJabuticaba."}
```

**`result` encerra o turno.** `message_stop` encerra apenas uma mensagem;
pode haver execução de ferramenta e outra mensagem depois. `assistant`
repete texto já entregue nos deltas, portanto não deve ser falado novamente.

## Interrupção

Enviada logo após o primeiro delta de uma história longa:

```json
{"type":"control_request","request_id":"3463916f-1a94-4549-96f9-c6e8fdab2cb7","request":{"subtype":"interrupt"}}
```

Voltou a confirmação, uma mensagem `user` com `[Request interrupted by user]`
e, depois, o fim do turno (recortes):

```json
{"type":"control_response","response":{"subtype":"success","request_id":"3463916f-1a94-4549-96f9-c6e8fdab2cb7","response":{"still_queued":[]}}}
{"type":"result","subtype":"error_during_execution","is_error":true,"terminal_reason":"aborted_streaming"}
```

O processo continuou aceitando turnos. A integração drena até `result` e
trata esse erro como interrupção esperada quando o usuário já cancelou.

## Permissões

**`--permission-prompts host` sozinho não funcionou como canal de pedidos.**
Não emitiu `control_request`; o Bash retornou `This command requires approval`
e não executou. Acrescentando `--permission-prompt-tool stdio`, usado pelo
[transporte oficial do SDK Python](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/_internal/transport/subprocess_cli.py),
o pedido chegou pelo stdout. Nenhum SDK foi instalado.

Pedido real de `git --version` (campos de sugestão de regras omitidos):

```json
{"type":"control_request","request_id":"5a0d8066-e72c-48db-993d-b00d9a30f72d","request":{"subtype":"can_use_tool","tool_name":"Bash","display_name":"Bash","input":{"command":"git --version","description":"Show installed git version"},"description":"Show installed git version","decision_reason":"This command requires approval","decision_reason_type":"other","tool_use_id":"toolu_01D3bWtwCueSGmNDmdnwA8ot"}}
```

Resposta exata enviada no stdin para autorizar somente essa execução:

```json
{"type":"control_response","response":{"subtype":"success","request_id":"5a0d8066-e72c-48db-993d-b00d9a30f72d","response":{"behavior":"allow","updatedInput":{"command":"git --version","description":"Show installed git version"}}}}
```

O comando retornou `git version 2.55.0.windows.5`. O segundo comando era
`git -c alias.debora='!echo debora' debora`, um alias temporário que só faria
echo. A resposta exata de negação foi:

```json
{"type":"control_response","response":{"subtype":"success","request_id":"b92f7b59-dc75-43cc-acb0-841e402b0044","response":{"behavior":"deny","message":"Negado pelo probe."}}}
```

O Claude confirmou a negação, não tentou outro comando, e `result` incluiu
o Bash em `permission_denials`. A aplicação usa o mesmo formato, nega
pedidos por padrão e registra ferramentas, pedidos e decisões em `app.log`.

## Regras de diagnóstico sem pergunta

Prova manual do feedback de 2026-10-09, com `claude.exe` **2.1.295**, modelo
`claude-opus-5-5`, `acceptEdits` e o mesmo transporte stream-json da aplicação.
O [probe](probe_allowed_tools.py) usa `HarnessSession.send`, o
`permission_handler` real (sempre retorna `False`) e `permission_response` da
Débora. Cada processo tem cwd, memória, prompt e arquivo de sessões temporários.
`--setting-sources "" --strict-mcp-config --tools PowerShell` isola as regras de
settings pessoais/projeto e conectores; a rodada Bash troca apenas a ferramenta.
Nenhuma instância da Débora ou do servidor TTS foi iniciada ou encerrada.

```powershell
.venv/Scripts/python.exe docs/benchmarks/harness-2026-10-09/probe_allowed_tools.py none
.venv/Scripts/python.exe docs/benchmarks/harness-2026-10-09/probe_allowed_tools.py exact
.venv/Scripts/python.exe docs/benchmarks/harness-2026-10-09/probe_allowed_tools.py colon
.venv/Scripts/python.exe docs/benchmarks/harness-2026-10-09/probe_allowed_tools.py glob
.venv/Scripts/python.exe docs/benchmarks/harness-2026-10-09/probe_allowed_tools.py packaged
.venv/Scripts/python.exe docs/benchmarks/harness-2026-10-09/probe_allowed_tools.py packaged Bash
```

Argumentos são elementos separados de argv, sem shell intermediário. Por exemplo:

```text
--allowedTools "PowerShell(docker info:*)" "PowerShell(docker ps:*)"
```

Resultados observados (número de chamadas ao handler, todas negadas):

| Regras / ferramenta | `docker ps` | `docker ps --all` | `docker info; docker ps` | `docker rm debora-permission-probe-<uuid>` |
|---|---:|---:|---:|---:|
| Nenhuma regra adicional / PowerShell | 0 | 0 | 1 | 1 |
| `PowerShell(docker info)`, `PowerShell(docker ps)` | 0 | 0 | 0 | 1 |
| `PowerShell(docker info:*)`, `PowerShell(docker ps:*)` | 0 | 0 | 0 | 1 |
| `PowerShell(docker info *)`, `PowerShell(docker ps *)` | 0 | 0 | 0 | 1 |
| Padrão empacotado / PowerShell | 0 | 0 | 0 | 1 |
| Padrão empacotado / Bash | 0 | 0 | 0 | 1 |

**As três sintaxes funcionaram no composto do PowerShell.** O padrão usa formas
exatas, sem wildcard de argumentos. As duas rodadas `packaged` usaram o próprio
`harness_command` e também executaram `docker info` sozinho sem pedido. As
rodadas iniciais de comparação começaram por `docker ps`; o script inclui agora
`docker info` como primeiro comando para facilitar a reprodução.

`docker ps` isolado já passa pela política interna do CLI sem nossa regra;
portanto, sozinho não prova que `--allowedTools` funcionou. O composto foi a
comparação decisiva: sem regras, negado; com uma regra para cada parte, executado
em uma única chamada PowerShell. A execução retornou exit code 1 porque o daemon
Docker estava indisponível (`dockerDesktopLinuxEngine` não encontrado), mas o
cliente rodou e `docker info` informou versão **29.8.2**. Não iniciamos o Docker.

O comando não listado chegou como `control_request.request.subtype: can_use_tool`
ao handler da Débora, com `tool_name: PowerShell` (ou `Bash` na rodada equivalente).
O resultado da ferramenta foi `Permissão negada pela Débora.`; nenhuma remoção
foi executada. O nome de container descartável usa UUID em vez de um nome real.
Não aceitamos as sugestões de persistir regras em `localSettings`.

`harness_allowed_tools: null` seleciona `DEFAULT_ALLOWED_TOOLS`; `[]` não
acrescenta regras; uma lista substitui o padrão. O padrão inclui somente formas
de diagnóstico escolhidas, sem variantes de escrita (`git branch -D`,
`--output`, `nvidia-smi -pl`) nem leitura contínua (`-f`, `Get-Content -Wait`).
Get-Content fica restrito a README.md e Test-Path a `.`, com variantes `-Path`;
outros caminhos podem ser acrescentados explicitamente na configuração.
As regras são aditivas às permissões do CLI: não revogam permissões preexistentes.
Essas provas verificam o transporte e os casos descritos, não todos os comandos
da lista nem uma revisão de segurança. Interface multi-monitor/DPI e áudio
continuam sem validação visual/auditiva nesta passada.

## Memória de voz e `--add-dir`

Verificação manual em 2026-10-09 com o mesmo `claude.exe` 2.1.295, usando
`HarnessSession` já com a memória integrada, cwd temporário e estes argumentos
além do comando base: `--permission-prompt-tool stdio --add-dir
C:\Users\alexandre-machado\.debora\harness`. O arquivo configurado foi
`~/.debora/harness/voice-memory-probe-<uuid>.md`, descartável para não alterar
a memória do usuário. A inicialização criou somente o cabeçalho e incluiu o
caminho/conteúdo no prompt datado.

| Operação real sob `acceptEdits` | Pedidos stdio | Resultado no disco |
|---|---:|---|
| `Read` + `Write` da correção no arquivo de memória | 0 | Conteúdo gravado |
| `Read` + `Edit` de `dictation_engine.py` para `dictation_engine` | 0 | Conteúdo alterado |
| `Write` em `~/.debora/voice-memory-outside-<uuid>.md` | 1, negado | Arquivo não criado |

O pedido externo foi `can_use_tool`, ferramenta `Write`, com
`decision_reason_type: "workingDir"` e motivo
`Path is outside allowed working directories`; sugeria acrescentar
`~/.debora` às pastas permitidas. O host negou e não aceitou essa sugestão.
Arquivos de prova foram removidos ao terminar. `--add-dir` funcionou como
esperado; nenhuma alternativa foi necessária. Isso verifica permissões de
edição, não isolamento de leitura nem uma sandbox: permissões existentes
continuam valendo, e `harness_cwd` na pasta pessoal já inclui `~/.debora`.

A injeção retém as últimas 100 linhas completas dentro de 8 KiB e registra
truncamento. O conteúdo é relido em cada início de processo, inclusive retomadas;
editar a memória não reinicia um processo vivo. Só o caminho resolvido entra
em `harness_key`, pois trocá-lo exige atualizar o prompt e a pasta permitida.
Não foram executados suíte de testes nem revisão de segurança nesta tarefa.

## Prompt, retomada e integração

`--append-system-prompt-file` funciona com stdin em stream-json: o arquivo
mandava começar com `VOZCONFIRMADA`, e os dois turnos obedeceram.
Após fechar o processo, o probe mudou o arquivo, acrescentou idioma/data e
abriu outro processo com `--resume 6c56f122-2d9c-472f-b7e2-dda14fe8ba9f`.
Com `--system-prompt-snapshot off`, a resposta começou com `VOZRETOMADA`,
lembrou `jabuticaba` e informou `pt` e `2026-10-09, 13:48`, como no arquivo.
O processo retomado terminou com código zero.

Também executei `HarnessSession` diretamente com o CLI real, pasta e arquivo
de sessões temporários: guardou `pitanga`, negou `git --version` por padrão,
salvou o UUID e lembrou a palavra após `stop()` e nova instanciação. Não foi
executada nem criada uma suíte de testes.

Limites desta passada: não validei áudio/latência do Chatterbox nem interação
visual com Settings; o executável observado foi `.exe`, não o wrapper `.cmd`.
Idioma/data são acrescentados ao arquivo ao iniciar cada processo; a data
não é atualizada a cada turno de um processo que continua vivo. Permissões
por voz e revisão de segurança ficam para as próximas fases.
