# Plano: permissões no chat de voz com Claude

Data: 2026-10-10
Status: proposta

Objetivo: permitir consultas úteis por voz — arquivos, Gmail, Calendar, Drive e
web — sem transformar um pedido de leitura em autorização para enviar, apagar
ou modificar dados. Este documento propõe a próxima etapa de permissões;
nenhuma dessas mudanças de autorização está implementada nesta entrega.

## Como funciona hoje

`harness_command`, em `debora_whisper/harness.py`, inicia `claude -p` com
`--permission-mode acceptEdits`, `--permission-prompts host`,
`--permission-prompt-tool stdio` e `--allowedTools`. O padrão
`DEFAULT_ALLOWED_TOOLS` contém diagnósticos de shell em formas exatas.
Um `control_request` com `can_use_tool` recebe a decisão de
`harness_permission_response` (`deny` por padrão) ou do `permission_handler`.
Autorizações do próprio Claude podem executar a ferramenta sem esse pedido.
Assim, negar no handler não transforma `acceptEdits` em modo somente leitura.

Os relatos de hoje incluem Gmail `search_threads`, WebSearch, Grep/Glob,
`docker info` e cópia de backup negados. `docker info` já está na lista atual:
antes de ampliar regras, conferir o nome canônico da ferramenta, os argumentos,
uma lista personalizada que substitua o padrão, regras herdadas e se o processo
foi reiniciado. Não se conclui a causa apenas pelo relato. `Copy-Item` é escrita,
mesmo quando serve para backup; pode sobrescrever um arquivo ou copiar segredos.

A tela `ui/settings.py` oferece backend e pasta. O README explica as regras
atuais; [VOICE_HARNESS_IDEA.md](VOICE_HARNESS_IDEA.md) reserva a fase 3 para
permissões por voz. O callback atual é um ponto de integração, mas ainda falta
encaminhar uma resposta de voz para ele sem abrir outro turno do Claude.

## O que o CLI oferece

Verificado localmente com `claude --version` e `claude --help` em 2026-10-10:
Claude Code **2.1.296**.

| Opção | Uso nesta proposta |
|---|---|
| `--allowedTools` / `--allowed-tools` | Regras que permitem ferramentas; não limitam sozinhas todas as demais permissões. |
| `--disallowedTools` / `--disallowed-tools` | Bloqueios definitivos; não usar para uma ação que precisa poder ser aprovada. |
| `--permission-mode` | O help lista `acceptEdits`, `auto`, `bypassPermissions`, `manual`, `dontAsk` e `plan`. Proposta: `manual`. |
| `--settings <file-or-json>` | Carregar a política gerada para este processo. |
| `--setting-sources` | Selecionar fontes de settings; avaliar no teste de integração como isolar autorizações herdadas sem perder conectores. |
| `--permission-prompts host` | Entregar pedidos ao host; `none` os nega automaticamente. |
| `--name <name>` / `-n` | Definir o nome exibido da sessão. Existe; não é necessário inventar alternativa. |

O help menciona `--permission-prompt-tool` na descrição de `host`, mas não o
lista como opção separada. O uso de `stdio` já tem evidência no
[teste do protocolo](benchmarks/harness-2026-10-09/README.md).

As regras são avaliadas em ordem `deny`, `ask`, `allow`. Um `ask` amplo não tem
exceção por um `allow` mais específico. MCP admite o nome exato da ferramenta
e padrões como `mcp__server__*`; este último libera também escritas daquele
servidor. O padrão `mcp__github__get_*` restringe o prefixo da ferramenta, mas
seu nome não comprova ausência de efeitos colaterais. Esses detalhes vêm da
[documentação de permissões](https://code.claude.com/docs/en/permissions).

`--settings` não deve ser tratado como substituição de todas as regras:
listas de settings podem ser combinadas e políticas gerenciadas têm precedência.
Ver [settings e precedência](https://code.claude.com/docs/en/settings).

## Níveis e padrão recomendado

Os níveis determinam o que pode ser aprovado automaticamente pela política da
Débora. O usuário ainda pode aprovar uma ação individual fora desse conjunto.

| Nível | Sem pergunta, dentro do escopo configurado | Exige confirmação |
|---|---|---|
| `local_read` — Leitura local | Read, Grep, Glob e diagnósticos locais conhecidos | Conectores, web, escrita e comandos desconhecidos |
| `connected_read` — Leitura local, conectores e web **(padrão proposto)** | Nível anterior + consultas revisadas nos conectores habilitados e pesquisa pública na web | Escritas locais/remotas, envio, exclusão, alteração de agenda, acesso fora do escopo |
| `local_write` — Trabalho local | Nível anterior + edições reversíveis em pastas de trabalho explicitamente selecionadas | Envios/publicações, exclusões, sobrescrita de backup/configuração, eventos de agenda e comandos desconhecidos |

Não haverá nível que autorize automaticamente envios, remoções, lixeira,
compartilhamento, compras ou mudanças de credenciais. Criar/alterar evento exige
confirmação mesmo sem convidados. Criar rascunho remoto também é escrita e pede
confirmação. O terceiro nível não autoriza modificar a política da própria
Débora, settings do Claude, hooks, credenciais ou arquivos de inicialização.

Para leitura local, resolver os caminhos e limitar ao `harness_cwd` e a pastas
adicionais explicitamente selecionadas; não confundir `--add-dir` com uma
fronteira de segurança. Informar quando a pasta escolhida for toda a home.
Segredos conhecidos (`.env`, chaves, tokens) e caminhos externos pedem aprovação
específica; symlinks e caminhos de rede entram nos testes de escopo. A memória
de voz não dá permissão implícita de edição a toda a pasta `~/.debora`.

Nos conectores, manter catálogo versionado por servidor, ferramenta e esquema:
consultas de mensagens, agenda e arquivos revisadas podem ser automáticas.
`mcp__claude_ai_Gmail__search_threads` é o nome observado no relato; os nomes de
Calendar/Drive e de suas ferramentas devem vir do inventário real da sessão,
sem inventar equivalentes. Conta e conector habilitados aparecem nas configurações.
Uma ferramenta nova, renomeada ou com esquema diferente pede confirmação até
ser revisada. Descrições MCP e anotações de leitura são pistas, não autoridade.
Nunca gerar `mcp__claude_ai_Gmail__*` como autorização de leitura.

`WebSearch` pode pesquisar uma consulta pública do usuário. Conteúdo privado
obtido de mail/Drive/arquivos não pode virar consulta, URL ou parâmetro externo
automaticamente. `WebFetch` fica restrito a destinos públicos revisados, com
validação de redirecionamentos; rede interna/local pede confirmação. Consultar
um conector tampouco autoriza copiar seus resultados para outro serviço.

Diagnósticos como `docker info` podem ser aprovados pelo catálogo de comandos
exatos. Composição de comandos, redirecionamento, expansão, scripts e opções
não reconhecidas pedem confirmação. Não ampliar para `PowerShell(*)`,
`Bash(*)`, `docker *` ou um prefixo amplo de `Copy-Item`. Uma cópia proposta
mostra origem, destino e se haverá sobrescrita antes do sim.

## Onde aplicar a decisão

A política deve funcionar fora do modelo. Um prompt pedindo cuidado não é
controle de acesso. Proposta inicial: `manual` e um JSON temporário em
`--settings` com `permissions.ask: ["*"]`, mais bloqueios explícitos do usuário.
O handler classifica cada pedido, autoriza leituras conhecidas pelo nível e
abre a confirmação para os demais. Isso faz o `ask` prevalecer sobre permissões
amplas herdadas; a lista `--allowedTools` deixa de ser o mecanismo principal.
`--disallowedTools` fica reservado às proibições definitivas.

Antes de entregar essa política, uma prova com o CLI instalado deve demonstrar
que todas as ferramentas relevantes, inclusive as descobertas depois do início,
passam pelo controle. Testar também hooks, subagentes, permissões lembradas pelo
Claude, `acceptEdits` herdado, plugins e políticas gerenciadas. Não auto-aprovar
uma ferramenta que possa delegar execução sem controlar as ferramentas delegadas.
O handler não pode contrariar um bloqueio organizacional.

Se alguma execução escapar desse caminho, isolar as fontes de settings e
customizações, ou usar um controle anterior à execução com suporte comprovado
no CLI. Preservar autenticação e inventário dos conectores é critério da prova;
não ligar `--strict-mcp-config` indiscriminadamente e depois perder o Gmail.
Até provar a mediação, manter a ação bloqueada e explicar a incompatibilidade.
Não recorrer a `auto`, `bypassPermissions` nem `harness_permission_response:
"allow"`. O help alerta que settings inválidos podem ser ignorados em `-p`:
validar o arquivo e sua aplicação efetiva antes de aceitar pedidos do usuário.

## Pergunta por voz e no overlay

1. Ao receber `can_use_tool`, registrar um pedido pendente com `request_id`,
   sessão, ferramenta, argumentos imutáveis e prazo monotônico. Serializar
   pedidos de confirmação para que um “sim” nunca sirva para duas ações.
2. A Débora pausa a reprodução comum e fala um resumo criado a partir dos
   argumentos: **“Débora pede permissão para enviar este e-mail para Ana — sim
   ou não?”** O overlay mostra ação, conta, destinatários, assunto/conteúdo ou
   arquivos afetados, com **Permitir uma vez**, **Negar** e, quando cabível,
   **Sempre nesta sessão**. Destinos e consequências nunca ficam ocultos só
   porque a justificativa do modelo diz que a ação é segura.
3. Entrar em um estado específico de confirmação. A próxima transcrição final
   elegível responde ao pedido localmente, antes de `VoiceChat.respond`; não
   vira mensagem do Claude, texto ditado ou gatilho de interrupção do turno
   pendente. Um “sim” isolado só vale após terminar a pergunta, sem áudio
   antigo na fila. Não aceitar a própria voz/TTS como aprovação.
4. “Sim”/“permitir uma vez” aprova exatamente aquele pedido; “não”/“cancelar”
   nega. Respostas ambíguas pedem repetição sem reiniciar indefinidamente o
   prazo. Para envio, exclusão e mudanças de agenda, preferir uma frase ligada
   à ação (“sim, enviar para Ana”) ou o botão correspondente. Voz não autentica
   a identidade de quem fala; permitir desativar aprovação por voz.
5. Prazo padrão: **30 segundos após a pergunta**, com limite absoluto de
   **60 segundos desde o pedido** para cobrir TTS indisponível. Timeout, falha
   de áudio, troca de modo, fechamento do overlay de confirmação, reset,
   cancelamento, saída do app ou processo morto negam. Uma resposta tardia não
   autoriza outro pedido. No modo somente overlay, contar desde sua exibição.
6. Enviar uma única `control_response` com o mesmo `request_id`, por
   `permission_response`; ao permitir, manter os argumentos aprovados. Alteração
   de argumentos exige novo pedido. Drenar o resultado como no fluxo atual.

O worker pode aguardar um `Event`/fila com prazo; a UI usa `root.after` e nunca
bloqueia a thread Tk. STT e os eventos de cancelamento continuam ativos. O
`permission_handler` booleano atual precisa integrar esse estado e o `stop`,
sem deadlock com `_turn_lock` nem com a fila de turnos. Sem APIs novas do Windows.

## “Sempre nesta sessão”

Significa **nesta conversa e nesta execução da Débora**, nunca globalmente.
Guardar só em memória uma concessão por UUID, pasta, conta/conector, ferramenta,
escopo normalizado e versão da política. Uma aprovação de leitura numa pasta
não libera toda a máquina; uma regra de comando não vira prefixo arbitrário.

Oferecer essa opção apenas para leituras adicionais e edições locais reversíveis
com escopo delimitado. Não oferecer para enviar, excluir/trash, compartilhar,
alterar agenda, sobrescrever configurações/backup ou executar código arbitrário.
Essas ações sempre pedem confirmação individual, inclusive no terceiro nível.

Reiniciar apenas o subprocesso pode conservar concessões se UUID, conta, pasta
e política forem iguais. “Nova conversa”, troca de pasta/conta, redução do
nível, mudança de política e reinício da Débora as revogam. A UI lista concessões
ativas e oferece **Revogar permissões desta sessão**. Nada vai para auto-memory,
`voice_memory.md`, `harness_session.json` ou `.claude/settings.local.json`.

## Configuração e migração

Adicionar um grupo **Permissões do Claude** perto das opções de harness:
seletor de nível com exemplos, conectores/contas habilitados, pastas permitidas,
forma de confirmação, prazo e concessões ativas. Mostrar o que ainda exige
confirmação ao trocar de nível. Não pedir ao usuário para escrever padrões MCP.

| Chave proposta | Padrão | Validação e sentido |
|---|---|---|
| `harness_permission_policy_version` | `1` | Inteiro; permite migração explícita e invalidação de concessões. |
| `harness_permission_tier` | `"connected_read"` | Enum dos três níveis. |
| `harness_read_connectors` | `[]` | IDs exatos dos conectores selecionados; onboarding oferece Gmail/Calendar/Drive disponíveis, sem conceder contas silenciosamente. |
| `harness_read_roots` | `[]` | Pastas adicionais; o cwd aparece como raiz de leitura atual. |
| `harness_write_roots` | `[]` | Pastas de escrita reversível no terceiro nível; nunca toda a home por padrão. |
| `harness_permission_ui` | `"voice_and_overlay"` | Também aceita `"overlay"`; ausência de UI capaz de confirmar nega. |
| `harness_permission_timeout_seconds` | `30` | Inteiro entre 5 e 60; rejeitar booleano. Limite absoluto continua 60 s. |
| `harness_permission_session_allow` | `true` | Booleano; exibe a opção para ações elegíveis, sem concedê-las por si só. |

O padrão de nível suporta conectores; a primeira seleção explícita habilita as
contas revisadas. Quem já escolheu conectores na migração não precisa confirmar
cada consulta. O catálogo de ferramentas é da aplicação, não uma lista arbitrária
de nomes produzida pelo Claude.

Migração proposta, aplicada só quando esta fase for implementada:

- Sem chaves novas: apresentar a migração uma vez, com resumo concreto do
  acesso proposto; até a escolha, usar leitura local e negar pedidos pendentes.
- `harness_allowed_tools: null`: mapear diagnósticos revisados para o catálogo.
  Lista personalizada vira sugestões de concessões; importar automaticamente
  só entradas exatas reconhecidas como leitura no escopo. `[]` mantém a escolha
  de não acrescentar diagnósticos. Regras amplas ficam pendentes de revisão.
- `harness_permission_response: "deny"`: migrar para confirmação com timeout
  que nega, deixando clara a mudança. `"allow"` não vira autorização geral:
  desativar a aprovação irrestrita e explicar a migração na UI e no log.
- `harness_permission_mode`: os novos perfis controlam o modo `manual`.
  Configurações antigas de auto/bypass/acceptEdits não podem reabrir acesso por
  trás do perfil. Conflitos são apresentados, sem fallback permissivo.
- Após salvar a versão nova, chaves legadas ficam sem efeito e geram aviso de
  depreciação, sem apagar silenciosamente a configuração original. A UI permite
  revisar as sugestões, mas não restaura “permitir tudo”.

## Conteúdo não confiável e logs

Mail, páginas, documentos, nomes de arquivos e respostas MCP são dados, mesmo
quando dizem “o usuário autorizou” ou contêm “sim”. Não podem elevar o nível,
editar a política, fabricar eventos de UI ou preencher a resposta de permissão.
O resumo de aprovação usa modelo fixo com campos escapados e limites de tamanho;
o modelo não escolhe o risco nem substitui destinatários por uma justificativa.
Revalidar o pedido ao confirmar e invalidar se sessão, argumentos ou escopo
mudarem. Não aceitar `permission_suggestions` do Claude como concessões.

Registros em `app.log`: UUID, ID do pedido, ferramenta canônica, classe de ação,
origem da decisão (nível, sessão, voz, botão, timeout), regra/versão, duração e
resultado observado. Registrar concessão/revogação e motivo da negativa.
Não registrar corpos de e-mail, conteúdo de arquivos, tokens ou argumentos
integrais; destinos sensíveis devem ser mascarados. O log atual que interpola
`request.input` precisa ser substituído antes de habilitar conectores. O overlay
pode mostrar o detalhe necessário à decisão sem copiá-lo ao log.

Permissões de leitura não eliminam risco de exposição, e confirmação por voz
não elimina erro de STT. Testar injeções que tentem enviar dados por busca web,
URL, shell ou outro conector, além de envios diretos.

## Sessões e identificação: entrega separada

A mudança de sessão é independente deste plano de permissões:
`harness_new_session_on_start` fica **false** por compatibilidade. Quando true,
o primeiro uso de cada pasta numa execução da Débora recebe UUID novo; reinícios
do harness conservam a conversa daquela execução. “Nova conversa” continua
descartando o vínculo. A política futura de permissões não reutiliza concessões
apenas porque retomou um UUID salvo de outra execução.

O CLI verificado tem `--name`/`-n`. A implementação usa
`harness_session_name: "Débora Whisper"`, inclusive nas retomadas. É texto
literal e estável; `"Debora Whisper"` é a alternativa ASCII configurável.
Um formato com data como “Débora · 2026-10-10 16:40” pode ser escrito como nome,
mas expansão automática de data não faz parte desta entrega. Versões antigas
sem `--name` precisariam ser atualizadas ou ter uma adaptação de compatibilidade;
não há probe nem retry automático de flags nesta mudança.

## Etapas e critérios de aceite

1. Provar regras e precedência com o CLI real e ferramentas de teste sem efeitos
   externos; inventariar nomes/esquemas dos conectores habilitados. A leitura
   deve funcionar e nenhuma escrita deve escapar do controle.
2. Implementar política, catálogo e testes unitários de classificação; depois
   integrar estado pendente, overlay e voz, com logs reduzidos e migração.
3. Validar Gmail search/read, leitura de Calendar/Drive, WebSearch, Grep/Glob e
   `docker info`. Validar que Copy-Item pede confirmação, mostra sobrescrita e
   só executa os argumentos aprovados. Testar os dois shells e comandos compostos.
4. Testar sim/não, ambiguidade, eco, dois pedidos simultâneos, resposta tardia,
   timeout, TTS/STT indisponível, interrupção, fechamento, troca de modo/pasta,
   retomada e revogação. Nenhum teste automatizado envia mail ou cria evento real.
5. Testar nome MCP novo, mudança de esquema, settings inválidos, permissões
   herdadas amplas, caminho indireto e injeção de prompt. Envios, lixeira,
   exclusões e eventos continuam exigindo confirmação a cada chamada.

Esta fase só está pronta quando o usuário consegue consultar seus dados por voz
e consegue entender, negar e cancelar qualquer ação que altere ou exponha dados.
