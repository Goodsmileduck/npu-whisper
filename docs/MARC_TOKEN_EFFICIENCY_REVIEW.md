# Revisão do uso de tokens do mARC neste trabalho

Data: 2026-10-05. Autor: Codex, a pedido do usuário.

Este documento registra uma análise preliminar para obter uma segunda opinião
do Claude Code e preparar uma possível atualização do mARC. Não é uma medição
de faturamento nem uma especificação aprovada para implementar mudanças.

## Pedido e contexto

O usuário relatou que o consumo aumenta muito quando adiciona o mARC ao chat.
Pediu uma análise do processo usado nesta conversa e, depois, que o relatório
fosse salvo no clone com contexto suficiente para validação independente.

O trabalho original foi no `npu-whisper`: melhorar latência e suavidade do
aviso sonoro, adicionar testes no GitHub Actions e corrigir problemas de
gravação, encerramento e recuperação após falha fatal da GPU OpenVINO.
Depois da invocação explícita de `$marc:tech-lead`, foram usados especialistas
via `dispatch_agent.py`, com implementação, revisão de segurança e revisão
de correção funcional.

- Issue principal: https://github.com/alexandre-machado/debora-whisper/issues/11
- PR: https://github.com/alexandre-machado/debora-whisper/pull/12
- Commit final revisado: `c95350b6ab7a312488a384cf571c78086f9f38b9`.
- Base do trabalho: `bab51c3c2b51857788cb64b4c0eb3bf18cb17bb3`.
- Houve três rodadas de implementação/revisão; seis relatórios de revisão.
- A validação final teve 154 testes passando e CI verde em Python 3.10/3.12/3.14.
- As revisões encontraram problemas reais. Seu custo não deve ser tratado
  integralmente como desperdício.

Ao salvar este relatório, o checkout principal está em `fix/audio-recovery-11`.
O usuário pediu explicitamente que a branch fosse trazida para a pasta principal.
As sete alterações locais anteriores foram preservadas em
`backup/local-before-audio-recovery-20261005-142304` e também em um stash
identificado pela mensagem `Preserved local changes before audio recovery switch`.
Na ocasião ele era `stash@{0}`; esse índice pode mudar. A árvore do backup e a do
stash foram comparadas e eram idênticas. Não houve merge no GitHub.
O worktree `.worktrees/audio-recovery-11` ficou em detached HEAD no mesmo commit.

## Como consultar este chat diretamente

Identificador da conversa, confirmado pelas variáveis locais
`CODEX_THREAD_ID` e `CODEX_SESSION_ID`:

```text
01a1075e-bfcf-72a1-b3fb-a51890e697dd
```

Registro local encontrado:

```text
C:/Users/alexandre-machado/.codex/sessions/2026/10/04/rollout-2026-10-04T11-43-32-01a1075e-bfcf-72a1-b3fb-a51890e697dd.jsonl
```

Não copiei o registro para o repositório: ele contém contexto da sessão,
instruções e resultados de ferramentas que não precisam ser publicados.
O caminho acima permite leitura local pelo Claude Code, se tiver acesso.
Se o arquivo não estiver disponível em outra máquina, não inferir seu conteúdo;
usar os artefatos abaixo e declarar a limitação.

Âncoras encontradas na inspeção de 2026-10-05, com timestamps UTC:

| Linha aproximada | Timestamp | Conteúdo |
| --- | --- | --- |
| 345 | 2026-10-04T21:11:52.435Z | Usuário invoca `$marc:tech-lead` |
| 1136 | 2026-10-05T17:28:22.720Z | Usuário pede análise da ineficiência de tokens |
| 1174 | 2026-10-05T17:30:26.393Z | Resposta com o relatório inicial e a expressão `7.670 palavras` |
| 1181 | 2026-10-05T17:36:04.408Z | Usuário pede para salvar o relatório e permitir validação no chat |

As linhas podem mudar se o registro for reescrito. Prefira buscar por conteúdo
e timestamp. Leia intervalos pequenos, não o JSONL inteiro no contexto do modelo.
Nos registros `response_item`, as mensagens têm `payload.type == "message"`,
`payload.role` e uma lista `payload.content` com campos de texto. Há também
chamadas/resultados de ferramentas e um evento `compacted`; diferencie o
histórico original de resumos antes de atribuir ações ou contar ocorrências.

### Correção importante sobre disponibilidade de métricas

O relatório inicial não calculou tokens faturados. Os oito resultados JSON
de especialistas inspecionados não têm campos de tokens, cache ou custo.
Isso **não significa que não exista telemetria em nenhum lugar**.

Ao preparar este documento, encontrei no registro do Codex 151 entradas do tipo
`token_usage_record` e 154 eventos `event_msg` com `payload.type == "token_count"`.
Esses são números de registros, não quantidades de tokens. Não inspecionei sua
semântica nem somei seus valores. O arquivo continua crescendo durante a sessão.

O revisor pode investigar essa telemetria para melhorar a análise. Antes de
somar, deve identificar contadores cumulativos versus incrementais, registros
duplicados do mesmo uso, cache e limites de cobertura. Não presumir que esses
eventos incluam as execuções externas do Claude Code. Não atribuir todo o consumo
posterior à invocação da skill ao mARC: escopo, revisões, correções e tamanho do
histórico também mudaram. Não existe neste trabalho um experimento equivalente
com e sem mARC.

## Evidências locais

Versão do plugin inspecionada:

```text
C:/Users/alexandre-machado/.codex/plugins/cache/nexaduo/marc/26.10.3
```

Arquivos relevantes, relativos à raiz acima:

- `skills/tech-lead/SKILL.md`: fluxo e leitura antes de cada dispatch, linha 74.
- `skills/tech-lead/references/dispatch.md`: modelos, limites, delegação e guard.
- `skills/tech-lead/references/review-release.md`: revisões independentes no HEAD.
- `scripts/dispatch_agent.py`: construção do comando, linha 292; para Claude Code,
  linha 302; execução e captura do resultado, função `dispatch`.

Artefatos do trabalho, relativos ao clone:

- `.venv/marc/dev-prompt.txt`, `dev-pr12-fixes.txt`, `dev-pr12-final-fixes.txt`.
- `.venv/marc/dev-pr12-fixes-result.json` e `dev-pr12-final-fixes-result.json`.
- `.venv/marc/sec-pr12-result.json`, `sec-pr12-round2-result.json`,
  `sec-pr12-round3-result.json`, e os equivalentes com prefixo `rev`.
- `.venv/marc/sec-pr12-review.md`, `sec-pr12-round2-review.md`,
  `sec-pr12-round3-review.md`, e os equivalentes com prefixo `rev`.
- `.venv/marc/pr12.diff`, `pr12-head2.diff`, `pr12-head3.diff`,
  `pr12-delta2.diff`, `pr12-delta3.diff`.
- Perfis usados: `.worktrees/audio-recovery-11/.claude/agents/engineer.md` e
  `.venv/marc/review-harness/.claude/agents/{security,review}.md`.

Esses artefatos são locais e ignorados pelo Git; não viajarão automaticamente
com um clone novo. Os perfis foram copiados para estabelecer os limites de
ferramentas; o cache instalado do plugin não foi editado.

## Achados preliminares

### 1. Repetição de uma revisão adicional sabidamente inconclusiva

**Evidência forte de desperdício evitável.** Os relatórios de segurança das
três rodadas dizem que `/security-review` leu o checkout principal em vez do
worktree do PR, retornando contexto/diff vazios. O resultado não foi usado
para o parecer, que se apoiou na revisão manual.

Ver `sec-pr12-review.md:7`, `sec-pr12-round2-review.md:11` e
`sec-pr12-round3-review.md:9`. Os prompts posteriores já reconheciam o problema,
mas ainda permitiam tentar o passe adicional.

Proposta: validar diretório e commit antes do passe; após uma falha estrutural,
não repetir sem mudança na configuração que explique por que funcionará.
Parte desta ineficiência foi decisão do operador Codex, não apenas regra do plugin.

### 2. Relatórios extensos retornando ao contexto do operador

**Volume medido, economia ainda não medida.** Os seis arquivos `*review.md`
listados acima somam 7.670 palavras, por divisão simples em espaços, e 51.723
bytes no disco. Não são 7.670 tokens. Os relatórios foram lidos integralmente
pelo operador e voltaram a servir de entrada para rodadas posteriores.

Proposta: resumo de até 300 palavras, com veredito, SHA, achados novos e
disposição dos pendentes. Preservar detalhes em arquivo para leitura seletiva.
Evitar repetir descrições de correções aceitas, limitações inalteradas e
relatos operacionais sem impacto no parecer.

### 3. Sessões novas nas continuações do mesmo especialista

**Comando observado; impacto quantitativo não estabelecido.** Os oito resultados
JSON disponíveis não contêm `--resume`. O despachante constrói o comando
`claude --dangerously-skip-permissions --agent <perfil> -p <prompt>`.
Os prompts das continuações apontam para relatórios anteriores e reconstituem
o contexto da tarefa. O histórico inclui ainda a implementação inicial, cujo
resultado JSON não está entre esses oito arquivos.

Proposta: considerar retomada do mesmo especialista ou um estado compacto de
continuação. Não afirmar que retomar sempre economiza: uma sessão longa também
pode carregar contexto desnecessário. Medir ambas as opções, inclusive cache.
Manter independência entre implementador e revisores.

### 4. Revisões adicionais dentro dos especialistas revisores

**Duplicação de processo confirmada; utilidade variável.** Os perfis combinam
checklist manual e `/security-review` ou `/code-review`. O relatório de código
da segunda rodada registra uma execução adicional e um achado depois
confirmado manualmente; portanto, nem todo passe adicional foi inútil.
Na terceira rodada, o revisor de código deixou de executar esse passe.

Proposta: explicitar quando o passe adicional é necessário, qual pergunta
responde e como evitar repetir o mesmo exame. Considerar também a compatibilidade
com os limites de ferramentas e a proibição de subdelegação do dispatch.

### 5. Instruções extensas e repetidas

**Regra confirmada; frequência real de releitura precisa de auditoria.**
`SKILL.md:74` manda ler `dispatch.md` antes de cada dispatch. O arquivo tinha
1.807 palavras na contagem local. Ele reúne regras atuais, explicações e
histórico de regras substituídas.

Proposta: leitura uma vez por versão/sessão, com uma lista curta por execução;
separar procedimento ativo de histórico. Não transformar tamanho de arquivos
em tokens consumidos sem verificar quais leituras realmente ocorreram.

### 6. Métricas e limites insuficientes no despachante

**Confirmado no código e nos resultados examinados.** O wrapper registra
`duration_sec`, status, comando, stdout e stderr, mas não normaliza métricas
de uso. Os prompts têm limites de chamadas e critérios de parada; o wrapper
inspecionado aplica timeout via subprocesso, sem contador de chamadas ou
orçamento de tokens implementado nele.

Proposta: capturar métricas estruturadas quando o executor oferecer suporte,
registrar identidade da sessão e distinguir consumo do operador/especialistas.
Verificar limites disponíveis no executor antes de projetar flags ou garantias.
Timeout de execução não equivale a orçamento de tokens.

### 7. Modelo padrão e custo não são a mesma coisa que volume de tokens

Os três perfis locais usados declaram `model: opus`, de acordo com a regra
de `dispatch.md:47`. Isso confirma a configuração, não uma medição do modelo
efetivamente servido ou de cobrança. A seleção deve ser avaliada conforme
complexidade e risco; não houve aqui comparação controlada entre modelos.
Não usar preços atuais ou argumentos de benchmarks citados na skill como
evidência validada sem consultar suas fontes.

## Responsabilidade do operador e valor das revisões

O Codex reconhece que contribuiu para o consumo: aceitou relatórios extensos,
trouxe conteúdo integral ao contexto principal e repetiu passes inconclusivos.
Até durante esta auditoria uma leitura imprudentemente ampla imprimiu campos
`command` e `command_str` com prompts duplicados; a leitura seguinte selecionou
apenas métricas e metadados necessários. A lição aplica-se ao operador também.

Por outro lado, as revisões localizaram falhas reais: fallback da GPU engolia
erro fatal, callbacks mantinham referência ao engine antigo, havia condições
de corrida na troca de configurações/saída de texto e problemas na contagem
de loaders e em callbacks sob lock. Parte do retrabalho era necessária.
Dois pareceres ADVISE também motivaram correções adicionais por decisão do
operador. Não concluir que todo ADVISE deveria encerrar o trabalho: avaliar
gravidade e relação com o objetivo.

As revisões finais estão publicadas em:

- https://github.com/alexandre-machado/debora-whisper/pull/12#issuecomment-5998798281
- https://github.com/alexandre-machado/debora-whisper/pull/12#issuecomment-5998798701

## Pedido para a segunda opinião do Claude Code

1. Validar ou refutar cada achado usando os arquivos e, quando necessário,
   os trechos relevantes do chat. Não tratar este relatório como autoridade.
2. Separar defeito do mARC, decisão do operador e custo necessário da tarefa.
3. Investigar a telemetria disponível sem dupla contagem e informar lacunas;
   não apresentar percentual de economia ou faturamento sem medição adequada.
4. Propor mudanças pequenas e priorizadas, preservando revisão independente,
   identificação do SHA e isolamento de trabalho quando necessário.
5. Dizer quais propostas exigem mudança de código, instruções ou configuração.
   Prioridades iniciais sugeridas: impedir repetição inconclusiva, compactar
   retornos dos especialistas e instrumentar uso por execução.

Esta etapa é uma revisão para preparar uma atualização, não autorização para
editar o cache instalado, alterar a aplicação, publicar conteúdo do chat ou
enviar uma contribuição upstream automaticamente. Trabalhar no repositório-fonte
do mARC deve ser uma etapa posterior claramente delimitada.
