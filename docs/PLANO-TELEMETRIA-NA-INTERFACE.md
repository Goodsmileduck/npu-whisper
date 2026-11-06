# Plano: telemetria e logs na interface

Objetivo: usar a Débora no dia a dia sem terminal aberto. A telemetria e os logs
que hoje só aparecem no console passam a ser vistos na bandeja e numa janela de
status. A ideia de TUI fica abandonada por enquanto.

## Restrição: não aumentar a dependência do Windows

**Por enquanto, nenhuma mudança deste plano pode aumentar a dependência do
Windows.** Todo código novo usa só Python puro, Tk/customtkinter, pystray e
Pillow, que são multiplataforma. Não se adicionam chamadas `ctypes.windll`,
DWM, registro do Windows nem APIs `win32`. Se algo só existir no Windows, fica
atrás de um `try/except` com alternativa neutra, como o overlay já faz.

### Situação atual (2026-10-10)

A interface só funciona no Windows. A parte visual seria portável, mas o app em
volta dela não roda no Linux:

- **Seria portável:** customtkinter/Tk, pystray e Pillow. As chamadas do
  Windows no overlay (DPI, estilo da janela, largura da tela) estão em
  `try/except` e têm alternativa em Tk.
- **Quebra no Linux:**
  - **Instância única:** usa um mutex do `kernel32` sem proteção
    (`app.py`, `_claim_single_instance`).
  - **Atalho global:** a biblioteca `keyboard` exige root no Linux e não
    funciona no Wayland.
  - **Digitar na janela ativa:** usa APIs do Windows (`GetForegroundWindow`,
    posição do cursor de texto, envio de teclas por `user32`).
  - **Efeito de vidro:** usa a API de janelas do Windows (`ui/glass.py`).
  - **CI:** os testes só rodam no Windows. No Ubuntu a CI só confere se o
    pacote instala.

Levar o app para o Linux é um projeto à parte. Este plano não deve piorar essa
situação.

## Como funciona hoje

- `log()` (`dictation_engine.py`) faz `print()` de tudo e grava em
  `~/.debora/logs/app.log` ou `telemetry.log`.
- A telemetria é só texto: a cada 5 s uma linha com estado, CPU, RAM do app,
  RAM do sistema e VRAM (CUDA) ou memória alocada na NPU, que o OpenVINO
  informa; a ocupação % da NPU só existe nos contadores do Windows e fica de
  fora.
- Pelo atalho do menu Iniciar o app abre com `pythonw`, que não tem console.
  Quem não abre o terminal não vê nada disso.

## Fase 1: dados em memória e informação rápida na bandeja (~meio dia a 1 dia)

- **Logs em memória:** `log()` passa a guardar as últimas ~2.000 linhas num
  buffer em memória (`collections.deque`). Cada linha leva hora, origem (app,
  telemetria, TTS, LLM) e nível (info, aviso, erro). Continua gravando os
  arquivos e imprimindo no console como hoje, então o `debora-cli` não muda.
- **Telemetria estruturada:** o loop de telemetria também atualiza um dicionário
  com os valores: estado, CPU, RAM, VRAM usada/total, underflows de áudio,
  dispositivo e modelo do STT, se houve fallback, e o RTF e a latência do último
  ditado. O app lê os valores dali, sem precisar interpretar o texto do log.
- **Bandeja:** o texto ao passar o mouse sobre o ícone mostra
  `Pronta · CPU 5% · VRAM 5,3/8,2 GB`. O menu ganha **"Abrir pasta de logs"**.
- **Cuidado com threads:** o Tk só pode ser usado pela thread principal. As
  outras threads só escrevem no buffer, e a interface lê o buffer com
  `root.after` a cada ~250 ms.

## Fase 2: janela "Status" (~1 a 2 dias)

Uma janela nova no mesmo estilo de Configurações e Histórico, aberta pelo item
**"Status"** no menu da bandeja, com duas abas:

- **Visão geral:**
  - estado atual;
  - hardware de cada modelo (STT em NPU/GPU/CPU, LLM e TTS) e se houve fallback;
  - CPU, RAM e VRAM com minigráficos dos últimos 5 min, desenhados em Canvas,
    sem dependência nova;
  - saúde do áudio (dispositivo de entrada, underflows);
  - voice chat: servidor de TTS no ar, backend do LLM e tempo até o primeiro
    áudio do último turno.
- **Logs:**
  - leitura ao vivo do buffer, com filtro por origem e nível e busca;
  - botão para pausar a rolagem automática, copiar e abrir o arquivo;
  - limite de linhas e inserção em lote, para não pesar a interface.

## Fase 3: logs dos outros processos e alertas de erro (~1 dia)

- **Outros processos:** a aba Logs passa a acompanhar também `tts_server.log` e
  `llm_server.log`, que vêm de processos separados, lendo o arquivo conforme ele
  cresce.
- **Alertas de erro:** hoje um erro só aparece no log. Passa a marcar o ícone da
  bandeja com um aviso e a destacar a linha na aba Logs. Ao abrir a janela
  Status, o aviso some.
- **Histórico de ditados:** a Visão geral ganha os tempos dos últimos ditados
  (duração, RTF, dispositivo).

## Regras gerais

- **Não aumentar a dependência do Windows** (ver a restrição acima).
- **Privacidade:** os logs trazem o texto ditado e a conversa. Tudo continua
  local, mas o botão "Copiar" deve deixar claro que copia esse conteúdo.
- Um PR por fase, cada um com testes:
  - buffer de logs e telemetria estruturada;
  - texto da bandeja;
  - filtros e limite de linhas da aba Logs;
  - leitura dos arquivos dos outros processos.

## Critérios de aceite

- [ ] Dá para usar o app no dia a dia sem nenhum terminal aberto e ver estado,
      recursos e erros pela bandeja e pela janela Status.
- [ ] Tudo o que hoje sai no console aparece na aba Logs.
- [ ] O `debora-cli` continua imprimindo no console como antes.
- [ ] A janela Status aberta não aumenta de forma perceptível a CPU do app
      (medir com a própria telemetria).
- [ ] Nenhuma chamada nova a API exclusiva do Windows (`windll`, `win32`, DWM,
      registro) no código adicionado.
