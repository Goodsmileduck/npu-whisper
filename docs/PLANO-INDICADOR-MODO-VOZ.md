# Indicador do modo de voz

Data: 2026-10-10
Status: implementado

Um toque no Alt direito (`voice_chat_hotkey`) alterna entre ditado e conversa
com a inteligência. Os indicadores refletem a configuração inicial e mudam
assim que o modo é trocado, inclusive pelas configurações. A troca pelo Alt
direito ou pela bandeja também liga a escuta contínua se o microfone estiver
parado; o Ctrl+Espaço continua só ligando e desligando a captura, sem mudar o
modo.

- **Bandeja:** durante a escuta/gravação, verde indica ditado (`#047857` no
  silêncio, `#10B981` com voz ativa); roxo indica conversa (`#7E22CE` no
  silêncio, `#A855F7` com voz ativa). Os frames são pré-renderizados.
- **Demais estados:** pronto continua cinza, processamento ciano, fala azul,
  carregamento dourado e erro vermelho. O menu acompanha o modo selecionado.
- **Balão:** no modo conversa, um brilho roxo (`#A855F7`) acompanha a beirada
  do balão em volta da mascote, com o mesmo raio da pílula: forte junto da
  borda e esfumaçado para dentro em cerca de 4 px lógicos, em todos os estados.
  A janela recorta as quinas por cor-chave, sem alfa parcial, então elas
  serrilham; por isso fica 1 px escuro fora do roxo, e a curva do roxo é
  desenhada em 4x e suavizada contra o painel. Com o
  balão recolhido, contorna o balão inteiro. No ditado, não há brilho. Acompanha
  o DPI, não muda o tamanho da janela nem o espaço do texto, e fica em cache
  porque o balão redesenha a cada quadro da mascote.
- **Ao abrir:** se o app inicia já escutando (`continuous_listening`), ele
  sempre começa no ditado, mesmo que o modo conversa tenha ficado salvo. O
  `--voice-chat` na linha de comando continua forçando a conversa. Mesmo
  começando no ditado, o TTS e o LLM são carregados na abertura; sem isso, o
  primeiro toque no Alt esperava ~20 s pelo TTS, e o texto da Débora (que só
  entra no balão junto com o áudio) não aparecia.

Sem selo (badge), toast, som de troca ou `winsound`, e sem novas APIs exclusivas
do Windows, conforme [o plano de telemetria](PLANO-TELEMETRIA-NA-INTERFACE.md).

Cobertura: cores e cache dos ícones, troca de modo na thread da bandeja,
integração no app e desenho da borda sem alteração de geometria em vários DPIs.
