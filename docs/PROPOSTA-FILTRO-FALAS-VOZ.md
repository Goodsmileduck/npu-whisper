# Proposta: ignorar falas de voz que não são para a Débora

Data: 2026-10-09
Status: proposta (não implementada; código do debora-whisper ainda não inspecionado)

## Problema

1. O Claude recebe o texto transcrito pelo Whisper sem marca de origem. Não distingue
   fala de texto digitado no console e responde a tudo o que o microfone capta,
   inclusive vídeos ou conversas paralelas. Exemplo: "Um novo teste trouxe mais uma
   questão interessante.", que o usuário não reconheceu como dele.
2. Sons sem conteúdo viram mensagens. Exemplo: um pigarro transcrito como "Ahem."
   (2026-10-09 20:17:37) gerou uma resposta.

## Parte A: filtro de sons sem conteúdo (antes de enviar ao Claude)

Observado no app.log para o pigarro: VAD Max prob 0.881 (fala normal: 1.000),
Max amp 0.2148, segmento de 2.6s, texto final 'Ahem.'.

Filtros sugeridos no debora-whisper:
- Lista de interjeições descartadas quando são o texto inteiro: ahem, hum, hmm, cof,
  ah, eh (normalizar caixa e pontuação).
- Confiança do faster-whisper por segmento: descartar se `no_speech_prob` alto ou
  `avg_logprob` baixo.
- VAD: descartar segmentos cujo pico de probabilidade fique abaixo de um limite
  (ex.: 0.95, a calibrar).
- Registrar no log, para cada fala, VAD prob, amplitude, duração, `no_speech_prob`
  e `avg_logprob`, para calibrar os limites com pigarros e tosses de teste.
- Fala descartada: registrar `Voice chat: descartada (sem conteúdo): '...'` e não enviar.

## Parte B: falas não endereçadas à Débora

1. **Etiqueta na origem.** Antes de enviar ao Claude, a Débora prefixa cada fala:
   `[voz | chamou_debora=sim | seg_desde_ultima_resposta=8]`
   Mensagens digitadas no console seguem sem etiqueta.
2. **Regra nas instruções do Claude.** Mensagem com etiqueta de voz que não chama a
   Débora pelo nome e não continua a conversa em andamento: responder apenas
   `[ignorar]`. Mensagens sem etiqueta (console) são sempre atendidas.
3. **Tratamento do token.** Ao receber `[ignorar]`, a Débora não sintetiza voz e
   registra `Voice chat: descartada (não endereçada): '...'`.

## Pontos em aberto

- Janela de "continuação da conversa" (segundos após a última resposta).
- Detecção do nome tolerante a erros do Whisper ("Débora", "Debora", "Dé").
- Alternativa futura: classificador local leve antes do Claude, para economizar
  latência e tokens.

## Próximo passo

Ler o código em `D:\repos\alexandre-machado\debora-whisper\debora_whisper`
(leitura bloqueada por permissão em 2026-10-09) para localizar onde a transcrição
final é enviada ao Claude e onde as respostas vão para o TTS.
