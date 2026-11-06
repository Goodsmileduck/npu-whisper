# Plano de Benchmark de Modelos ASR

Este documento define um benchmark reproduzível para comparar os modelos
disponíveis no projeto em CPU, GPU e NPU. O objetivo é medir separadamente
qualidade de transcrição, latência, consumo de recursos e estabilidade.

## Objetivos

O benchmark deve responder:

- qual modelo apresenta menor WER por idioma;
- qual combinação modelo/dispositivo apresenta menor latência;
- quais modelos processam áudio mais rápido que o tempo real;
- qual é o custo de carregamento e compilação inicial;
- qual é o comportamento após várias execuções consecutivas;
- quais combinações apresentam falhas, fallback ou `DEVICE_LOST`.

O **WER (Word Error Rate)** mede erros de substituição, exclusão e inserção de
palavras. Ele é principalmente uma característica do modelo e do conjunto de
dados; CPU, GPU e NPU devem ser comparados usando o mesmo modelo e os mesmos
áudios para medir o efeito da plataforma sobre desempenho e estabilidade.

## Matriz inicial

Como o foco prioritário é português brasileiro, inglês e áudio longo, o
primeiro ciclo deve cobrir:

| Plataforma | Modelo principal | Modelo de referência |
|---|---|---|
| CPU | Whisper `small` | Whisper `tiny` |
| GPU | Whisper `turbo` | Whisper `medium` |
| NPU | Whisper `small` | Whisper `base` |

Também deve ser executado um cenário complementar:

| Plataforma | Modelo |
|---|---|
| NPU + GPU/CPU | Parakeet TDT 0.6B |

O Parakeet deve ser reportado como **pipeline híbrido**: o pré-processamento
usa CPU, o encoder usa o dispositivo selecionado e o decoder usa GPU, com
fallback para CPU. Ele não deve ser apresentado como uma execução
exclusivamente NPU.

O `Whisper tiny` fica reservado para o cenário de baixo consumo e fallback. O
`Whisper turbo` é o principal candidato para qualidade multilíngue em GPU; o
`Whisper medium` funciona como referência de qualidade. O `Whisper small` é a
principal alternativa para NPU quando qualidade em fala contínua for mais
importante que a menor latência.

## Fontes de áudio

O benchmark não precisa de gravações próprias. Recomenda-se combinar três
camadas de dados:

1. **Dataset público de fala humana**, usado como resultado principal de WER.
2. **Áudio gerado por TTS**, usado para frases controladas e reproduzíveis.
3. **Sinais sintéticos determinísticos**, usados apenas para latência,
   compilação e estabilidade.

### Fala humana pública

Usar datasets que já forneçam áudio e transcrição de referência, como:

- Common Voice, especialmente para português;
- FLEURS, para cobertura multilíngue;
- LibriSpeech, para inglês;
- Multilingual LibriSpeech, quando a cobertura de idiomas for necessária.

Registrar no benchmark a fonte, a versão, a licença e a seleção de arquivos.
Uma amostra inicial razoável é de 30 a 50 arquivos por idioma; para uma
avaliação mais robusta, usar aproximadamente 100 arquivos por idioma.

Separar os arquivos por:

- duração curta: 2 a 5 segundos;
- duração média: 5 a 10 segundos;
- duração longa: 10 a 30 segundos;
- long-form: 30 segundos, 2 minutos, 5 minutos, 10 minutos e 30 minutos;
- ambiente silencioso;
- ruído moderado;
- fala rápida;
- locutores diferentes.

Os conjuntos principais devem conter amostras em português brasileiro e
inglês. O português europeu deve ser identificado separadamente, pois os
resultados publicados do Parakeet podem refletir essa variante e não devem ser
generalizados automaticamente para português brasileiro.

### TTS

Gerar frases específicas da aplicação, incluindo comandos curtos, ditado
contínuo, números, datas e nomes próprios. O texto usado para gerar cada
arquivo é a referência exata do teste.

Os resultados de TTS devem ser apresentados separadamente dos resultados de
fala humana. Vozes sintéticas são úteis para controle e repetibilidade, mas
não representam integralmente variação de pronúncia, ruído e prosódia humana.
Usar múltiplas vozes e velocidades para reduzir viés.

### Sinais sintéticos

Para medir exclusivamente desempenho, gerar localmente sinais determinísticos
com seed fixa, incluindo silêncio e ruído pseudoaleatório. Usar durações de
aproximadamente 2, 5, 9 e 16 segundos para exercitar os buckets do Parakeet.
Esses sinais não devem ser usados para calcular WER.

## Preparação dos áudios

Converter todos os arquivos para:

- WAV;
- mono;
- 16 kHz;
- PCM 16-bit.

Criar um manifesto versionado, sem incorporar necessariamente os áudios ao
repositório:

```csv
audio,reference,language,speaker_id,source
audio/pt_001.wav,"abra o navegador",pt,spk_001,common_voice
audio/pt_002.wav,"inicie o terminal",pt,spk_002,tts
```

Aplicar as mesmas regras à referência e à saída do modelo:

- converter para minúsculas;
- normalizar espaços;
- padronizar ou remover pontuação;
- definir o tratamento de números;
- definir o tratamento de hesitações;
- preservar a regra usada para acentos e caracteres especiais.

## Métricas

### Qualidade

- WER médio, mediano e p95;
- WER por idioma;
- WER por duração;
- CER (Character Error Rate), quando útil;
- taxa de transcrição vazia;
- taxa de falha;
- taxa de truncamento do Parakeet;
- idioma detectado, quando aplicável.

### Desempenho

Registrar:

- tempo de carregamento;
- tempo de compilação OpenVINO;
- tempo de pré-processamento;
- tempo de inferência;
- latência total;
- RTF (Real-Time Factor);
- mínimo, média, mediana, p95 e máximo;
- primeira execução (*cold run*);
- execuções seguintes (*warm runs*).

O RTF é calculado como:

```text
RTF = tempo de processamento / duração do áudio
```

Valores menores que 1 indicam processamento mais rápido que o tempo real.

Para áudios longos, medir duas estratégias separadamente:

1. transcrição nativa do backend, quando suportada;
2. chunking com segmentos de 10 a 15 segundos e sobreposição controlada.

Não comparar um modelo usando áudio inteiro com outro usando segmentos sem
registrar essa diferença no resultado.

### Recursos e estabilidade

Registrar, quando disponível:

- memória RAM;
- uso da CPU;
- uso da GPU;
- estado e uso da NPU;
- temperatura;
- potência ou energia;
- falhas de carregamento;
- fallback de dispositivo;
- ocorrências de `DEVICE_LOST`;
- duração da execução contínua.

## Protocolo experimental

Para cada combinação de modelo e dispositivo:

1. Registrar Windows, driver, Python, OpenVINO, `openvino-genai`, hardware e
   versão ou hash do modelo.
2. Confirmar que o modelo está baixado e que a configuração de idioma é
   válida.
3. Executar uma rodada fria, medindo carregamento e compilação.
4. Executar de 3 a 5 rodadas de aquecimento.
5. Executar de 3 a 5 rodadas medidas por arquivo.
6. Usar exatamente os mesmos arquivos, idioma e normalização em todas as
   plataformas.
7. Separar resultados de cold start e warm runs.
8. Salvar transcrições, métricas, logs e erros.
9. Repetir o ciclo quando houver variação significativa.

Durante o teste, registrar o perfil de energia e reduzir interferências de
aplicativos em segundo plano, atualizações, gravação de tela e outros
processos que usem CPU, GPU ou NPU.

## Formato dos resultados

O benchmark genérico deve produzir um CSV ou JSON por execução com campos
equivalentes a:

```csv
run_id,model,device,backend,audio_file,language,audio_duration_s,
load_time_s,preprocess_s,inference_s,total_time_s,rtf,
wer,cer,ram_mb,cpu_percent,gpu_percent,npu_status,
truncated,error
```

O relatório agregado deve conter:

```csv
model,device,language,samples,
wer_mean,wer_median,wer_p95,
latency_mean_s,latency_median_s,latency_p95_s,
rtf_mean,rtf_p95,failure_rate
```

Guardar também a transcrição produzida e a referência correspondente para
permitir auditoria manual. Qualquer texto de transcrição exportado para CSV
deve ser tratado como dado não confiável ao abrir em planilhas, evitando
interpretação como fórmula.

## Implementação planejada

Adicionar um benchmark genérico em:

```text
benchmarks/benchmark_asr.py
benchmarks/datasets/manifest.csv
benchmarks/results/
```

O script deve:

- carregar qualquer modelo de `MODEL_REGISTRY`;
- aceitar `--model`, `--device`, `--manifest`, `--runs` e `--output`;
- processar arquivos do manifesto;
- gerar transcrições;
- calcular WER e CER;
- separar cold start de warm runs;
- registrar fallback e falhas;
- gerar CSV e resumo no terminal.

Exemplos:

```powershell
python benchmarks\benchmark_asr.py `
  --model base `
  --device NPU `
  --manifest benchmarks\datasets\manifest.csv `
  --runs 5 `
  --output benchmarks\results\base_npu.csv
```

```powershell
python benchmarks\benchmark_asr.py `
  --model parakeet `
  --device NPU `
  --manifest benchmarks\datasets\manifest.csv `
  --runs 5 `
  --output benchmarks\results\parakeet_npu.csv
```

O `benchmarks\bench_parakeet.py` existente deve continuar sendo usado para
medições específicas de compilação por buckets e latência por estágio do
Parakeet. Antes de usar o Parakeet como solução de áudio longo, o benchmark
deve confirmar que a implementação não trunca a entrada: o maior bucket atual
é de aproximadamente 16 segundos. Até existir chunking com recomposição de
texto, entradas maiores devem ser marcadas como incompatíveis com o cenário
long-form, e não comparadas como se fossem transcrições completas.

O Qwen3-ASR-0.6B deve ser mantido como prova de conceito separada. Só deve
entrar na matriz oficial após validação do backend, conversão/quantização para
OpenVINO, execução em Windows e disponibilidade real no NPU.

## Relatório final

O relatório deve conter pelo menos:

1. **Qualidade** — modelo, dispositivo, idioma, amostras, WER médio, WER p95 e
   falhas.
2. **Latência** — cold start, latência média, p95 e RTF.
3. **Recursos** — RAM, CPU, GPU e NPU.
4. **Estabilidade** — falhas, fallback, `DEVICE_LOST` e truncamentos.
5. **Recomendação por cenário** — menor latência, melhor qualidade e melhor
   equilíbrio.

Não declarar um modelo superior combinando WER de um dataset com latência de
outro. Resultados de fala humana, TTS e sinais sintéticos devem permanecer
identificados separadamente.

## Critérios iniciais de decisão

Os seguintes limites são referências para interpretar os resultados, não
garantias do hardware:

- tempo real: RTF menor que 0,5;
- uso confortável em comandos curtos: p95 de latência abaixo de 1 segundo;
- estabilidade: zero `DEVICE_LOST` em uma bateria longa;
- comparação de qualidade: menor WER no mesmo idioma, dataset e normalização;
- Parakeet: comparar apenas nos idiomas que ele suporta e reportar sua
  arquitetura híbrida;
- português: reportar português brasileiro e europeu separadamente;
- áudio longo: rejeitar ou marcar explicitamente execuções truncadas;
- qualidade máxima: priorizar menor WER em Whisper turbo/medium;
- equilíbrio no NPU: priorizar Whisper small, desde que o p95 de latência e o
  WER permaneçam aceitáveis.
