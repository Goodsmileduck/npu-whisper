# Modelos de Decisão/Classificação e Detecção de Fim de Turno (VAD) — 2026-10-10

> **Nota de autoria:** Pesquisa, análise técnica e documentação elaboradas pelo assistente de IA **Antigravity** (Google DeepMind) em sessão de pair programming com Alexandre Machado, a partir de pesquisa na web e do benchmark publicado por Fabio Akita em 10/10/2026.
>
> **Verificação:** o artigo citado, o serviço Jev e os modelos de decisão (Jebadiah, Kev, Mapika, Eikos, Laya, Clef) e seus números não foram conferidos de forma independente neste repositório. O Smart Turn v3 (Pipecat, BSD-2) é o único candidato já confirmado; validar os demais antes de usar.

## 1. Contexto e Retificação do Histórico

No documento anterior ([turn-detection-2026-10-09.md](turn-detection-2026-10-09.md)), registrou-se que o termo "JEV" recebido por comando de voz não havia sido identificado e fora presumido como erro de transcrição para "TEN VAD".

Em 10/10/2026, o artigo de Fabio Akita (*[Qual modelo de decisão/classificador de IA tipo Jev é melhor?](https://akitaonrails.com/2026/10/10/qual-modelo-de-decisao-classificador-de-ia-tipo-jev-e-melhor/)*) detalhou a nova safra de **modelos de decisão/classificadores estruturados** baseados no protocolo *SystemOne*, iniciada pelo serviço proprietário **Jev** (TypeSafe) e acompanhada por alternativas abertas para execução local (**Jebadiah 4B**, **Kev**, **Mapika Decider 4B**, **Eikos 4B**, **Laya**, **GLiNER** e **Clef**).

Este relatório avalia a aplicabilidade desses modelos no Débora Whisper frente aos desafios de VAD (Voice Activity Detection), interrupções de fala e detecção de fim de turno (*turn detection* / *endpointing*).

---

## 2. Distinção Arquitetural no Débora Whisper

Para avaliar qualquer classificador, é essencial diferenciar onde cada modelo opera no pipeline:

```
[ Microfone: 16 kHz PCM ]
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. VAD Acústico por Frame (Tempo Real Estrito: ~32 ms)       │
│    Atual: Silero VAD v5 (ONNX em CPU)                        │
│    Função: Detecção binária de voz vs ruído/silêncio        │
│    * Modelos tipo Jev NÃO rodam aqui (não recebem áudio)    │
└─────────────────────────────────────────────────────────────┘
            │
            ├─────────────────────────────────────────┐
            │ Silêncio inicial (~200–300 ms)          │ Fala ativa
            ▼                                         ▼
┌─────────────────────────────────────────┐   ┌───────────────────────────┐
│ 2. Fim de Turno Acústico / Prosódico     │   │ Whisper ASR (CUDA / NPU)  │
│    Candidato: Smart Turn v3 (Pipecat)   │   └─────────────┬─────────────┘
│    Função: Entonação/ritmo nos últimos  │                 │
│    8s de áudio. Decide se encerra cedo. │                 ▼
└─────────────────────────────────────────┘   ┌───────────────────────────┐
                                              │ 3. Endpointing Semântico  │
                                              │    Atual: regex em texto  │
                                              │    Candidato: Laya / Kev  │
                                              └───────────────────────────┘
```

1. **VAD Acústico Contínuo**: Processa fatias de 32 ms (512 amostras) 30 vezes por segundo. Precisa de latência sub-milissegundo e consumo de CPU desprezível. **Modelos tipo Jev não operam nesta camada**, pois operam sobre texto.
2. **Turn Detection Acústico (Prosódia de Áudio)**: Analisa o áudio bruto recente (2 a 8 segundos) durante uma pausa curta para determinar se o falante concluiu o pensamento ou fez apenas uma pausa para respirar.
3. **Endpointing Semântico (Texto do Rascunho ASR)**: Avalia o texto intermediário emitido pelo Whisper. Hoje feito heuristicamente via regex em `vad_endpoint.py:incomplete_sentence()` (checando conjunções finais como `"e"`, `"mas"`, `"porque"`).

---

## 3. Análise dos Modelos de Decisão/Classificação (Benchmark Akita)

Resultados sintetizados do benchmark executado localmente em AMD Strix Halo (64 GB) e nuvem:

| Modelo | Parâmetros | Onde Roda | Acerto PT-BR (Compacto) | Latência Mediana | Requisitos de Memória | Adequação ao Débora Whisper |
|---|:---:|:---:|:---:|:---:|---|---|
| **Jev 1.13.0** | Proprietário | API (Nuvem) | **97,6%** | ~255 ms (WAN) | US$ 0,19 / 10k decisões | Inviável offline; adiciona dependência de nuvem e latência de rede. |
| **Clef (27B)** | 27B | API / Local | 94,0% | ~302 ms (WAN) | ~54 GB VRAM em BF16 | Pesado demais para máquina cliente comum. |
| **Jebadiah 4B v2** | 4B | Local | **89,3%** | ~205 ms (LAN) | ~8 GB VRAM | **Inviável para loop de voz**: consome 8 GB de VRAM (concorrendo com o Whisper) e adiciona 200 ms por rascunho. |
| **Kev 4B** | 4B | Local | 86,9% | ~246 ms (LAN) | ~8 GB VRAM | Mesma restrição de pegada de memória do Jebadiah. |
| **Mapika Decider 4B**| 4.2B | Local | 85,7% | ~186 ms (LAN) | ~8 GB VRAM | Mais rápido dos 4B, mas ainda requer GPU dedicada grande. |
| **Eikos 4B** | 4B | Local | 79,8% | ~235 ms (LAN) | ~8 GB VRAM | Treinado em EN/PT-BR, licença MIT, mas mantém peso de 4B. |
| **Kev 0.8B** | 0.8B | Local | 61,9% | ~86 ms (LAN) | ~1.6 GB VRAM ou CPU | Latência baixa, porém acerto genérico cai significativamente. |
| **Laya multilingual**| 322M | Local | 38,1% | **19 ms (LAN)** | **< 2 GB VRAM / CPU ONNX** | **Único com latência e consumo compatíveis**. Zero-shot fraco, mas viável com fine-tuning direcionado. |
| **GLiNER2.5-multi** | 287M | Local | 35,7% | ~63 ms (LAN) | < 2 GB VRAM | Focado em extração/NER; inadequado para decisão de turno. |

### Veredito sobre a Safra Jev no Débora Whisper
- **Modelos 4B (Jebadiah, Kev, Mapika)**: Embora entreguem 85–90% de acerto em tarefas complexas, são inviáveis para rodar a cada pausa de fala no desktop: alocam 8 GB de VRAM adicional e demoram ~200 ms.
- **Laya (ModernBERT / mmBERT)**: É a única arquitetura dessa safra cujo perfil computacional (19–40 ms, execução viável em ONNX na CPU) permite coexistir com o Whisper sem esgotar recursos. Contudo, seu checkpoint de fábrica erra tarefas genéricas; seu uso só faz sentido se treinado especificamente para a classificação binária de completude de frase (`completa` vs `incompleta`).

---

## 4. Classificadores Específicos para Fim de Turno / VAD

Para o problema real do aplicativo (evitar cortes prematuros durante o ditado e chat de voz sem introduzir atrasos longos de silêncio fixo), os modelos de fim de turno acústico/prosódico são a resposta direta:

### A. Smart Turn v3 (Pipecat) — Candidato Principal
- **Tipo**: Classificador prosódico/acústico sobre os últimos 8 segundos de áudio bruto.
- **Tamanho**: **8 MB** (ONNX quantizado em INT8).
- **Consumo**: **12 a 95 ms em CPU pura** (não toca a GPU/NPU do Whisper).
- **Acurácia em PT-BR**: **95,42%** com apenas **2,79% de cortes precoces** (dados medidos em benchmark público pela Daily/Pipecat).
- **Licença**: BSD-2 (totalmente permissiva).
- **Arquitetura de integração**:
  - Silero VAD segue ativo no loop de 32 ms para detecção rápida de silêncio.
  - Ao detectar silêncio breve (~200–300 ms), o Smart Turn v3 é consultado:
    - Se $p(\text{completo}) > 0.5$: encerra o turno imediatamente (economiza 300–600 ms de latência percebida).
    - Se $p(\text{incompleto})$: aguarda o teto de silêncio estendido (1.5s–2.0s), permitindo que o usuário respire ou elabore o pensamento sem interrupção.

### B. LiveKit Turn Detector
- **Tipo**: Modelo multimodal acústico/semântico otimizado para inferência de CPU.
- **Vantagem**: Reduz interrupções mantendo ritmo natural de conversação em 14+ línguas.
- **Consideração**: Peso ligeiramente maior e ecossistema atrelado ao LiveKit Agents.

### C. TEN VAD
- **Tipo**: VAD acústico puro por frame (10/16 ms), concorrente direto do Silero.
- **Vantagem**: RTF 0.0086 vs 0.0127 do Silero.
- **Desvantagem**: Não resolve semântica/prosódia de pausas e possui restrições de licença concorrencial com a Agora.

---

## 5. Matriz Comparativa de Abordagens

| Abordagem | Modelo | Entrada | Onde Roda | Latência | Custo de RAM/VRAM | Impacto no Débora Whisper |
|---|---|---|---|---|---|---|
| **VAD Acústico Atual** | Silero VAD v5 | Áudio (32 ms) | CPU (ONNX) | < 1 ms | ~2 MB | Base contínua mantida; não sabe quando a frase terminou. |
| **Turn Detection Acústico** | **Smart Turn v3** | Áudio (últimos 8s) | CPU (ONNX) | 12–95 ms | **8 MB** | **Máximo**: Evita cortes prematuros em PT-BR sem custo de GPU. |
| **Endpointing de Texto (Atual)** | Heurística regex | Texto transcrito | CPU (Python) | < 0.1 ms | 0 MB | Básico: só verifica pontuação e conjunções no final. |
| **Endpointing Semântico IA** | Laya (ajustado) | Texto transcrito | CPU (ONNX) | ~20 ms | ~600 MB | Alto: substitui o regex por entendimento de contexto do texto. |
| **Decisão Geral (Jev-like)** | Jebadiah / Kev 4B | Texto estruturado | GPU (CUDA/ROCm) | ~200 ms | ~8 GB | Inviável no desktop: pesado demais para rodar em paralelo. |

---

## 6. Próximos Passos Recomendados

1. **Prioridade 1 (Turn Detection Acústico)**: Validar o **Smart Turn v3** como complemento ao Silero VAD, conforme o plano de testes traçado em [turn-detection-2026-10-09.md](turn-detection-2026-10-09.md), seção "Plano de teste offline".
2. **Prioridade 2 (Endpointing Textual)**: Manter a heurística em `vad_endpoint.py` no curto prazo; caso se deseje sofisticação semântica no texto transcrito, prototipar um classificador baseado no **Laya** (exportado para ONNX int8 em CPU) ajustado especificamente para detecção de completude de sentenças em português.
