# NPU DEVICE_LOST com whisper-turbo — 2026-10-06

## Sintoma

Quase toda inicialização carregava o `whisper-turbo-openvino` do cache em cerca de 3s. Poucos segundos depois, no aquecimento ou numa das primeiras transcrições, a NPU caía com `L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST`. Isso acontecia antes do modo contínuo existir.

## Causa

A causa era o export do modelo, `FluidInference/whisper-large-v3-turbo-int4-ov-npu`. App, cache e OpenVINO foram descartados. O teste que separou as variáveis carrega o `WhisperPipeline` num script isolado, com o app fechado, e chama `generate` em loop. A entrada é fala real (TTS do Windows, 16 kHz), em recortes que crescem de 1s a 26s, imitando os rascunhos do modo contínuo.

| Modelo | OpenVINO | Cache | Resultado na NPU |
|---|---|---|---|
| turbo int4 FluidInference | 2025.4.1 | existente | 40/40 com ruído (2 tokens por chamada); **DEVICE_LOST na 31ª** com fala |
| turbo int4 FluidInference | 2025.4.1 | recompilado do zero | **DEVICE_LOST na 1ª** |
| turbo int4 FluidInference | 2026.4.1 | novo | não carrega: `Stateful models without beam_idx input are not supported` |
| turbo int8 `OpenVINO/whisper-large-v3-turbo-int8-ov` | 2025.4.1 | novo | 80/80 |
| turbo int8 `OpenVINO/whisper-large-v3-turbo-int8-ov` | 2026.4.1 | novo | 80/80; 1,5–2,2s por chamada; carga com cache em 2,9s |
| base int8 `OpenVINO/whisper-base-int8-ov` | 2025.4.1 | existente | sem DEVICE_LOST; com 25s de fala: `Check '*roi_end <= *max_dim' failed` |

Hipóteses descartadas:

- **App:** o modo contínuo, o VAD (ONNX Runtime em CPU) e a telemetria (só `nvidia-smi`) não tocam a NPU. As chamadas de inferência já eram serializadas por `_model_lock`, e a falha se reproduz sem o app.
- **Cache corrompido:** um cache recompilado do zero falhou ainda mais cedo.
- **Driver:** não houve atualização de driver nos dias anteriores. Intel AI Boost `32.0.100.5540`.

## Mudança

- `MODEL_REGISTRY["turbo"]` passa a usar `OpenVINO/whisper-large-v3-turbo-int8-ov`, como os outros modelos Whisper, em `local_dir` novo (`whisper-turbo-int8-openvino`). Assim instalações existentes baixam o modelo novo em vez de reaproveitar o int4.
- OpenVINO e OpenVINO GenAI sobem para `>=2026.4,<2027.0`. O `optimum-intel` passa a ter teto `<2.0` explícito, porque o antigo `openvino<2026` era o que o segurava na linha 1.x.

## Pendências

- O Parakeet não foi validado no OpenVINO 2026.4.1: o modelo não estava baixado na máquina de teste. No 2026.4.1 foram validados o turbo int8 (NPU 40/40, iGPU 20/20, CPU 3/3) e o base int8 (NPU 20/20), pelo `create_model` do app.
- O erro `roi_end <= max_dim` na NPU: a saída do decodificador tem tamanho estático, e ditados longos com muito texto estouram esse limite. Isso foi visto no base com 25s de fala. O turbo int8 passou com 26s, mas o limite existe.
- O `~/.npu-dictation/models/whisper-turbo-openvino` (int4) e os blobs dele em `ov-cache` ficam órfãos e podem ser apagados à mão.
