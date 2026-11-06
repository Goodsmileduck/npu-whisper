# Auditoria de versões do NPU Dictation — 2026-10-05

Consulta somente leitura da instalação usada pelo launcher: `C:/Users/alexandre-machado/.npu-dictation/venv/Scripts/python.exe`. A `.venv` do repositório não foi usada como referência. Nenhum pacote, driver ou configuração foi alterado. `pip check` retornou `No broken requirements found`.

Foram consultados os metadados oficiais do PyPI para **todos os 86 pacotes instalados**. O [inventário JSON completo](whisper-package-inventory-2026-10-05.json) registra versão local, última versão estável não retirada, data de publicação, dependências declaradas e link individual. "Última" nesta auditoria significa o resultado consultado em 2026-10-05; não é garantia de compatibilidade em execução.

## Resultado mais relevante

O driver Intel Arc instalado é **32.0.101.8860**. Há **32.0.101.9033 WHQL**, com notas datadas de 24/09/2026, e **32.0.101.9034 Non-WHQL**, de 01/10/2026. O primeiro é candidato conservador para um ensaio controlado; o segundo é o download mais recente, mas sem certificação WHQL. Ambos incluem o runtime OpenCL usado pelo caminho que falhou. Não foi encontrada uma declaração oficial de correção específica para este erro de Whisper nessa máquina. Fontes: [9033 WHQL](https://downloadmirror.intel.com/929557/ReleaseNotes_101.9033_WHQL.pdf), [9034 Non-WHQL](https://downloadmirror.intel.com/929959/ReleaseNotes_101.9034_Non-WHQL.pdf), [download corrente Intel](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html).

O computador é um **Alienware m16 R2**, Core Ultra 9 185H / Meteor Lake. Convém comparar primeiro o pacote validado para esse modelo no [suporte Dell](https://www.dell.com/support/product-details/en-us/product/alienware-m16-r2-laptop/drivers). A página dinâmica da Dell não forneceu, nesta consulta, versão de pacote suficiente para afirmar qual é a última versão OEM. A disponibilidade do pacote Intel genérico foi confirmada.

## Drivers, gerenciador gráfico e Python

| Componente | Instalado | Disponível consultado | Avaliação |
|---|---|---|---|
| Intel Arc / OpenCL | 32.0.101.8860 | 32.0.101.9033 WHQL; 32.0.101.9034 Non-WHQL | Maior relevância para a falha observada; testar um pacote por vez |
| Intel Graphics Software | Appx 26.32.2604.0 | Installer 26.32.2604.4 nos pacotes 9033/9034 | Reparar a instalação é pertinente pelos erros do serviço; a versão do installer não é diretamente comparável à versão Appx |
| Intel AI Boost / NPU | 32.0.100.5540 | 32.0.100.5540 | Já coincide com a última versão Intel consultada; não é o dispositivo do erro GPU |
| NVIDIA RTX 4070 Laptop | 32.0.16.1714 = 617.14 | 617.14 WHQL, 22/09/2026 | Já coincide com a versão oficial consultada; não participa do backend GPU Intel do app |
| Python | 3.14.7 | 3.14.8, 30/09/2026 | Atualização de manutenção/segurança disponível; sem evidência de corrigir o erro OpenCL |

Fontes adicionais: [Intel NPU](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html), [NVIDIA 617.14](https://www.nvidia.com/de-de/geforce/drivers/details/279818/), [Python 3.14.8](https://www.python.org/downloads/release/python-3148/). A página Python ainda apresenta a linha 3.15 como pré-lançamento; esta auditoria recomenda a atualização de manutenção 3.14.8, não migrar o app para uma linha preview.

A investigação paralela dos logs identificou erros recorrentes do serviço Intel Graphics Software tentando executar `IntelGraphicsSoftware.Service.Graphics.exe` ausente no pacote Appx. Isso sustenta reparar/reinstalar o pacote gráfico oficial, mas não estabelece que esse serviço causou o timeout OpenCL. Não se recomenda copiar executáveis manualmente entre versões.

## Dependências diretas e principais transitivas

Os limites abaixo combinam `requirements.txt` com metadados dos pacotes instalados. Não houve resolução completa nem teste de inferência de uma instalação alternativa.

| Componente | Instalado | Última estável no PyPI | Situação na stack atual |
|---|---|---|---|
| [OpenVINO](https://pypi.org/project/openvino/) | 2025.4.1 | 2026.4.1 | 2025.4.1 já é a última versão permitida por `<2026` |
| [OpenVINO GenAI](https://pypi.org/project/openvino-genai/) | 2025.4.1.0 | 2026.4.1.0 | Mesmo limite; deve acompanhar runtime/tokenizers |
| [OpenVINO Tokenizers](https://pypi.org/project/openvino-tokenizers/) | 2025.4.1.0 | 2026.4.1.0 | Vinculado à mesma linha do runtime |
| [Optimum](https://pypi.org/project/optimum/) | 1.27.0 | 2.3.0 | 1.27.0 já é a última dentro de `<2` |
| [Optimum Intel](https://pypi.org/project/optimum-intel/) | 1.25.2 | 2.2.0 | Versões 1.26/1.27 migram para optimum-onnx, que exige Optimum 2.x; 2.2 exige OpenVINO >=2026 |
| [Transformers](https://pypi.org/project/transformers/) | 4.53.3 | 5.18.0 | 4.53.3 já é a última dentro de `<4.54` |
| [PyTorch](https://pypi.org/project/torch/) | 2.14.1 | 2.14.1 | Atual; usado no export opcional, não na inferência distribuída |
| [NumPy](https://pypi.org/project/numpy/) | 2.3.5 | 2.5.3 | OpenVINO instalado exige `<2.4`; 2.3.5 já é a última compatível com esse teto |
| [Hugging Face Hub](https://pypi.org/project/huggingface-hub/) | 0.36.2 | 2.1.1 | Transformers instalado exige `<1`; 0.36.2 já é a última nessa faixa |
| [Tokenizers HF](https://pypi.org/project/tokenizers/) | 0.21.4 | 0.23.2 | Transformers exige `>=0.21,<0.22`; 0.21.4 já é a última nessa faixa |
| [ONNX Runtime](https://pypi.org/project/onnxruntime/) | 1.30.0 | 1.30.0 | Atual; caminho de pré-processamento Parakeet |
| [ONNX](https://pypi.org/project/onnx/) | 1.23.1 | 1.23.1 | Atual; export |
| [NNCF](https://pypi.org/project/nncf/) | 3.4.0 | 3.4.0 | Atual; conversão/quantização |
| [sounddevice](https://pypi.org/project/sounddevice/) | 0.5.6 | 0.5.6 | Atual; captura de áudio |
| [keyboard](https://pypi.org/project/keyboard/) | 0.13.5 | 0.13.5 | Atual no índice; última publicação é antiga |
| [pyperclip](https://pypi.org/project/pyperclip/) | 1.8.2 | 1.11.0 | Última permitida pelo pin `<1.9` já instalada |
| [pystray](https://pypi.org/project/pystray/) | 0.19.5 | 0.19.5 | Atual |
| [CustomTkinter](https://pypi.org/project/customtkinter/) | 5.2.2 | 6.0.0 | Última permitida pelo pin `<6` já instalada |
| [Pillow](https://pypi.org/project/pillow/) | 11.3.0 | 12.3.0 | Última permitida pelo pin `<12` já instalada |
| [pytest](https://pypi.org/project/pytest/) | 8.4.2 | 9.1.1 | Última permitida pelo pin `<9` já instalada; ferramenta de testes |
| [pip](https://pypi.org/project/pip/) | 26.2.1 | 26.2.1 | Atual |

Os três pacotes OpenVINO 2026.4.1 foram publicados em 01/10/2026 segundo o PyPI. O pacote `openvino-genai` exige `openvino_tokenizers~=2026.4.1.0.dev`, e este exige `openvino~=2026.4.1.dev`. Embora apareça `dev` na expressão de dependência, as versões identificadas na tabela são releases estáveis; isso não autoriza misturar as linhas 2025/2026. A linha 2026.4.1 declara NumPy `<2.6`, mais ampla que a linha atualmente instalada.

## Outras atualizações transitivas encontradas

Estas não integram diretamente a execução do kernel GPU. A coluna de última versão não é uma proposta para instalar todos os pacotes juntos.

| Pacote | Instalado | Última estável |
|---|---|---|
| [aiohttp](https://pypi.org/project/aiohttp/) | 3.14.3 | 3.14.4 |
| [datasets](https://pypi.org/project/datasets/) | 5.0.1 | 5.1.0 |
| [filelock](https://pypi.org/project/filelock/) | 4.0.10 | 4.0.12 |
| [fsspec](https://pypi.org/project/fsspec/) | 2026.6.0 | 2026.9.0 |
| [mpmath](https://pypi.org/project/mpmath/) | 1.3.0 | 1.4.1 |
| [multidict](https://pypi.org/project/multidict/) | 6.9.1 | 7.0.0 |
| [networkx](https://pypi.org/project/networkx/) | 3.6.1 | 3.7 |

Os demais pacotes coincidiram com a última versão estável consultada. Restrições transitivas, como limites impostos por aiohttp, datasets ou sympy, ainda precisam ser resolvidas antes de atualizar essa lista; `pip check` apenas valida as versões atualmente instaladas.

## Limites e ordem de validação sugerida

1. Registrar o estado atual e a ocorrência reproduzível. O pacote gráfico é a primeira variável a testar: comparar a versão OEM da Dell e o Intel 9033 WHQL, incluindo reparo do Graphics Software. Reiniciar após a instalação e usar os mesmos áudios/modelo. Não há garantia de que um desses pacotes resolva a falha.
2. Se persistir, testar o trio OpenVINO **2026.4.1** em um ambiente separado com o mesmo modelo pré-exportado. Preservar o venv atual e validar GPU, CPU e NPU antes de mudar os pins do projeto. O teto `<2026` foi colocado deliberadamente porque a linha nova não estava validada nesse hardware.
3. Atualizar Python 3.14.7 para 3.14.8 em uma etapa separada de manutenção. Testar GUI, gravação, atalhos, bandeja e transcrição. Não combinar essa mudança com o ensaio do driver/runtime, para preservar a comparação.
4. Planejar a modernização da stack de export separadamente. O app baixa modelos OpenVINO já exportados; atualizar torch/Optimum não é uma correção direta para o kernel GPU que falhou.

Há uma incompatibilidade opcional já documentada no repositório: o Optimum 1.27.0 instalado ainda referencia `_attention_scale` em `optimum/exporters/onnx/model_patcher.py`; o arquivo correspondente no Torch 2.14.1 instalado não define essa função. A inspeção foi estática, sem executar export nesta auditoria. Manter `torch>=2.10`, conforme a política de segurança do projeto; não fazer downgrade para contornar o export legado. Migrar Optimum/Optimum Intel demanda testar conjuntamente seus novos requisitos. Além disso, mesmo a versão mais nova de Optimum Intel encontrada (`2.2.0`) declara `transformers<5.6`, portanto instalar simplesmente o último Transformers (`5.18.0`) também não produz uma stack compatível.

A pesquisa paralela encontrou uma [correção upstream de CL_OUT_OF_RESOURCES em convolução](https://github.com/openvinotoolkit/openvino/pull/36127), mas ela trata feature 1/filtro 2048. O IR local do Whisper usa conv1 `[-1,128,3000]` / pesos `[1280,128,3]` e conv2 `[-1,1280,3000]` / pesos `[1280,1280,3]`. Logo, essa correção não foi identificada como correspondência do defeito local. As [notas OpenVINO 2026](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html) justificam investigar a linha nova, não afirmar que ela resolve este incidente.

Não foram feitos testes de estabilidade, export ou instalação alternativa nesta auditoria. O modelo `FluidInference/whisper-large-v3-turbo-int4-ov-npu` foi mantido; substituí-lo seria outra variável experimental.
