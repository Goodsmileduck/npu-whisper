### 📝 Resumo do Problema: Latência de Captura de Áudio no Windows (WASAPI / Sounddevice)

**Contexto:**  
Estamos desenvolvendo uma aplicação de ditado push-to-talk em Python, utilizando OpenVINO para inferência rápida do Whisper. A aplicação captura áudio através da biblioteca `sounddevice` (PortAudio/WASAPI). O usuário pressiona uma hotkey, o app toca um pequeno "bip" de confirmação e imediatamente começa a gravar.

**O Problema Central:**  
Sempre que a gravação se inicia, **a primeira sílaba ou palavra do usuário (aproximadamente os primeiros 0.5 a 1.5 segundos) é perdida**. Além disso, o próprio som do "bip" (chime) frequentemente sofre delay para começar a tocar. O problema é restrito ao pipeline de áudio do Windows, a inferência do modelo na NPU/GPU está aquecida e extremamente rápida.

**Diagnóstico e Telemetria (O que descobrimos):**  
Injetamos rastreadores de telemetria (`time.perf_counter()`) no ciclo de início da gravação. Os logs revelaram comportamentos críticos do Windows Audio Engine:
1. **O Delay do Callback (WASAPI Stall):** O comando `sd.InputStream().start()` executa quase que instantaneamente (0.02s), porém, **o primeiro frame de áudio demora entre 0.8s e 1.6s para chegar no callback do Python**.
2. **Conflito Play/Record:** Inicialmente, o "bip" era gerado via `winsound.PlaySound`. O uso dessa API antiga do Windows (`mmsystem`) ativava mecanismos do sistema operacional que congelavam/mutavam o microfone por até 1.6s, possivelmente devido à reconfiguração de grafos de áudio (Device Topology) ou intervenção do Cancelamento de Eco Acústico (AEC).
3. Trocar o `winsound` por `sounddevice.play()` mitigou parte do problema de travamento severo do `winsound`, mas o WASAPI ainda sofre uma negociação custosa (engasgo de ~0.8s) ao tentar iniciar o stream de entrada (microfone) e o de saída (bip) simultaneamente.

**Arquitetura Atual (O que já implementamos como mitigação):**  
Para contornar as limitações do Windows, alteramos o motor de gravação (`AudioRecorder`) para o seguinte padrão de mercado:
1. **Continuous Background Stream:** O `sd.InputStream` do microfone agora abre silenciosamente durante a tela de "Loading" e nunca mais é fechado. O callback roda initerruptamente em background.
2. **Lookback Ring Buffer (1.5s):** A thread do callback armazena continuamente os últimos 1.5 segundos de áudio em uma memória circular (Ring Buffer).
3. **Inversão da Ordem de Inicialização:** Quando a hotkey é pressionada, o app *primeiro* copia instantaneamente os 1.5 segundos do buffer retroativo para "salvar" a respiração e a fala inicial do usuário e, *somente depois*, dispara a thread que toca o bip de início via `sd.play()`. Assim, mesmo que o WASAPI congele a thread de áudio momentaneamente por causa do bip, a fala do usuário daquele instante já estava armazenada no passado.

**Onde Precisamos de Ajuda do Novo Harness:**
- Analisar se a arquitetura atual (*Continuous Stream + Ring Buffer + sd.play()*) é de fato a abordagem mais resiliente para capturas push-to-talk zero-latency no Windows usando Python.
- Avaliar se a abertura paralela de um `sd.OutputStream` (para o bip) está quebrando/corrompendo o fluxo contínuo do `sd.InputStream` do microfone via PortAudio, e sugerir uma alternativa que evite stalls de WASAPI em hardware genérico.
- Identificar se existe alguma configuração na inicialização do stream (ex: desativar AEC via `WASAPI_EXCLUSIVE`, forçar buffers menores em blocksize, ou mudar HostAPI) que torne o fluxo de áudio impenetrável a congelamentos do SO.
