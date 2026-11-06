"""Same Portuguese conversations on several LLMs: quality, emoji, offers, speed.

usage: llm_bench.py MODEL_ID [prompt: full|short]"""
import pathlib, random, re, sys, time
import openvino_genai as ov_genai
from debora_whisper import voice_chat as vc

model_id = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else "full"
SHORT = ("Você é a Débora, uma assistente de voz brasileira, simpática e direta. "
         "Suas respostas são faladas por um sintetizador de voz: responda em uma "
         "ou duas frases curtas, só com palavras, sem emoji nem formatação. "
         "O texto do usuário vem de um reconhecedor de voz e pode ter erros.")
system = (vc.VOICE_CHAT_PROMPTS["pt"] if which == "full" else SHORT) + \
    " Agora é sexta-feira, 09/10/2026, 11:05."

SCENES = [
    ("cumprimento", [], "Olá, tudo bem?"),
    ("nome", [("Qual que é o seu nome?", "Meu nome é Débora."),
              ("Meu nome é Alexandre.", "Prazer, Alexandre!")],
     "E... qual que é o meu nome?"),
    ("erro STT", [], "Você sabe que dia que eu agir?"),
    ("data", [], "Débora, que dia é hoje?"),
    ("explicar", [], "Me explica rapidinho o que é um buraco negro."),
    ("conselho", [], "Tô cansado, acho que vou parar um pouco. O que você acha?"),
]

CLEAN = ("Você recebe a transcrição de um reconhecedor de voz em português. "
         "Corrija só erros de reconhecimento, pontuação e termos técnicos mal "
         "transcritos, sem mudar o sentido nem responder ao pedido. Se o texto "
         "já estiver certo, devolva igual. Devolva só o texto corrigido.")
# Real misrecognitions from app.log, plus coding requests a harness would get.
DIRTY = [
    "Você sabe que dia que eu agir?",
    "E o meu nome é Você Sabe?",
    "Tudo ótimo, eu sou o Alexandre.",
    "Débora, abre o ríd mi do projeto e me diz o que falta",
    "cria um commit com a mensagem fiks no bug do áudio",
    "roda os testes do vóis chat e me fala se passou",
    "muda o default do modelo pra turbo no dictêixon engine",
]

path = vc._model_path(model_id, print)
t0 = time.time()
VLM = (pathlib.Path(path) / "openvino_vision_embeddings_model.xml").exists()
pipe = (ov_genai.VLMPipeline if VLM else ov_genai.LLMPipeline)(str(path), "GPU")
print(f"# {model_id} [{which}] {'VLM' if VLM else 'LLM'} loaded in {time.time() - t0:.1f}s")
tok = pipe.get_tokenizer()


def gen(prompt, g, streamer=None):
    if VLM:
        kw = {"streamer": streamer} if streamer else {}
        return pipe.generate(prompt, generation_config=g, **kw)
    return pipe.generate([prompt], g, streamer) if streamer else pipe.generate([prompt], g)

if which == "clean":
    for text in DIRTY:
        msgs = [{"role": "system", "content": CLEAN}, {"role": "user", "content": text}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         extra_context={"enable_thinking": False})
        g = ov_genai.GenerationConfig()
        g.apply_chat_template = False
        g.max_new_tokens = 80
        g.do_sample = False
        start = time.time()
        out = re.sub(r"<think>.*?</think>", "", gen(prompt, g).texts[0], flags=re.DOTALL).strip()
        print(f"{time.time() - start:4.1f}s  {text!r}\n       -> {out!r}")
    sys.exit()

for label, history, question in SCENES:
    msgs = [{"role": "system", "content": system}]
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": question})
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     extra_context={"enable_thinking": False})
    for _ in range(3):
        g = ov_genai.GenerationConfig()
        g.apply_chat_template = False
        g.max_new_tokens = 150
        g.do_sample, g.temperature, g.top_p, g.top_k = True, 0.7, 0.8, 20
        g.repetition_penalty = 1.1
        g.rng_seed = random.randrange(1 << 31)
        first = []
        start = time.time()

        def streamer(chunk):
            if not first:
                first.append(time.time() - start)
            return ov_genai.StreamingStatus.RUNNING

        res = gen(prompt, g, streamer)
        text = res.texts[0]
        n = res.perf_metrics.get_num_generated_tokens()
        took = time.time() - start
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        flags = []
        if vc._EMOJI.search(text):
            flags.append("EMOJI")
        if any(vc.is_help_offer(s) for s in re.split(r"(?<=[?!.])\s+", text)[1:]):
            flags.append("OFFER")
        print(f"{label:11} {first[0] if first else 0:4.1f}s {n / max(took, 1e-6):5.1f} tok/s "
              f"{' '.join(flags):11} {text!r}")
