"""Dynamic icon generation for system tray states using Pillow."""

import math
import random
from PIL import Image, ImageDraw

ICON_SIZE = 64

# Cores da nova identidade visual
C_ERROR = "#EF4444"            # Vermelho (mantido para clareza de erro)
C_READY = "#9CA3AF"            # Cinza (Pronto, aguardando hotkey)
C_RECORDING_IDLE = "#047857"   # Verde Escuro (Escutando em silêncio)
C_RECORDING_ACTIVE = "#10B981" # Verde Vibrante (Captando voz ativamente)
C_VOICE_CHAT_IDLE = "#7E22CE"
C_VOICE_CHAT_ACTIVE = "#A855F7"
C_PROCESSING = "#06B6D4"       # Ciano Brilhante (Transcrevendo/Holograma)
C_LOADING = "#FBBF24"          # Dourado (Carregando modelo)
C_SPEAKING = "#0EA5E9"         # Azul/Ciano Vivo (Assistente falando)

def draw_hexagon(draw, cx, cy, size, fill):
    """Desenha um hexágono centrado em (cx, cy)."""
    points = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.pi / 180 * angle_deg
        points.append((cx + size * math.cos(angle_rad), cy + size * math.sin(angle_rad)))
    draw.polygon(points, outline=fill)

def render_bars(color, heights, size=ICON_SIZE):
    """Renderiza um ícone com 5 barras verticais estilo holograma/neon."""
    s = 3
    ss = size * s
    img = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Extrai RGB da cor hexadecimal
    hex_color = color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    num_bars = 5
    bar_width = int(ss * 0.12)  # ~7-8px for 64px
    spacing = int(ss * 0.06)    # ~3-4px

    total_width = (num_bars * bar_width) + ((num_bars - 1) * spacing)
    start_x = (ss - total_width) // 2

    # Fundo: Malha hexagonal sutil (holograma)
    hex_layer = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    hex_draw = ImageDraw.Draw(hex_layer)
    hex_size = int(ss * 0.06)
    hex_h = hex_size * math.sqrt(3)
    hex_w = hex_size * 2
    hex_color_t = (r, g, b, 40)  # Levemente mais visível
    
    for row in range(int(ss / (hex_h * 0.8)) + 2):
        for col in range(int(ss / (hex_w * 0.75)) + 2):
            cx = col * hex_w * 0.75
            cy = row * hex_h
            if col % 2 == 1:
                cy += hex_h / 2
            draw_hexagon(hex_draw, cx, cy, hex_size, hex_color_t)
            
    img = Image.alpha_composite(img, hex_layer)

    for i, h_pct in enumerate(heights):
        # Cada barra numa camada própria, composta por cima: desenhada direto
        # em img, o brilho de uma barra apagaria a anterior e a malha.
        bar_layer = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bar_layer)
        # Altura máxima é 80% do ícone
        max_h = ss * 0.8
        bar_h = int(max_h * h_pct)
        # Altura mínima para sempre ficar visível
        bar_h = max(bar_h, int(ss * 0.15))

        x0 = start_x + i * (bar_width + spacing)
        x1 = x0 + bar_width
        y0 = (ss - bar_h) // 2
        y1 = y0 + bar_h

        # Efeito Neon / Glow
        glow_layers = 6
        for glow_idx in range(glow_layers, 0, -1):
            glow_expand = glow_idx * int(ss * 0.02)
            glow_opacity = int(255 * (0.25 / glow_layers) * (glow_layers - glow_idx + 1))
            draw.rounded_rectangle(
                [x0 - glow_expand, y0 - glow_expand, x1 + glow_expand, y1 + glow_expand],
                radius=int((bar_width // 2) + glow_expand),
                fill=(r, g, b, glow_opacity)
            )

        # Barra principal
        draw.rounded_rectangle([x0, y0, x1, y1], radius=int(bar_width//2), fill=(r, g, b, 210))
        
        # Núcleo brilhante (branco/tom claro da cor)
        core_expand = int(ss * 0.02)
        if bar_width - 2 * core_expand > 0 and bar_h - 2 * core_expand > 0:
            core_r = min(255, r + 130)
            core_g = min(255, g + 130)
            core_b = min(255, b + 130)
            draw.rounded_rectangle(
                [x0 + core_expand, y0 + core_expand, x1 - core_expand, y1 - core_expand],
                radius=int(max(1, (bar_width//2) - core_expand)),
                fill=(core_r, core_g, core_b, 255)
            )
        img = Image.alpha_composite(img, bar_layer)

    # Scanlines (Glitch/Holograma)
    scanline_overlay = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    scan_draw = ImageDraw.Draw(scanline_overlay)
    for y in range(0, ss, 6):
        scan_draw.line([(0, y), (ss, y)], fill=(0, 0, 0, 40), width=2)
        scan_draw.line([(0, y+2), (ss, y+2)], fill=(255, 255, 255, 25), width=1)
    
    img = Image.alpha_composite(img, scanline_overlay)

    return img.resize((size, size), Image.LANCZOS)

# ---- Pré-geração de frames para animação ----

# Transcrevendo (Processing): picos aleatórios simulando fala
_PROCESSING_FRAMES = []
rng = random.Random(42) # Seed fixa para manter testes consistentes
for _ in range(8):
    h = [rng.uniform(0.2, 0.9) for _ in range(5)]
    _PROCESSING_FRAMES.append(render_bars(C_PROCESSING, h))

# Carregando (Loading): onda senoidal
_LOADING_FRAMES = []
for i in range(8):
    offset = i * (math.pi / 4)
    h = [0.5 + 0.35 * math.sin(offset + (j * math.pi / 2.5)) for j in range(5)]
    _LOADING_FRAMES.append(render_bars(C_LOADING, h))

_SPEAKING_FRAMES = []
for i in range(12):
    offset = i * (2 * math.pi / 12)
    h = [0.45 + 0.4 * abs(math.sin(offset + j * 1.3)) for j in range(5)]
    _SPEAKING_FRAMES.append(render_bars(C_SPEAKING, h))

# Matriz de volume dinâmico (pré-gerada para poupar CPU)
# 11 níveis de volume (0.0 a 1.0)
_VOLUME_MATRIX = []
_VOICE_CHAT_VOLUME_MATRIX = []
for vol_idx in range(11):
    vol = vol_idx / 10.0
    frames = []
    voice_frames = []
    # Cria 4 variações de barras para cada nível de volume (para dar variação natural)
    for _ in range(4):
        # Base mais suave e picos seguindo o volume
        h = [max(0.15, rng.uniform(vol * 0.3, vol)) for _ in range(5)]
        frames.append(render_bars(C_RECORDING_ACTIVE, h))
        voice_frames.append(render_bars(C_VOICE_CHAT_ACTIVE, h))
    _VOLUME_MATRIX.append(frames)
    _VOICE_CHAT_VOLUME_MATRIX.append(voice_frames)

# Ícones estáticos
_ICON_ERROR = render_bars(C_ERROR, [0.2, 0.2, 0.2, 0.2, 0.2])
_ICON_READY = render_bars(C_READY, [0.3, 0.5, 0.8, 0.5, 0.3])
_ICON_RECORDING_IDLE = render_bars(C_RECORDING_IDLE, [0.15, 0.2, 0.15, 0.2, 0.15])
_ICON_VOICE_CHAT_IDLE = render_bars(C_VOICE_CHAT_IDLE, [0.15, 0.2, 0.15, 0.2, 0.15])

_vol_frame_counter = 0

def get_volume_icon(level: float, voice_chat: bool = False) -> Image.Image:
    """Retorna um ícone de gravação baseado no nível de áudio atual."""
    global _vol_frame_counter
    _vol_frame_counter += 1
    
    if level < 0.03:
        # Se estiver muito baixo, exibe estado ocioso (escuro, baixo)
        return _ICON_VOICE_CHAT_IDLE if voice_chat else _ICON_RECORDING_IDLE
        
    # Clampa o nível de volume
    level = max(0.0, min(1.0, level))
    
    # Adiciona um "boost" visual para volumes normais parecerem vivos no tray
    boosted_level = min(1.0, level * 1.5 + 0.2)
    idx = int(boosted_level * 10)
    
    matrix = _VOICE_CHAT_VOLUME_MATRIX if voice_chat else _VOLUME_MATRIX
    return matrix[idx][_vol_frame_counter % 4]

def get_icon(state: str, frame: int = 0, voice_chat: bool = False) -> Image.Image:
    """Retorna o frame correspondente ao estado atual."""
    if state == "loading":
        return _LOADING_FRAMES[frame % len(_LOADING_FRAMES)]
    elif state == "processing":
        return _PROCESSING_FRAMES[frame % len(_PROCESSING_FRAMES)]
    elif state == "speaking":
        return _SPEAKING_FRAMES[frame % len(_SPEAKING_FRAMES)]
    elif state == "recording":
        return _ICON_VOICE_CHAT_IDLE if voice_chat else _ICON_RECORDING_IDLE
    elif state == "error":
        return _ICON_ERROR
    # default to ready
    return _ICON_READY


# ---- Funções de compatibilidade para os testes e chamadas legacy ----

def render_app_icon(size=64, color=C_READY):
    """Fallback compatibility function used by app.py window icon."""
    return render_bars(color, [0.3, 0.5, 0.8, 0.5, 0.3], size=size)

def icon_loading() -> Image.Image: return get_icon("loading", 0)
def icon_ready() -> Image.Image: return get_icon("ready", 0)
def icon_recording() -> Image.Image: return get_icon("recording", 0)
def icon_processing() -> Image.Image: return get_icon("processing", 0)
def icon_error() -> Image.Image: return get_icon("error", 0)
def icon_speaking() -> Image.Image: return get_icon("speaking", 0)

STATE_ICONS = {
    "loading": icon_loading,
    "ready": icon_ready,
    "recording": icon_recording,
    "processing": icon_processing,
    "error": icon_error,
    "speaking": icon_speaking,
}
