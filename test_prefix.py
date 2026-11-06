import os
import re

def simulate(drafts):
    screen = ""
    typed_text = ""
    for text, is_final in drafts:
        original = text
        stripped = text.strip()
        if stripped and not re.search(r'[.,!?;\:]$', stripped):
            text = stripped + '... '
        elif not text.endswith(' '):
            text += ' '

        text_to_type = text
        if typed_text:
            common = os.path.commonprefix([typed_text, text])
            dels = len(typed_text) - len(common)
            screen = screen[:-dels] if dels > 0 else screen
            text_to_type = text[len(common):]
        else:
            dels = 0
            
        screen += text_to_type
        print(f"[{is_final}] In: {repr(original):<20} | dels: {dels} | type: {repr(text_to_type):<15} | Screen: {repr(screen)}")
        
        if is_final:
            typed_text = ""
        else:
            typed_text = text

print("Scenario 1: Standard")
simulate([
    ("Olá", False),
    ("Olá,", False),
    ("Olá, tudo bem?", True)
])

print("\nScenario 2: Cut segment")
simulate([
    ("Olá", False),
    ("Olá,", True),
    ("Olá. Tudo", False),
    ("Olá. Tudo bem?", True)
])
