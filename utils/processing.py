# utils/processing.py

import random

def get_transition_from_gpt(para_a, para_b, examples, client, model="gpt-4", context=None):
    """
    Generate a context-aware French transition (max 5 words)
    using few-shot prompting from the examples list and OpenAI GPT.
    Optionally includes a retrieved context (for Self-RAG).
    """

    # Select 3 random examples for few-shot context
    selected_examples = random.sample(examples, min(3, len(examples)))

    system_prompt = (
        "Tu es un assistant de presse francophone. "
        "Ta tâche est d'insérer une transition brève et naturelle (5 mots maximum) "
        "entre deux paragraphes d'actualité régionale. "
        "La transition doit être journalistique, fluide, neutre et ne pas répéter les débuts comme 'Par ailleurs' ou 'En parallèle'. "
        "La dernière transition de l'article doit être une conclusion claire, choisie parmi : "
        "Enfin, Et pour finir, Pour terminer, Pour finir, En guise de conclusion, En conclusion, En guise de mot de la fin, "
        "Pour clore cette revue, Pour conclure cette sélection, Dernier point à noter, Pour refermer ce tour d’horizon. "
        "Ces expressions doivent apparaître une seule fois, uniquement à la fin."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Incluir ejemplos few-shot
    for ex in selected_examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["transition"]})

    # Si hay contexto adicional desde Self-RAG, inclúyelo
    if context:
        para_a = f"Contexte pertinent :\n{context.strip()}\n\n{para_a.strip()}"

    # Par de párrafos a conectar
    messages.append({
        "role": "user",
        "content": f"{para_a.strip()}\nTRANSITION\n{para_b.strip()}"
    })

    # Llamada al modelo de OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=20
    )

    return response.choices[0].message.content.strip()
