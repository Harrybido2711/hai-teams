def classify_query(prompt):
    if "code" in prompt.lower():
        return "code"
    elif len(prompt) > 400:
        return "complex"
    else:
        return "casual"