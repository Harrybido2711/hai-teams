from models.gpt35 import GPT35
from models.claude import Claude
from models.gemini import Gemini
# Add more imports as needed

def route_query(prompt, classification):
    fallback_chain = [GPT35(), Claude(), Gemini()]
    for model in fallback_chain:
        try:
            response = model.generate(prompt)
            return model.name, response
        except Exception as e:
            print(f"Error from {model.name}: {e}")
            continue
    return "No available model", "All models failed."