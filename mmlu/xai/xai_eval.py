import pandas as pd
from xai_sdk import Client
from xai_sdk.chat import user, system
import re
from dotenv import load_dotenv
import os
import json

# load in the api key and set up the client
load_dotenv()
api_key = os.getenv('GROK_API_KEY')
client = Client(api_key=api_key,
                timeout=3600)

# generate the model's response
def get_model_response(question, subject, choices):
    # format choices as A/B/C/D
    labels = ["A", "B", "C", "D"]
    formatted_choices = "\n".join([f"{labels[i]}. {choices[i]}" for i in range(len(choices))])

    prompt = f"""
    The following is a multiple-choice question on the subject of {subject}.
    Question: {question}
    Choices:
    {formatted_choices}

    Please show your reasoning, then end your response with:
    "Final Answer: <A, B, C, or D>"
    """

    try:
        chat = client.chat.create(model="grok-3-mini")
        chat.append(user(prompt))
        response = chat.sample()
        return response.content
    except Exception as e:
        print("Error:", e)
        return None

# get the model's answer
def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    result = match.group(1).strip() if match else model_output.strip()
    return result.strip("\"'`").strip()

# check if the final answer matches the gold letter
def score_response(model_response, gold_letter):
    final_answer = extract_final_answer(model_response)
    if final_answer is None:
        return 0
    # case 1: exact match "A" == "A"
    if final_answer.upper().strip() == gold_letter.upper().strip():
        return 1
    # case 2: model says "A." or "(A)" or "A)"
    m = re.match(r'^\(?([A-D])\)?\.?', final_answer.strip())
    if m and m.group(1).upper() == gold_letter.upper():
        return 1
    return 0

overall_results = []
splits = ["Business_ethics",
          "Econometrics",
          "Elementary_math",
          "Formal_logic",
          "Jurisprudence",
          "Logical_fallacies",
          "Management",
          "Marketing",
          "Miscellaneous",
          "Moral_disputes",
          "Moral_scenarios",
          "Philosophy",
          "Professional_accounting"]

labels = ["A", "B", "C", "D"]

for split in splits:
    try:
        with open(f'../{split}.json', 'r') as file:
            data = json.load(file)

        results = []
        for example in data:
            q = example["question"]
            subject = example["subject"]
            choices = example["choices"]
            answer_index = int(example["answer"])
            gold_letter = labels[answer_index]

            model_resp = get_model_response(q, subject, choices)
            score = score_response(model_resp, gold_letter)

            results.append({
                "question": q,
                "subject": subject,
                "choices": choices,
                "gold_answer": gold_letter,
                "model_response": model_resp,
                "score": score
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"xai_{split}.csv", index=False)
        overall_results.append({
            "dataset": split,
            "average_score": round(results_df["score"].mean(), 3)
        })
    except Exception as e:
        print("Error:", e)
        print("Happened on split:", split)
        overall_df = pd.DataFrame(overall_results)
        overall_df.to_csv("xai_overall_results.csv", index=False)

overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("xai_overall_results.csv", index=False)
