from datasets import load_dataset
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import re
from dotenv import load_dotenv
import os

# load in the api key and set up the client
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# generate the model's response
def get_model_response(question):
    # prompt the model so it's easy to check the answer
    prompt = f"""
    You are a helpful assistant.
    Question: {question}

    Please show your reasoning, then end your response with:
    "Final Answer: <your concise answer here>"
    """

    # create the response
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # or another model
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("Error:", e)
        return None

# get the model's answer
def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    return match.group(1).strip() if match else model_output.strip()

# check if the final answer matches the gold
def score_response(model_response, gold_answer):
    final_answer = extract_final_answer(model_response)
    if final_answer is None:
        return 0
    return int(final_answer.lower().strip() == gold_answer.lower().strip())

# start with an empty list for the overall scores and the list of splits to evaluate
overall_results = []
splits = ["multistep_arithmetic_two"]

# iterate over each of the splits
for split in splits:
    # load the dataset
    dataset = load_dataset("maveriq/bigbenchhard", split, split='train')

    df = dataset.to_pandas()
    results = []
    # iterate through the dataset
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # get the question and gold answer
        q = row["input"]
        gold = row["target"]
        # generate and score the response
        model_resp = get_model_response(q)
        score = score_response(model_resp, gold)

        # append to the results csv for this split
        results.append({
            "question": q,
            "gold_answer": gold,
            "model_response": model_resp,
            "score": score
        })

    results_df = pd.DataFrame(results)

    # save the results on this split
    results_df.to_csv(f"llm_benchmark_results_{split}.csv", index=False)
    # append the average score on this split to the overall results
    overall_results.append({
        "dataset": split,
        "average_score": results_df["score"].mean()
    })

# save the overall results
overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("overall_results.csv", index=False)