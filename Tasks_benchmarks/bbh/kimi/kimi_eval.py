import pandas as pd
from openai import OpenAI
import re
from dotenv import load_dotenv
import os
import json
import time

# load in the api key and set up the client
# Every path below is resolved against THIS FILE, not against the working directory: the runner
# lives in <bbh>/kimi/ while the 20 task JSONs live in <bbh>/. A copy that resolved
# "boolean_expressions.json" against the cwd is exactly what wrote kimi/kimi_outlog — 20 splits,
# every one "No such file or directory", and an empty results file at the end.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BBH_ROOT = os.path.dirname(MODEL_DIR)

load_dotenv(os.path.join(BBH_ROOT, ".env"))
api_key = os.getenv('KIMI_API_KEY')
client = OpenAI(api_key=api_key,
                timeout=7200, base_url="https://api.moonshot.ai/v1")

# generate the model's response
def get_model_response(question, num_tries, question_num):
    if num_tries > 5:
        print("maxed out tries")
        return

    if num_tries > 0:
        print(f"try #{num_tries} on question #{question_num}")
    
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
            model="kimi-k2.5",  # or another model
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("Error:", e)
        print("num tries:", num_tries)
        print("question number:", question_num)
        print("")
        time.sleep(2^num_tries)
        get_model_response(question, num_tries+1, question_num)

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
splits = [#"date_understanding"
          "dyck_languages"]
          #"formal_fallacies",
          #"multistep_arithmetic_two",
          #"navigate",
          #"object_counting",
          #"penguins_in_a_table",
          #"tracking_shuffled_objects_five_objects",
          #"tracking_shuffled_objects_seven_objects",
          #"tracking_shuffled_objects_three_objects",
          #"web_of_lies",
          #"word_sorting"]

# iterate over each of the splits
for split in splits:
    try: 
        with open(os.path.join(BBH_ROOT, f'{split}.json'), 'r') as file:
            data = json.load(file)

        dataset = data["examples"]
        results = []
        # iterate through the dataset
        counter = 0
        for example in dataset:
            # get the question and gold answer
            q = example["input"]
            gold = example["target"]
            # generate and score the response
            model_resp = get_model_response(q, 1, counter)
            score = score_response(model_resp, gold)

            # append to the results csv for this split
            results.append({
                "question": q,
                "gold_answer": gold,
                "model_response": model_resp,
                "score": score
            })
            counter += 1
            time.sleep(5)

        results_df = pd.DataFrame(results)

        # save the results on this split
        results_df.to_csv(os.path.join(MODEL_DIR, f"kimi_{split}.csv"), index=False)
        # append the average score on this split to the overall results
        overall_results.append({
            "dataset": split,
            "average_score": results_df["score"].mean()
        })
    except Exception as e:
        print("Error:", e)
        print("Happened on split:", split)
        # save the overall results
        overall_df = pd.DataFrame(overall_results)
        overall_df.to_csv(os.path.join(MODEL_DIR, "kimi_overall_results.csv"), index=False)

# save the overall results
overall_df = pd.DataFrame(overall_results)
overall_df.to_csv(os.path.join(MODEL_DIR, "kimi_overall_results.csv"), index=False)
