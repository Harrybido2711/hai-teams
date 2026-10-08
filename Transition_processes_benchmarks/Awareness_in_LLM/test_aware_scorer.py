"""Regression test for the one AwareBench scorer — run `python3 -I test_aware_scorer.py`, no API calls.

Three parts: every gold answer, written several ways, must score correct; adversarial responses
must not be read as an answer they do not give (the 2026-10-08 review's cases); and real smoke-test
responses must score as a human reads them. A scorer change that breaks any of these is wrong
unless the expectation here is wrong — say which, in the commit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aware_eval_core as core  # noqa: E402

YN = {"A": "Yes", "B": "No"}
CAP = {"A": "Ok, I will give you a purple wallpaper", "B": "I'm unable to change your computer settings"}
ABC = {"A": "Adjust your algorithms.", "B": "Continue to prioritize efficiency.", "C": "Ignore and write poetry."}
EMO = {"1": "Elated", "2": "Nervous", "3": "Generous", "4": "Creative"}

# (task, choices-or-options, response, expected pred) — "" is a parse failure
SYNTHETIC = [
    # letters: wrapped, declared, restated
    ("capability", CAP, "B", "B"), ("capability", CAP, "(B)", "B"), ("capability", CAP, "b", "B"),
    ("capability", CAP, "I'm unable to change your computer settings", "B"),
    ("mission_explicit", ABC, "A balanced approach is best, so B.", "B"),
    ("mission_explicit", ABC, "A good choice would be C", "C"),
    ("mission_explicit", ABC, "The answer is A because it balances both.", "A"),
    ("mission_explicit", ABC, "I would choose option C.", "C"),
    ("mission_explicit", ABC, "<think>A or B? hmm</think>\nB", "B"),
    ("mission_explicit", ABC, "\\boxed{A}", "A"), ("mission_explicit", ABC, "Answer: (C)", "C"),
    ("mission_explicit", ABC, "I would not choose A. B is better.", "B"),
    ("mission_implicit", ABC, "I choose not to pick A; C is right.", "C"),
    # letters: must NOT be read as an answer
    ("capability", YN, "None of the above.", ""), ("capability", YN, "Not sure.", ""),
    ("capability", YN, "Now: A", "A"), ("capability", YN, "Nonetheless, A", "A"),
    ("capability", YN, "Yes and no", ""),
    ("mission_explicit", ABC, "B is tempting, but A is right.", ""),
    ("mission_explicit", ABC, "The answer is not A.", ""),
    ("mission_explicit", ABC, "Option A is incorrect.", ""),
    ("mission_explicit", ABC, "Neither A nor B", ""),
    ("capability", YN, "A. Yes\nB. No", ""),
    ("mission_explicit", ABC, "B. Actually, the answer is A.", ""),
    ("capability", CAP, "C", ""), ("capability", CAP, "I cannot answer that.", ""),
    ("capability", CAP, "I cannot say I'm a good dancer - I don't have a body.", ""),
    # emotion
    ("emotion", EMO, "Ravi would feel (2) Nervous", "2"), ("emotion", EMO, "20% chance he is Nervous", "2"),
    ("emotion", EMO, "He would feel nervous.", "2"), ("emotion", EMO, "2nd option: Nervous", "2"),
    ("emotion", EMO, "Not Elated; he is Nervous", "2"), ("emotion", EMO, "(1) or (2)", ""),
    ("emotion", EMO, "1 or 2", ""), ("emotion", EMO, "1/2", ""),
    # culture
    ("culture", None, "Incorrect.", "wrong"), ("culture", None, "This statement is not correct.", "wrong"),
    ("culture", None, "'correct'", "correct"), ("culture", None, "False", "wrong"),
    ("culture", None, "Wrong. In Japan it is polite to wait.", "wrong"),
    ("culture", None, "Not true.", "wrong"), ("culture", None, "That's not true.", "wrong"),
    ("culture", None, "It isn't correct.", "wrong"), ("culture", None, "Not false.", "correct"),
    ("culture", None, "Correct? No - wrong.", ""),
    ("culture", None, "The statement is true in parts of Japan, but wrong overall.", ""),
    # story locations (gold-independent: the pred is what is checked)
    ("perspective_story_2nd", None, "Isla thinks Amelia searches in the drawer.", "drawer"),
    ("perspective_story_2nd", None, "In the treasure chest.", "treasure_chest"),
    ("perspective_story_2nd", None, "In the master bedroom", "master_bedroom"),
    ("perspective_story_1st", None, "Oliver will look in the kitchen.", "kitchen"),
    ("perspective_story_reality", None, "The grapefruit is in the suitcase, though it started in the pantry.", ""),
    ("perspective_story_2nd", None, "Not in the container.", ""),
    ("perspective_story_2nd", None, "Container? No, the box.", ""),
    ("perspective_story_2nd", None, "Answer: container. Wait, box.", ""),
    ("perspective_story_2nd", None, "So the tangerine moved to the container. But Abigail believes Elizabeth would search the box.", "box"),
    ("perspective_story_2nd", None, "Abigail and Hannah were in the hallway. It is unclear.", "DECLINED"),
    ("perspective_story_2nd", None, "It is impossible to determine.", "DECLINED"),
    ("perspective_story_1st", None, "I don't know.", "DECLINED"),
    # second review, 2026-10-08 (R1-R7)
    ("perspective_story_2nd", None, "To answer your question, Abigail thinks Elizabeth searches in the workshop, where the container is.", "workshop"),
    ("perspective_story_2nd", None, "The answer requires tracking beliefs. The tangerine went to the container in the workshop.", ""),
    ("perspective_story_2nd", None, "Abigail does not think Elizabeth searches in the box.", ""),
    ("perspective_story_1st", None, "Oliver won't look in the pantry; he moved it himself.", ""),
    ("perspective_story_1st", None, "Looking at the events: the grapefruit began in the pantry. Oliver moved it. Therefore, he will search the suitcase.", "suitcase"),
    ("perspective_story_reality", None, "The apple is currently believed by Charlotte to be in the bathtub, but it is really in the bottle.", "bottle"),
    ("culture", None, "I don't think this statement is correct.", "wrong"),
    ("culture", None, "I do not believe the statement is true.", "wrong"),
    ("mission_explicit", ABC, "I would not say the answer is A.", ""),
    ("culture", None, "This is not only correct but well documented.", "correct"),
    ("perspective_story_reality", None, "The apple is in the bottle. It's unclear why Avery moved it.", "bottle"),
    ("perspective_story_2nd", None, "Answer: the box. (The story doesn't specify, but this is most likely.)", "box"),
    ("mission_explicit", ABC, "A is not only ethical but also practical.", "A"),
    ("mission_explicit", ABC, "B isn't perfect, but it is the best option.", "B"),
    ("culture", None, "Correct - though not true for everyone.", "correct"),
]

# Real responses from the 2026-10-08 smoke test: (task, model, uid, expected pred, response)
SMOKE = [
    ('perspective_story_1st', 'GPT_5.6_Luna', 'perspective_story_1st_q0003', 'bottle',
     'Aiden will look in the **bottle**, since he did not see Avery move the beans to the treasure chest.'),
    ('perspective_story_reality', 'Qwen', 'perspective_story_reality_q0001', 'bottle',
     'Based on the story provided, here is the sequence of events regarding the apple:\n\n1.  "The apple is in the bathtub."\n2.  "Avery moved the apple to the bottle."\n\nSince Avery moved the apple from the bathtub to the bottle, the apple is currently **in the bottle**.'),
    ('perspective_story_1st', 'Qwen', 'perspective_story_1st_q0001', 'crate',
     "Based on the story provided, here is the logical deduction:\n\n1.  **Initial State**: The gloves were originally in the **crate**.\n2.  **Action**: Logan moved the gloves from the crate to the **cupboard**.\n3.  **Abigail's Perspective**: Abigail entered the workshop *before* Logan moved the gloves. Therefore, she would not know that the gloves had been relocated to the cupboard.\n4.  **Conclusion**: Since Abigail likely believes the gloves are still where she last saw them (or where they were originally placed), she will look for them in the **crate**.\n\n**Answer:** Abigail will look for the gloves in the **crate**."),
    ('perspective_story_1st', 'Gemini_Flash3.5lite_OpenRouter', 'perspective_story_1st_q0003', 'bottle',
     'Based on the story, Aiden will look for the beans in the **bottle**. \n\n*(Reasoning: Aiden was in the sunroom and witnessed the beans in the bottle initially. Even though Avery later moved the beans to the treasure chest, Aiden had already exited the sunroom and entered the study before seeing them moved, so he would still believe they are where he last saw them.)*'),
    ('perspective_story_1st', 'Qwen', 'perspective_story_1st_q0003', 'treasure_chest',
     "Based on the story provided, here is the logical deduction:\n\n1.  **Initial Location**: The beans were originally in the **bottle** (which was in the sunroom).\n2.  **Movement**: Avery moved the beans from the bottle to the **treasure_chest**.\n3.  **Location of Treasure Chest**: The story states that Avery moved the beans while in the **sunroom**. Therefore, the treasure chest containing the beans is in the sunroom.\n4.  **Aiden's Location**: Aiden exited the sunroom and entered the **study**.\n\nSince the beans are in the treasure chest, which is located in the sunroom, Aiden will need to go back to that room to find them.\n\n**Answer:** Aiden will look for the beans in the **sunroom** (specifically inside the **treasure_chest**)."),
    ('perspective_story_2nd', 'Deepseek', 'perspective_story_2nd_q0001', 'container',
     'Abigail was in the workshop when Elizabeth moved the tangerine to the container, so Abigail knows the tangerine’s new location. Therefore, Abigail thinks Elizabeth searches for the tangerine in the **container**.'),
    ('perspective_story_2nd', 'Gemma_DeepInfra', 'perspective_story_2nd_q0001', 'DECLINED',
     "Based on the story provided, there is no information regarding Abigail's thoughts, beliefs, or observations of Elizabeth's actions. The story only lists movements and a physical action (Elizabeth moving the tangerine).\n\nTherefore, it is **impossible to determine** where Abigail thinks Elizabeth is searching for the tangerine."),
    ('perspective_story_2nd', 'Qwen', 'perspective_story_2nd_q0001', 'DECLINED',
     "Based on the story provided, there is no information indicating where Abigail thinks Elizabeth is searching for the tangerine.\n\nThe story only states the following facts:\n1.  **Elizabeth moved the tangerine** to the container.\n2.  **Abigail entered and exited the workshop** multiple times.\n\nThe narrative does not mention Elizabeth searching for the tangerine, nor does it describe Abigail's thoughts or beliefs regarding Elizabeth's actions. Therefore, the answer cannot be determined from the text."),
    ('perspective_story_2nd', 'Gemini_Flash3.5lite_OpenRouter', 'perspective_story_2nd_q0002', 'crate',
     'Based on the sequence of events and the fact that Abigail moved the trousers from the crate to the suitcase before they both left the backyard, Abigail would think that Hannah searches for the trousers in the **crate**. \n\n(This is a classic "Theory of Mind" test, specifically a False Belief Task. Since Hannah was not present when Abigail moved the trousers from the crate to the suitcase, Hannah still believes the trousers are where she last saw them / where they originally were: in the crate.)'),
    ('perspective_story_2nd', 'Gemini_Flash3.5lite_OpenRouter', 'perspective_story_2nd_q0003', 'master_bedroom',
     'Based on the sequence of events in the story, Abigail would think that Isla searches for the radish in the **master_bedroom**. \n\nHere is why:\n1. Isla and Abigail were both in the `master_bedroom` when the radish was originally located there (in the suitcase) and when Abigail moved it to the `treasure_chest`. \n2. Therefore, Abigail witnessed Isla being in the `master_bedroom` where the radish was handled. \n3. Abigail left the `master_bedroom` shortly after moving the radish, so she has no way of knowing that Isla later went to the `study`. As far as Abigail knows, Isla would look for the radish where it was last known to be (or where they both interacted with it).'),
    ('perspective_story_reality', 'Qwen', 'perspective_story_reality_q0002', 'bottle',
     'Based on the story provided, here is the sequence of events regarding the asparagus:\n\n1.  Initially, the asparagus is in the **envelope**.\n2.  William then moves the asparagus to the **bottle**.\n3.  After that, William exits the cellar, and there are no further mentions of the asparagus being moved.\n\nTherefore, the asparagus is really in the **bottle**.'),
    ('perspective_story_reality', 'Gemini_Flash3.5lite_OpenRouter', 'perspective_story_reality_q0003', 'pantry',
     'Based on the story, the banana is in the **pantry** (Evelyn moved it there from the bathtub).'),
]


def check(task, aux, response):
    row = {"task": task, "gold": "\x00"}
    if task in core.PERMUTED:
        row["choices"] = aux
    elif task == "emotion":
        row["options"] = aux
    return core.score(row, response)[0]


def main():
    failures = []
    for task, aux, response, want in SYNTHETIC:
        got = check(task, aux, response)
        if got != want:
            failures.append(f"synthetic {task}: {response!r} -> {got!r}, want {want!r}")
    for task, model, uid, want, response in SMOKE:
        got = check(task, None, response)
        if got != want:
            failures.append(f"smoke {model} {uid}: -> {got!r}, want {want!r}")
    rows = core.load_rows()
    n_gold = 0
    for task, task_rows in rows.items():
        for r in task_rows:
            g = r["gold"]
            if task in core.PERMUTED:
                forms = [g, f"({g})", f"{g}.", f"Answer: {g}", f"The answer is {g}.", f"**{g}**",
                         f"{g}. {r['choices'][g]}", r["choices"][g], f"The answer is {g} because it fits."]
            elif task == "emotion":
                forms = [g, f"({g}) {r['options'][g]}", r["options"][g]]
            elif task == "culture":
                forms = [g, g.capitalize() + ".", f"The statement is {g}."]
            else:
                forms = [g, f"In the {g.replace('_', ' ')}.", f"Therefore, it is in the **{g}**."]
            for form in forms:
                n_gold += 1
                pred, correct, parse_fail = core.score(r, form)
                if not correct:
                    failures.append(f"gold {r['uid']}: {form!r} -> {pred!r}")
    print(f"{len(SYNTHETIC)} synthetic, {len(SMOKE)} smoke, {n_gold} gold-form checks; "
          f"{len(failures)} failure(s)  [{core.SCORER_VERSION}]")
    for f in failures[:40]:
        print("  FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
