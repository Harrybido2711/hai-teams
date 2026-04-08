import re
import unittest


# get the model's answer
def extract_final_answer(model_output):
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    #group(1) means "No" in "Final Answer: No"
    return match.group(1).strip() if match else model_output.strip()

# check if the final answer matches the gold
def score_response(model_response, gold_answer, question=""):
    final_answer = extract_final_answer(model_response)
    if final_answer is None:
        return 0
    # case 1: accurate matching
    if final_answer.lower().strip() == gold_answer.lower().strip():
        return 1
    # case 2: deal with multiple choices (obvious gold_answer has)
    # case 2.1 gold_answer: (D), final_answer: "B" or "(B)" or "(B) choices"
    # case 2.2 gold_answer: (D), final_answer: "choices" with no Alphabet
    if re.match(r'^\([A-Z]\)$', gold_answer.strip()):
        m = re.match(r'^\(?([A-Z])\)?', final_answer.strip())
        if m and f"({m.group(1)})" == gold_answer.strip():
            return 1
     # case 2.2 gold_answer: (D), final_answer: "choices" with no Alphabet
    if question and re.match(r'^\([A-Z]\)$', gold_answer.strip()): # if question mean no empty string
        options = dict(re.findall(r'\(([A-Z])\)\s*([^\n(]+)', question)) #make a dict of the options
        gold_letter = gold_answer.strip("()")       # "(C)" → "C"
        gold_content = options.get(gold_letter, "").strip()
        if gold_content and final_answer.lower() == gold_content.lower():
            return 1
    # case 3: deal with "barn, damp" vs "barn damp"
    if final_answer.lower().replace(",", " ").split() == gold_answer.lower().split():
        return 1
       # case 4: deal with complete sequence of pparenthesis and brackets
    m = re.search(r'Input:\s*(.+)', question, re.IGNORECASE)
    if m:
        partial = m.group(1).strip()         # "[ ["
        full = partial + " " + gold_answer.strip()   # "[ [ ] ]"
        if final_answer.lower().strip() == full.lower().strip():
            return 1
    return 0

class TestScoreResponse(unittest.TestCase):

    def test_exact_match(self):
        self.assertEqual(score_response("Final Answer: No", "No"), 1)

    def test_case_insensitive(self):
        self.assertEqual(score_response("Final Answer: YES", "yes"), 1)

    def test_wrong_answer(self):
        self.assertEqual(score_response("Final Answer: Yes", "No"), 0)

    def test_mc_with_suffix(self):
        self.assertEqual(score_response("Final Answer: (B) heptagon", "(B)"), 1)

    def test_mc_no_parens(self):
        self.assertEqual(score_response("Final Answer: D", "(D)"), 1)

    def test_mc_exact(self):
        self.assertEqual(score_response("Final Answer: (D)", "(D)"), 1)
    
    def test_mc_content_match(self):
        question = """Jane and John married on Jan 2, 1958. What is the date today in MM/DD/YYYY?
    (A) 01/30/2008
    (B) 01/02/2085
    (C) 01/02/2008
    (D) 12/04/2007"""
        self.assertEqual(
            score_response("Final Answer: 01/02/2008", "(C)", question), 1)
    def test_comma_normalization(self):
        self.assertEqual(score_response("Final Answer: barn, damp", "barn damp"), 1)
    def test_dyck_full_sequence(self):
        question = "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: [ ["
        self.assertEqual(
            score_response("Final Answer: [ [ ] ]", "] ]", question), 1
        )
    def test_dyck_full_sequence(self):
        question = "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: < ("
        self.assertEqual(
            score_response("Final Answer: < [ ] >", "] >", question), 1
        )




if __name__ == "__main__":
    unittest.main()
