import json
import os

# Paths
questions_file = os.path.expanduser("~/Desktop/6344334/all_questions.json")
answers_file   = os.path.expanduser("~/Desktop/6344334/all_answers.json")

# Load original datasets
with open(questions_file) as f:
    all_questions = json.load(f)["questions"]
with open(answers_file) as f:
    all_answers = json.load(f)["answers"]

answer_lookup = {a["id"]: a for a in all_answers}
all_questions = all_questions[:500]
# ---- Filters ----
def is_yesno(qtext: str) -> bool:
    qtext = qtext.lower().strip()
    return qtext.startswith(("is", "are", "does", "do", "was"))

def is_equality(qtext: str) -> bool:
    return "equal to" in qtext.lower()

# ---- Subsets ----
yesno_questions = [q for q in all_questions if is_yesno(q["question"])]
equality_questions = [q for q in all_questions if is_equality(q["question"])]

# Collect referenced answers for each subset
def collect_answers(questions):
    subset_answer_ids = {aid for q in questions for aid in q.get("answers_ids", [])}
    return [answer_lookup[aid] for aid in subset_answer_ids if aid in answer_lookup]

yesno_answers = collect_answers(yesno_questions)
equality_answers = collect_answers(equality_questions)

# ---- Save JSONs ----
out_dir = os.path.expanduser("~/Desktop")

with open(os.path.join(out_dir, "yesno_questions.json"), "w") as f:
    json.dump({"questions": yesno_questions}, f, indent=2)

with open(os.path.join(out_dir, "yesno_answers.json"), "w") as f:
    json.dump({"answers": yesno_answers}, f, indent=2)

with open(os.path.join(out_dir, "equality_questions.json"), "w") as f:
    json.dump({"questions": equality_questions}, f, indent=2)

with open(os.path.join(out_dir, "equality_answers.json"), "w") as f:
    json.dump({"answers": equality_answers}, f, indent=2)

print(f"✅ Wrote {len(yesno_questions)} Yes/No questions "
      f"and {len(equality_questions)} Equality questions to {out_dir}")

