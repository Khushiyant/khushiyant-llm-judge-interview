import dspy
import pathlib
import json
import pandas as pd
import dspy.teleprompt
from llm_judge.database.models.questions import Question
from llm_judge.database.models.answers import Answer
from llm_judge.database.models.judgments import Judgment
from dspy.evaluate import answer_exact_match
import os
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from llm_judge.judges import BaselineComparisonJudge
import logging

llm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
dspy.settings.configure(lm=llm)


class EvaluationSignature(dspy.Signature):
    initial_prompt = dspy.InputField(desc="The initial prompt to evaluate.")

    judgement = dspy.InputField(desc="The judgement to evaluate.")
    question = dspy.InputField(desc="The question to evaluate.")
    answer = dspy.InputField(desc="The answer to evaluate.")

    refined_prompt = dspy.OutputField(desc="assesment prompt to be seen.")


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prgm = dspy.ChainOfThought(EvaluationSignature)

    def forward(self, question):
        return self.prgm(question=question)


class PromptTuner(BaselineComparisonJudge):
    def __init__(
        self,
        ideal_judgement_path,
        baseline_llm,
        judge_prompts,
        judge_llm="gpt-4-turbo-preview",
        judge_llm_params={"temperature": 0},
        num_retries=3,
    ):
        super().__init__(
            baseline_llm, judge_prompts, judge_llm, judge_llm_params, num_retries
        )
        self.cot = CoT()
        self.ideal_judgment_df = pd.read_csv(ideal_judgement_path)

    def refined_simple_prompt(self, initial_prompt):
        optimiser = BootstrapFewShotWithRandomSearch(
            metric=answer_exact_match, max_iters=10, num_samples=10
        )
        trainset = self.ideal_judgment_df
        compiled_cot = optimiser.compile(self.cot, trainset=trainset)

        return compiled_cot(initial_prompt)


if __name__ == "__main__":
    # Load questions and answers
    output_dir = pathlib.Path("output/test-2024-05-22-21-30")

    questions_ids = json.load(open(output_dir / "question_ids.json"))
    answers_ids = json.load(open(output_dir / "answer_ids.json"))

    questions = [Question(id=question_id) for question_id in questions_ids]
    answers = [Answer(id=answer_id) for answer_id in answers_ids]

    tuner = PromptTuner(
        ideal_judgement_path=output_dir / "enriched_judgments",
        baseline_llm="gpt-3.5-turbo",
        judge_prompts={
            "system": "662813e0e25b6076a9e03df8",
        },
    )
    tuner.make_pairs(questions, answers)

    refined_prompt = tuner.refined_simple_prompt("decide which is better.")
    logging.info(refined_prompt)
