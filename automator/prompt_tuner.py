import dspy

from typing import List
from llm_judge.database.models.questions import Question
from llm_judge.database.models.answers import Answer
from llm_judge.judges import BaselineComparisonJudge


class PromptTuner(BaselineComparisonJudge):
    def __init__(
        self,
        questions: List[Question],
        answers: List[Answer],
        baseline_llm: str,
        candidate_llm: str,
    ):
        super().__init__(
            baseline_llm=baseline_llm,
            judge_prompts={
                "simple": "Write a clear and concise answer to the question."
            },
            judge_llm=candidate_llm,
        )
        self.questions = questions
        self.answers = answers

    def tune_prompt(self, initial_prompt: str, num_iterations: int):
        # Define the objective function
        def objective_function(prompt: str) -> float:
            self.judge_prompts = {"simple": str(prompt)}
            judgments = self.make_pairs(self.questions, self.answers)
            score = sum([judgment.score[self.candidate_llm] for judgment in judgments])
            return score

        # Use DSPy to optimize the prompt
        optimizer = dspy.optimizers.RandomSearch(
            objective_function, num_iterations=num_iterations
        )
        optimized_prompt = optimizer.optimize()

        # Evaluate the performance of the optimized prompt
        self.judge_prompts = {"simple": str(optimized_prompt)}
        judgments = self.make_pairs(self.questions, self.answers)
        score = sum([judgment.score[self.candidate_llm] for judgment in judgments])
        print(f"Optimized prompt: {optimized_prompt}")
        print(f"Score: {score}")

        return optimized_prompt
