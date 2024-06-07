import os
import dspy
import pathlib

import pandas as pd
import dspy.teleprompt
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from llm_judge.judges import BaselineComparisonJudge
from dotenv import load_dotenv

from utils.metric import overall_metric
from utils.mongo_extract import MongoData
from utils.refiner import Prompter


llm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
dspy.settings.configure(lm=llm)


load_dotenv("./.env", override=True)


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

        self.ideal_judgment_df = pd.read_csv(ideal_judgement_path)

    def refined_prompt(self, initial_prompt, judgement, answer, reference_answer):
        cot = Prompter()
        cot.load("compiled_cot.json")

        return cot(
            initial_prompt=initial_prompt,
            reference_answer=reference_answer,
            judgement=judgement,
            answer=answer,
        )

    def train_data(self, initial_prompt):
        optimiser = BootstrapFewShotWithRandomSearch(
            metric=overall_metric, max_bootstrapped_demos=1
        )
        cot = Prompter()

        dataset = [
            dspy.Example(
                judgement=judgement,
                reference_answer=self.ideal_judgment_df["reference_answer"][i],
                answer=self.ideal_judgment_df["answer"][i],
                initial_prompt=initial_prompt,
            ).with_inputs("judgement", "reference_answer", "answer", "initial_prompt")
            for i, judgement in enumerate(self.ideal_judgment_df["judgments"])
        ]

        trainset = dataset[:-10]
        valset = dataset[-10:]
        compiled_cot = optimiser.compile(cot, trainset=trainset, valset=valset)
        compiled_cot.save("compiled_cot.json")


if __name__ == "__main__":
    # Load questions and answers
    output_dir = pathlib.Path("output/test-2024-06-06-23-42")
    tuner = PromptTuner(
        ideal_judgement_path=output_dir / "enriched_judgments",
        baseline_llm="gpt-3.5-turbo",
        judge_prompts={
            "system": "662813e0e25b6076a9e03df8",
            "prompt_template": "662821e23eb9ef01018e30e2",
            "reversed_prompt_template": "6628224eb84c0693351ca6a4",
        },
    )
    judgement = """
    ['1. **Does one answer follow all user instructions and the other one fails to do so?**\n   - Both the reference and candidate answers follow the user\'s instruction by selecting one of the given options as the treatment method. There are no additional instructions to follow, such as providing detailed explanations or summaries.\n\n2. **Does one answer respond to the user question and the other one mis-interpret the user question?**\n   - Both answers correctly interpret the user\'s question, which asks for the appropriate treatment method for stings from a bluebottle, stonefish, or sea urchin.\n\n3. **Is one answer less reasonable than the other given the context of the conversation?**\n   - Both answers provide the same response, "Hot water," which is reasonable and appropriate given the context of the question. There is no difference in the reasonableness of the answers.\n\n4. **If the question has an objectively correct answer and the candidate and reference answers have different results, one must be better than the other.**\n   - In this case, both the candidate and reference answers provide the same result, "Hot water," which is the correct treatment for stings from a bluebottle, stonefish, or sea urchin according to the options provided.\n\nGiven the analysis, both answers are equivalent in quality, correctly following the user\'s instructions, accurately interpreting the question, providing a reasonable response, and aligning with the correct answer.\n\nFinal Verdict: "[[C]]"', '1. Both the candidate and reference answers follow the user\'s instruction by selecting one of the given options without providing unnecessary explanations, which aligns with the user\'s request for a direct answer.\n2. Both answers correctly respond to the user\'s question without misinterpreting it.\n3. Neither answer is less reasonable than the other; both correctly identify "Hot water" as the treatment for stings from a bluebottle, stonefish, or sea urchin.\n4. The question has an objectively correct answer, and both the candidate and reference answers provide the same correct response.\n\nFinal Verdict: "[[C]]"']"""
    answer = "Hot water"

    tuner.train_data(initial_prompt=MongoData.get_prompt("662821e23eb9ef01018e30e2"))

    refined_prompt = tuner.refined_prompt(
        initial_prompt="decide which is better answer?",
        judgement=judgement,
        answer=answer,
        reference_answer=answer,
    )
    print("Refined Prompt: ", refined_prompt)
