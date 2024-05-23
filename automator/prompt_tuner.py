import dspy
import pathlib
import bson
import json
import pandas as pd
import dspy.teleprompt
from dspy.evaluate import answer_exact_match
import os
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from llm_judge.judges import BaselineComparisonJudge

llm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
dspy.settings.configure(lm=llm)

import os

import pymongo
import pickle
from dotenv import load_dotenv
import logging

load_dotenv("./.env", override=True)


class EvaluationSignature(dspy.Signature):
    initial_prompt = dspy.InputField(desc="The initial prompt to evaluate.")

    judgement = dspy.InputField(desc="The judgement to evaluate.")
    question = dspy.InputField(desc="The question to evaluate.")

    ideal_prompt = dspy.OutputField(desc="assesment prompt to be seen.")


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prgm = dspy.ChainOfThought(EvaluationSignature)

    def forward(self, initial_prompt, judgement, question, ideal_prompt):
        return self.prgm(
            initial_prompt=initial_prompt,
            judgement=judgement,
            question=question,
            ideal_prompt=ideal_prompt,
        )


class PromptTuner(BaselineComparisonJudge):
    """
    A class that represents a prompt tuner for baseline comparison judging.

    Parameters:
    - ideal_judgement_path (str): The file path to the ideal judgement CSV file.
    - baseline_llm (str): The baseline language model to use for comparison.
    - judge_prompts (list): A list of judge prompts.
    - judge_llm (str): The language model to use for judging.
    - judge_llm_params (dict): Additional parameters for the judge language model.
    - num_retries (int): The number of retries for judging.

    Attributes:
    - cot (CoT): An instance of the CoT class.
    - ideal_judgment_df (pd.DataFrame): A DataFrame containing the ideal judgement data.

    Methods:
    - refined_simple_prompt(initial_prompt): Refines the initial prompt using an optimiser.

    """

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

    def get_prompt(self, prompt_id):
        client = pymongo.MongoClient(os.environ["MARTIAN_MONGO_URI"])

        prompts = client["llm-judge"]["prompts"]
        prompt = prompts.find_one({"_id": bson.ObjectId(prompt_id)})
        return prompt["content"]

    def refined_prompt(self, ideal_prompt, initial_prompt, question, judgement):
        cot = CoT()
        cot.load("compiled_cot.json")

        return cot(
            initial_prompt=initial_prompt,
            question=question,
            judgement=judgement,
            ideal_prompt=ideal_prompt,
        )

    def train_data(self, ideal_prompt):
        """
        Refines the initial prompt using an optimiser.

        Parameters:
        - initial_prompt (str): The initial prompt to be refined.

        Returns:
        - str: The refined prompt.

        """
        optimiser = BootstrapFewShotWithRandomSearch(
            metric=answer_exact_match, max_iters=5, num_samples=5
        )
        cot = CoT()

        trainset = [
            dspy.Example(
                initial_prompt=self.get_prompt(
                    self.judge_prompts["reversed_prompt_template"]
                ),
                judgement=judgement,
                question=self.ideal_judgment_df["question"][i],
                ideal_prompt=ideal_prompt,
            )
            for i, judgement in enumerate(self.ideal_judgment_df["judgments"])
        ]
        compiled_cot = optimiser.compile(cot, trainset=trainset)
        compiled_cot.save("compiled_cot.json")


if __name__ == "__main__":
    # Load questions and answers
    output_dir = pathlib.Path("output/test-2024-05-22-21-30")
    tuner = PromptTuner(
        ideal_judgement_path=output_dir / "enriched_judgments",
        baseline_llm="gpt-3.5-turbo",
        judge_prompts={
            "system": "662813e0e25b6076a9e03df8",
            "prompt_template": "662821e23eb9ef01018e30e2",
            "reversed_prompt_template": "6628224eb84c0693351ca6a4",
        },
    )
    ideal_prompt = """
    Please act as an impartial judge to determine if the candidate answer is better, similarly good, or worse than the reference answer in response to the user query in the conversation.
    When judging which answer is better, consider the following criteria one by one:
    1. Does one answer follow **all user instructions** and the other one fails to do so?
        - For example, if the user asks to write a detailed explanation, then summarize the explanation, are both the detailed and the summarized version present?
        - For example, if the user asks to correct all grammar mistakes in the following essay, does the response go over all paragraphs of the essay or stops after the first paragraph?
        - For example, however, if the user asks to fill in the missing word in the sentence, it's ok to just provide the word as an answer without rewriting the sentence.
        - For example, if the user asks for the right answer without asking for an explaination, it's acceptable to not provide an explaination.
    2. Does one answer respond to the user question and the other one mis-interpret the user question?
    3. Is one answer less reasonable than the other given the context of the conversation?
    4. If the question has an objectively correct answer and the candidate and reference answers have different results, one must be better than the other. First solve the problem independently by thinking step by step, and see if your answer aligns with either the reference or candidate answers. If neither answer is correct, they are tied.
        - If both answers are correct, they are tied. The fact that one answer provides an explanation or a more through explanation does not make it better.
    5. If for any reason one answer refused to answer the question or fulfill the request, it is automatically the worse answer.
    
    Keep the following in mind while conducting your analysis:
    - DO NOT prefer an answer because it provided explanation or more detailed justifications. As long as both answers are functionally equivalent, they should tie.
    - If the candidate and reference answer interpreted the user question differently but both interpretations are reasonable, they should tie.
    - Do not bias towards longer or shorter answers.
    - The reference answer may or may not be correct.
    
    Begin your evaluation by judging the candidate answer on each of the 4 criteria above without making a decision. Think step by step and explain your reasoning. Then, decide if the candidate answer is as good as the reference answer.
    Avoid any position biases and ensure that the order in which the candidate and reference answers were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
    After you output your analysis on the criteria, output your final verdict at the end by **strictly following the following format**:
    ```Final Verdict: "[[A]]"``` if candidate answer is better than the reference answer.
    ```Final Verdict: "[[B]]"``` if the candidate is worse than the reference answer.
    ```Final Verdict: "[[C]]"``` if they are similar in quality.        
"""
    question = """
    user: If you are bitten or stung by a snake, funnel web spider, mouse spider, blue ringed octopus or cone shell it needs to be treated with:
    Select one of the following
    RICER
    PIT
    CPR
    All the aboveassistant: CPRuser: Box, Irukandji, Morbakka and Jimble Jellyfish stings should be treated with:
    Select one of the following
    Vinegar
    PIT
    Petrol
    Fresh waterassistant: Vinegaruser: If you receive a sting from a bluebottle, stonefish or sea urchin, how is it treated?
    Select one of the following
    Vinegar
    Fresh water
    PIT
    Hot water
    """
    judgement = """
    ['1. **Does one answer follow all user instructions and the other one fails to do so?**\n   - Both the reference and candidate answers follow the user\'s instruction by selecting one of the given options as the treatment method. There are no additional instructions to follow, such as providing detailed explanations or summaries.\n\n2. **Does one answer respond to the user question and the other one mis-interpret the user question?**\n   - Both answers correctly interpret the user\'s question, which asks for the appropriate treatment method for stings from a bluebottle, stonefish, or sea urchin.\n\n3. **Is one answer less reasonable than the other given the context of the conversation?**\n   - Both answers provide the same response, "Hot water," which is reasonable and appropriate given the context of the question. There is no difference in the reasonableness of the answers.\n\n4. **If the question has an objectively correct answer and the candidate and reference answers have different results, one must be better than the other.**\n   - In this case, both the candidate and reference answers provide the same result, "Hot water," which is the correct treatment for stings from a bluebottle, stonefish, or sea urchin according to the options provided.\n\nGiven the analysis, both answers are equivalent in quality, correctly following the user\'s instructions, accurately interpreting the question, providing a reasonable response, and aligning with the correct answer.\n\nFinal Verdict: "[[C]]"', '1. Both the candidate and reference answers follow the user\'s instruction by selecting one of the given options without providing unnecessary explanations, which aligns with the user\'s request for a direct answer.\n2. Both answers correctly respond to the user\'s question without misinterpreting it.\n3. Neither answer is less reasonable than the other; both correctly identify "Hot water" as the treatment for stings from a bluebottle, stonefish, or sea urchin.\n4. The question has an objectively correct answer, and both the candidate and reference answers provide the same correct response.\n\nFinal Verdict: "[[C]]"']"""

    tuner.train_data(ideal_prompt)

    refined_prompt = tuner.refined_prompt(
        ideal_prompt=ideal_prompt,
        initial_prompt="decide which is better.",
        question=question,
        judgement=judgement,
    )
    print("Refined Prompt: ", refined_prompt)
