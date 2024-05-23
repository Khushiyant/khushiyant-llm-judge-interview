"""
given a ground truth, evaluate quailty of the output of a judge.
"""

import os.path

import pandas as pd
from datetime import datetime
from typing import Literal
from llm_judge.database.models.prompts import Prompt
from llm_judge.utils.logger import info_logger as logging
import ast
import shutil
import json
import random

from scripts.gen_judgments import main as gen_judgments
from llm_judge.judges import BaselineComparisonJudge
from llm_judge.judges.concrete_judges.dynamic_few_shot_judge import DynamicFewShotJudge
from llm_judge.judges.abstract_judges.judge import Judge
from llm_judge.utils.common import load_from_json


class EvalJudge:
    """
    Evaluate the quality of a judge prompt by comparing it to a ground truth.

    Parameters:
    - ideal_judgment_path (str): The path to the ground truth judgment file.
    - judge_class (Judge): The class representing the judge to be used for evaluation.
    - experiments_run_folder (str): The folder where the evaluation results will be stored.
    - raw_data_folder (str): The folder containing the raw data for evaluation.
    - test_split (float): The fraction of data to be used for testing.
    - random_seed (int): The random seed for reproducibility.
    - judgments_for_few_shot_retrieval_fp (str): The file path for judgments used for few-shot retrieval.

    Attributes:
    - judge_class (Judge): The class representing the judge to be used for evaluation.
    - datetime_str (str): The current date and time in the format 'mm-dd_HH-MM-SS'.
    - raw_data_folder (str): The folder containing the raw data for evaluation.
    - ideal_judgment_df (pd.DataFrame): The DataFrame containing the ground truth judgments.
    - train_df (pd.DataFrame): The DataFrame containing the training data.
    - test_df (pd.DataFrame): The DataFrame containing the testing data.
    - output_dir (str): The directory where the evaluation results will be stored.
    - judgments_for_few_shot_retrieval_fp (str): The file path for judgments used for few-shot retrieval.

    Methods:
    - copy_config_files(): Copy the configuration files to the output directory.
    - evaluate_judge_prompt(judge_prompt: str, eval_on: Literal["train", "test", "all"]) -> pd.DataFrame:
        Evaluate the quality of a judge prompt by comparing it to a ground truth.
    - run_judge_with_prompt(judge_prompt: str) -> pd.DataFrame:
        Execute a judge prompt and return the results.
    - _prep_prompt(judge_prompt: str) -> str:
        Prepare the judge prompt by adding a base template.
    - _prep_config(judge_prompt_id: str) -> dict:
        Prepare the configuration data for running the judge.
    - _export_all_ids(judgment_df_to_eval: pd.DataFrame, output_folder: str):
        Export all question and answer IDs to JSON files.
    """

    def __init__(
        self,
        ideal_judgment_path: str,
        judge_class: Judge = BaselineComparisonJudge,
        experiments_run_folder: str = "output/auto_judge/",
        raw_data_folder: str = "data/DeepAI/test-2024-04-30-17-52",
        test_split: float = 0.9,
        random_seed: int = 42,
        judgments_for_few_shot_retrieval_fp: str = None,
    ):
        # set seed
        random.seed(random_seed)
        self.judge_class = judge_class
        self.datetime_str = datetime.now().strftime("%m-%d_%H-%M-%S")
        self.raw_data_folder = raw_data_folder
        self.ideal_judgment_df = pd.read_csv(ideal_judgment_path)
        self.train_df = self.ideal_judgment_df.sample(frac=1 - test_split)
        self.test_df = self.ideal_judgment_df.drop(self.train_df.index)
        self.output_dir = os.path.join(experiments_run_folder, self.datetime_str)
        self.judgments_for_few_shot_retrieval_fp = judgments_for_few_shot_retrieval_fp
        os.makedirs(self.output_dir, exist_ok=True)

        self.copy_config_files()

    def copy_config_files(self):
        # Copy gen_questions_config.yaml and gen_answers_config.yaml from raw_data_folder to output_dir
        shutil.copy(
            os.path.join(self.raw_data_folder, "gen_questions_config.yaml"),
            self.output_dir,
        )
        shutil.copy(
            os.path.join(self.raw_data_folder, "gen_answers_config.yaml"),
            self.output_dir,
        )

        assert "score" in self.ideal_judgment_df.columns
        # assert "judgements" in self.ideal_judgment_df.columns

        self.config_template = {
            "judge": {
                "init_args": {
                    "baseline_llm": "gpt-3.5-turbo-0125",
                    "judge_prompts": {
                        "system": None,
                        "prompt_template": "662821e23eb9ef01018e30e2",
                        "reversed_prompt_template": "6628224eb84c0693351ca6a4",
                    },
                    "judge_llm": "gpt-4-turbo-preview",
                    "judge_llm_params": {"temperature": 0, "max_tokens": 3000},
                },
            },
            "N": 1,
            "output_dir": self.output_dir,
            "num_workers": 30,
            "printout_limit": 500,
            "output_enriched": True,
        }

        if self.judge_class == DynamicFewShotJudge:
            assert (
                self.judgments_for_few_shot_retrieval_fp
            ), "judgments_for_few_shot_retrieval_fp must be provided for DynamicFewShotJudge"
            self.config_template["judge"]["init_args"][
                "ideal_judgment_path"
            ] = self.judgments_for_few_shot_retrieval_fp

    def evaluate_judge_prompt(
        self, judge_prompt: str, eval_on=Literal["train", "test", "all"]
    ) -> pd.DataFrame:
        """
        Evaluate the quality of a judge prompt by comparing it to a ground truth.
        """
        match eval_on:
            case "train":
                eval_df = self.train_df
            case "test":
                eval_df = self.test_df
            case "all":
                eval_df = self.ideal_judgment_df
            case _:
                raise ValueError("eval_data must be 'train', 'test', or 'all'")

        self._export_all_ids(eval_df, self.output_dir)

        try:
            exp_judge_results = self.run_judge_with_prompt(judge_prompt)
            merged_results = pd.merge(
                exp_judge_results,
                self.ideal_judgment_df[["answer_id", "score_x"]],
                on="answer_id",
                how="left",
            )
            merged_results["exp_judge_score"] = merged_results["score_x"].apply(
                ast.literal_eval
            )
            merged_results["ideal_judge_score"] = merged_results["score_y"].apply(
                ast.literal_eval
            )
            logging.info(
                "The judge score is: ",
                (
                    merged_results["exp_judge_score"]
                    == merged_results["ideal_judge_score"]
                ).mean(),
            )
        except Exception as e:
            logging.info(f"Error: {e}")
        return merged_results

    def run_judge_with_prompt(self, judge_prompt: str) -> pd.DataFrame:
        """
        Execute a judge prompt and return the results.
        """
        judge_prompt_id = self._prep_prompt(judge_prompt)
        configs = self._prep_config(judge_prompt_id)
        judge = self.judge_class(**configs["judge"]["init_args"])

        question_ids = load_from_json(
            os.path.join(configs["output_dir"], "question_ids.json")
        )
        answer_ids = load_from_json(
            os.path.join(configs["output_dir"], "answer_ids.json")
        )

        result_fp = gen_judgments(
            output_dir=configs["output_dir"],
            judge=judge,
            question_ids=question_ids,
            answer_ids=answer_ids,
            N=configs["N"],
            num_workers=configs["num_workers"],
            printout_limit=configs["printout_limit"],
            output_enriched=configs["output_enriched"],
        )
        logging.info("the ↑ above result is experimental judge results")
        enriched_judge_results = pd.read_csv(result_fp)

        return enriched_judge_results

    def _prep_prompt(self, judge_prompt: str) -> str:
        judge_prompt_base = """
    Think step by step and explain your reasoning. Then, decide if the candidate answer is as good as the reference answer.
    Avoid any position biases and ensure that the order in which the candidate and reference answers were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
    After you output your analysis on the criteria, output your final verdict at the end by **strictly following the following format**:
    ```Final Verdict: "[[A]]"``` if candidate answer is better than the reference answer.
    ```Final Verdict: "[[B]]"``` if the candidate is worse than the reference answer.
    ```Final Verdict: "[[C]]"``` if they are similar in quality.
        """
        judge_prompt_obj = Prompt(
            description="test judge prompt in eval_judge.py",
            content=judge_prompt + judge_prompt_base,
            tags=["test", "baseline_pairwise", "reversed_prompt_template"],
        )
        judge_prompt_obj = judge_prompt_obj.get_or_save()
        judge_prompt_id = judge_prompt_obj.id
        return str(judge_prompt_id)

    def _prep_config(self, judge_prompt_id: str) -> dict:
        config_data = self.config_template.copy()
        config_data["judge"]["init_args"]["judge_prompts"]["system"] = str(
            judge_prompt_id
        )

        return config_data

    def _export_all_ids(self, judgment_df_to_eval: pd.DataFrame, output_folder: str):
        judgment_df_to_eval.question_id.to_json(
            os.path.join(output_folder, "question_ids.json"), orient="values"
        )
        answer_ids = judgment_df_to_eval.answer_id.to_list()
        answer_ids += judgment_df_to_eval.comparison_answers.apply(
            lambda s: s.replace("[ObjectId('", "").replace("')]", "")
        ).to_list()
        with open(os.path.join(output_folder, "answer_ids.json"), "w") as file:
            json.dump(answer_ids, file)


if __name__ == "__main__":
    eval_judge2 = EvalJudge(
        ideal_judgment_path="output/test-2024-05-22-21-30/enriched_judgments",
        judge_class=BaselineComparisonJudge,
        raw_data_folder="output/test-2024-05-22-21-30",
        test_split=0.1,
    )
    null_prompt = ""
    negative_prompt = "determine which one is better"
    original_prompt = """
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
    i = 0
    for eval_judge in [eval_judge2]:
        for judge_prompt in [negative_prompt, null_prompt, original_prompt]:
            i += 1

            logging.info(f"========\n== {i}  ==\n========")
            eval_judge.evaluate_judge_prompt(judge_prompt, eval_on="test")
