import random
import pickle

from typing import Any, Optional, List, Dict, TypeVar
from pathlib import Path
import dataclasses
from adapters.types import Conversation
from llm_judge.question_generator.base_question_generator import BaseQuestionGenerator
from llm_judge.database.models.questions import Question
from llm_judge.utils.common import stringify_conversation, gen_hash
import contextlib

T = TypeVar("T")


@dataclasses.dataclass
class QuestionPickleGenerator(BaseQuestionGenerator):
    """
    A question generator that loads data from a pickle file and generates a list of question objects.

    Args:
        data_path (Path): The path to the input data file.
        num_samples (Optional[int]): The number of samples to generate. If None, all samples will be used. Default is None.
        random_seed (int): The random seed for shuffling the data. Default is 42.

    Methods:
        load_data() -> List[T]: Load the data from the input file.
        get_ground_truth_from_entry_or_none(entry: Any) -> Optional[str]: Extract a ground_truth string from a data entry if it exists.
        get_conversation_from_entry(entry: Any) -> Conversation: Extract a conversation from a data entry.
        get_args_from_entry(entry: Any) -> Dict[str, Any]: Extract additional arguments from a data entry.
        make_questions() -> List[Question]: Generate a list of question objects from the data.
    """

    data_path: Path
    num_samples: Optional[int] = None
    random_seed: int = 42

    def load_data(self) -> List[T]:
        """
        Load the data from the input file.
        """
        with contextlib.ExitStack() as stack:
            f = stack.enter_context(open(self.data_path, "rb"))
            data = pickle.load(f)
        return data

    def get_ground_truth_from_entry_or_none(self, entry: Any) -> Optional[str]:
        """
        Extract a ground_truth string from a data entry if it exists
        """
        return entry.get("ground_truth")

    def get_conversation_from_entry(self, entry: Any) -> Conversation:
        """
        Extract a conversation from a data entry.
        """
        return Conversation(entry["conversation"])

    def get_args_from_entry(self, entry: Any) -> Dict[str, Any]:
        """
        Extract additional arguments from a data entry.
        """
        args = {key: entry[key] for key in entry if key == "args"}
        return args

    def make_questions(self) -> List[Question]:
        """
        Generate a list of question objects from the data.
        """
        data = self.load_data()
        # Filter empty conversations
        data = [entry for entry in data if entry]
        if self.num_samples is None:
            random.seed(self.random_seed)
            random.shuffle(data)
            if self.num_samples and self.num_samples < len(data):
                data = data[: self.num_samples]
        questions = []
        for entry in data:
            conversation = self.get_conversation_from_entry(entry)
            ground_truth = self.get_ground_truth_from_entry_or_none(entry)
            args = self.get_args_from_entry(entry)
            questions.append(
                Question(
                    conversation=conversation,
                    conversation_hash=gen_hash(stringify_conversation(conversation)),
                    ground_truth=ground_truth,
                    args=args,
                )
            )
        return questions
