import os
import pymongo
import pickle
from jsonargparse import ActionConfigFile, ArgumentParser
from dotenv import load_dotenv
import logging

"""
This script is setup to dump the martian database transactions to disk. 

This dumps to pickle as ObjectID is not serializable to JSON. 

"""

load_dotenv("./.env", override=True)


def get_prompts(connection_string: str) -> list:
    client = pymongo.MongoClient(connection_string)
    db = client["llm-judge"]["prompts"]
    prompts = list(db.find())
    return prompts


def get_questions(connection_string: str) -> list:
    client = pymongo.MongoClient(connection_string)
    db = client["llm-judge"]["questions"]
    questions = list(db.find())
    return questions


def main(connection_string: str, output_file: str, prompt_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    questions = get_questions(connection_string)
    logging.info(f"Dumping {len(questions)} questions to {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(questions, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--connection_string", type=str, default=os.environ["MARTIAN_MONGO_URI"]
    )
    parser.add_argument("--output_file", type=str, default="question_dump.pkl")
    parser.add_argument("--prompt_file", type=str, default="prompt_dump.pkl")
    parser.add_argument("--config", action=ActionConfigFile)
    args = parser.parse_args()
    main(args.connection_string, args.output_file, args.prompt_file)
