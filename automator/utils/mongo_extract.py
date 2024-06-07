import pymongo
import bson
import os


class MongoData:
    @staticmethod
    def get_prompt(prompt_id):
        client = pymongo.MongoClient(os.environ["MARTIAN_MONGO_URI"])

        prompts = client["llm-judge"]["prompts"]
        prompt = prompts.find_one({"_id": bson.ObjectId(prompt_id)})
        return prompt["content"]
