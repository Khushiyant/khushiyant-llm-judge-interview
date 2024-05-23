# LLM Judge
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/) [![LLM](https://img.shields.io/badge/LLM-Judge-orange)](https://github.com/withmartian/llm-judge)

----

LLM Judge is a comprehensive toolset designed to analyze the outputs of multiple Language Model Models (LLMs) across a given set of prompts. It offers various functionalities, including generating outputs at different temperatures, determining winner labels by comparing LLM outputs to ground truth, selecting subsets of completions based on winner labels and LLMs, and conducting in-depth reviews of individual completions.

The toolset allows users to examine prompts, completions, ground truth, winner labels, and provides commentary on the reasons behind the chosen winner label. Additionally, it provides summary charts that describe individual LLM performance, oracle LLM performance, cost vs performance analysis, and configurations for defining true positives based on winner labels across different LLMs.

One of the key features of LLM Judge is its flexibility. Users can implement custom PromptGenerator modules and LLMJudge modules to tailor the toolset to their specific needs and requirements.

---

## Table of Contents
- [How to use the LLM Judge](#how-to-use-the-llm-judge)
  - [Step 1: Installation](#step-1-installation)
  - [Step 2: Understanding the gen_question, gen_answer, and gen_judgments configurations](#step-2-understanding-the-gen_question-gen_answer-and-gen-judgments-configurations)
  - [Step 3: Generating Questions, Answers, and Judgments](#step-3-generating-questions-answers-and-judgments)
  - [Step 4: Inspecting Ouptuts](#step-4-inspecting-ouptuts)



# How to use the LLM Judge
## Step 1: Installation
Create a conda environment with Python 3.11
```
conda create --name llm-judge python=3.11
```
Activate this environment.
```
conda activate llm-judge
```

 Clone the Adapters package locally in project root.
```
git clone https://github.com/withmartian/adapters.git
```
Go into the adapter's package and install it.
```
cd adapters
pip install poetry
poetry install
cd ..
```

Move the sub folder adapters folder to the root directory so, llm-judge can detect it as module with any changes.
```
mv adapters/adapters ./tmp && rm -rf adapters && mv ./tmp adapters
```

**If you have changed the name of the conda environment, please update it in the environment.yaml file.**
```
conda env update --file environment.yaml
```
Add a `.env` file at the root directory. The `.env` file should include the following information in addition to any API providers you intend to use directly or through adapters.

**USE COMPANY KEYS OTHERWISE YOU WIL GET RATE LIMITED**
```
OPENAI_API_KEY=<OpenAI API Token>
ANTHROPIC_API_KEY=<Anthropic API Token>
MONGO_URI = "mongodb+srv://martian-user:p3ltlA9ILlX2jgIg@atlascluster.wh6g5zs.mongodb.net/"
MARTIAN_MONGO_URI = "<martian mongo uri>"
```
Install pre-commit. To run pre-commit manually, run `pre-commit run --all-files`
```
pre-commit install
```
Finally, install the llm_judge package.
```
pip install -e .
```
## Step 2: Understanding the gen_question, gen_answer, and gen_judgments configurations
All configurations for the experiment are stored in yaml files in `scripts/configs`. These files contain parameters such as `printout_limit`, `num_workers`, `seed`, `output_dir`, and `--output_enriched`. The `gen_questions.yaml` file includes settings for question generation, classifiers, ground truth generation, and filtering. The `gen_answers.yaml` file specifies LLN parameters and the number of times to generate answers. The `gen_judgments.yaml` file includes the judge class and the number of times to generate judgments.

## Step 3: Generating Questions, Answers, and Judgments
After editing the yaml files, run this command to dump the questions from mongo as pickle:
```bash
python scripts/dump_martian_db.py --config scripts/configs/dump_martian_db.yaml
```

Change the questions generator to `QueQuestionPickleGenerator` in `gen_questions.yml` to read the dump file and then

```
python scripts/gen_questions.py --config scripts/configs/gen_questions.yaml
```
Change the output_dir arg of `gen_answers.yaml` and `gen_judgments.yaml` and run
```
python scripts/gen_answers.py --config scripts/configs/gen_answers.yaml
```
```
python scripts/gen_judgments.py --config scripts/configs/gen_judgments.yaml
```


Note: make sure to set the `MARTIAN_MONGO_URI` to the correct URI in the `.env` file to pull data

## Step 4: Inspecting Ouptuts

Run the `EvalJudge` located in `automator` to inspect the outputs. The `EvalJudge` class will generate a record of the outputs and save them to the specified output directory


NOTE: In case if you want automated generation of questions, answers, and judgments, run `automate.sh` script. This script will run the above commands in sequence.