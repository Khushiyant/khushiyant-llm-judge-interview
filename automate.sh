# Conda environment: llm-judge
conda create --name llm-judge python=3.11
conda activate llm-judge

# Install dependencies
git clone https://github.com/withmartian/adapters.git
cd adapters
pip install poetry
poetry install
cd ..

mv adapters/adapters ./tmp && rm -rf adapters && mv ./tmp adapters

# Generate questions, answers, and judgments
python scripts/gen_questions.py --config scripts/configs/gen_questions.yaml
python scripts/gen_answers.py --config scripts/configs/gen_answers.yaml
python scripts/gen_judgments.py --config scripts/configs/gen_judgments.yaml