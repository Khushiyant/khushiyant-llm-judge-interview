import dspy


# Start: Prompt Tuner Module and Signature
class Refiner(dspy.Signature):
    """Generate a refined prompt while taking question, answer and judgemnt in account."""

    reference_answer = dspy.InputField()
    initial_prompt = dspy.InputField()
    answer = dspy.InputField()
    judgement = dspy.InputField()

    better_prompt = dspy.OutputField()


class Prompter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_choices = dspy.ChainOfThought(Refiner)

    def forward(self, reference_answer, answer, judgement, initial_prompt):
        prompt = self.generate_choices(
            reference_answer=reference_answer,
            answer=answer,
            judgement=judgement,
            initial_prompt=initial_prompt,
        ).better_prompt
        return dspy.Prediction(prompt=prompt)


# End: Prompt Tuner Module and Signature
