import dspy


class AssessPrompt(dspy.Signature):
    """Assess the quality of a prompt based on a set of criteria."""

    criteria = dspy.InputField(
        desc="The criteria to assess the refined and initial prompt."
    )

    reference_answer = dspy.InputField()
    answer = dspy.InputField()
    judgement = dspy.InputField()
    initial_prompt = dspy.InputField(desc="The initial prompt to be refined.")

    better_prompt = dspy.InputField()

    quality = dspy.OutputField(
        desc="Just answer in range 0 to 1. DON'T USE STRINGS, JUST NUMBERS."
    )


def overall_metric(gold, pred, trace=None):
    reference_answer, judgement, initial_prompt, better_prompt = (
        gold.reference_answer,
        gold.judgement,
        gold.initial_prompt,
        pred.prompt,
    )

    criteria = "How much is the assessed refined prompt better than the initial prompt in aspect of comparing the reference answer to answer?"
    criteria_result = dspy.Predict(AssessPrompt)(
        reference_answer=reference_answer,
        judgement=judgement,
        criteria=criteria,
        answer=gold.answer,
        initial_prompt=initial_prompt,
        better_prompt=better_prompt,
    )
    return float(criteria_result.quality)
