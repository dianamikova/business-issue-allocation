import gradio as gr

from src.inference import Predictor

INPUT_GUIDANCE = """
Write 1-2 sentences that describe:
- the business problem
- the data situation
- the desired outcome or constraint
"""


def build_demo():
    predictor = Predictor()

    def classify_issue(text: str):
        if not text or not text.strip():
            return "Please enter a business issue description.", "", {}
        result = predictor.predict(text)
        confidence_text = f'{result["confidence"] * 100:.1f}%'
        top_scores = {
            prediction["label"]: round(prediction["confidence"], 4)
            for prediction in result["top_predictions"]
        }
        return result["predicted_label"], confidence_text, top_scores

    return gr.Interface(
        fn=classify_issue,
        inputs=gr.Textbox(
            label="Business issue description",
            lines=5,
            placeholder=(
                "Example: We combine customer data from 12 systems every night "
                "and need one trusted source for monthly KPI reporting."
            ),
        ),
        outputs=[
            gr.Textbox(label="Predicted label"),
            gr.Textbox(label="Confidence"),
            gr.Label(label="Top predictions"),
        ],
        title="Business Issue Allocation",
        description=(
            "Classify a business data problem into the most likely solution label.\n\n"
            f"{INPUT_GUIDANCE}"
        ),
        examples=[
            [
                "We need to detect fraudulent credit card transactions immediately before approval."
            ],
            [
                "We combine finance data from multiple systems every month and need one trusted source for group reporting."
            ],
            [
                "Analysts keep seeing different definitions of customer across systems and do not know which table is authoritative."
            ],
        ],
    )


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
