"""Gradio front end for the Module 8 tuned credit-default network."""

from pathlib import Path

import gradio as gr

from model import CreditRiskPredictor

predictor = CreditRiskPredictor(Path(__file__).parent / "artifacts")

PAY_CHOICES = [
    ("-2: no consumption", -2),
    ("-1: paid in full", -1),
    ("0: revolving credit", 0),
] + [(f"{m}: {m} month{'s' if m > 1 else ''} late", m) for m in range(1, 10)]
MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def score(limit_bal, sex, education, marriage, age, *rest):
    pays, bills, paid = rest[:6], rest[6:12], rest[12:18]
    record = {"LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
              "MARRIAGE": marriage, "AGE": age}
    record.update(dict(zip(PAY_COLS, pays)))
    record.update({f"BILL_AMT{i + 1}": v for i, v in enumerate(bills)})
    record.update({f"PAY_AMT{i + 1}": v for i, v in enumerate(paid)})
    p, label = predictor.predict_one(record)
    verdict = "Likely to DEFAULT next month" if label else "Likely to pay next month"
    return {"Default": p, "No default": 1 - p}, verdict


with gr.Blocks(title="Credit Default Risk") as demo:
    gr.Markdown(
        "# Credit-card default risk\n"
        "Tuned feed-forward neural network from my AAI 6610 Module 8 assignment, "
        "trained on the UCI *Default of Credit Card Clients* dataset (Taiwan, 2005). "
        "Amounts are in NT dollars. Decision threshold = 0.5. "
        "**Course project only — not for real lending decisions.**"
    )
    with gr.Row():
        limit_bal = gr.Number(50000, label="Credit limit (LIMIT_BAL)")
        sex = gr.Radio([("Male", 1), ("Female", 2)], value=2, label="Sex")
        education = gr.Dropdown([("Graduate school", 1), ("University", 2),
                                 ("High school", 3), ("Other", 4)], value=2, label="Education")
        marriage = gr.Dropdown([("Married", 1), ("Single", 2), ("Other", 3)],
                               value=2, label="Marital status")
        age = gr.Slider(21, 79, value=35, step=1, label="Age")
    gr.Markdown("### Repayment status by month (most recent first)")
    with gr.Row():
        pays = [gr.Dropdown(PAY_CHOICES, value=0, label=f"Status {m}") for m in MONTHS]
    gr.Markdown("### Bill statement amount")
    with gr.Row():
        bills = [gr.Number(20000, label=f"Bill {m}") for m in MONTHS]
    gr.Markdown("### Amount paid")
    with gr.Row():
        paid = [gr.Number(2000, label=f"Paid {m}") for m in MONTHS]

    btn = gr.Button("Score client", variant="primary")
    probs = gr.Label(label="Predicted probability")
    verdict = gr.Textbox(label="Decision")
    inputs = [limit_bal, sex, education, marriage, age, *pays, *bills, *paid]
    btn.click(score, inputs, [probs, verdict], api_name="predict")

    gr.Examples(
        examples=[
            [50000, 2, 2, 2, 35, 0, 0, 0, 0, 0, 0,
             20000, 20000, 20000, 20000, 20000, 20000, 2000, 2000, 2000, 2000, 2000, 2000],
            [20000, 1, 3, 1, 45, 2, 2, 2, 2, 2, 2,
             19000, 18500, 18000, 17500, 17000, 16500, 0, 0, 0, 0, 0, 0],
            [300000, 2, 1, 1, 40, -1, -1, -1, -1, -1, -1,
             5000, 4000, 6000, 3000, 4500, 5000, 5000, 4000, 6000, 3000, 4500, 5000],
        ],
        inputs=inputs,
        label="Example clients (typical / chronically late / pays in full)",
    )

if __name__ == "__main__":
    demo.launch()
