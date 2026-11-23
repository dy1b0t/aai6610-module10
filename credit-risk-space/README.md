---
title: AAI6610 Module10
emoji: 📉
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Assignment for AAI6610 Module 10 using Credit Risk model
---

# Credit-card default risk (AAI 6610 Module 10)

Gradio app serving the tuned feed-forward neural network from my Module 8
assignment (64-32-16-8 MLP, learning rate 1e-4, dropout 0.5), trained on the
UCI *Default of Credit Card Clients* dataset (Yeh, 2016).

Held-out test performance (6,000 clients): accuracy 0.813, default-class
precision 0.605, recall 0.446, F1 0.513.

## Layout

| Path | Purpose |
| --- | --- |
| `app.py` | Gradio UI and `/predict` API endpoint |
| `model.py` | Network definition and inference wrapper (scaling + sigmoid + threshold) |
| `artifacts/model.pt` | Trained weights from Module 8 |
| `artifacts/scaler.json` | StandardScaler mean/scale fit on the training split only |
| `tests/` | Pytest suite run by CI before every deploy |

## CI/CD

Every push to `main` runs the tests on GitHub Actions; only if they pass is
the `credit-risk-space/` folder mirrored to the Space with `huggingface/hub-sync`
(the workflow itself lives at the GitHub repo root, `.github/workflows/ci-cd.yml`). Configure:

- Repository **secret** `HF_TOKEN`: a Hugging Face token with write access.
- Repository **variable** `HF_SPACE_ID`: `Dylbot/AAI6610-Module10`.

## Local run

```bash
pip install -r requirements.txt gradio==6.32.0 pytest
pytest -v
python app.py
```

*Course project only; not intended for real credit decisions.*
