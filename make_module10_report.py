"""Generate the Word report (.docx) for the AAI 6610 Module 10 assignment:
deploying the tuned Module 8 network to a Hugging Face Space with a GitHub
Actions CI/CD pipeline."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(HERE, "..", "Module 8", "metrics8.json")) as f:
    M8 = json.load(f)
R = M8["reevaluation"]

# ------------------------------------------------------------ pipeline figure
fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.2)
ax.axis("off")
boxes = [
    (0.1, "Local dev\n(edit, pytest,\npython app.py)", "#dbe9f6"),
    (2.1, "GitHub repo\n(push / PR\nto main)", "#dbe9f6"),
    (4.1, "Actions: test job\n(Ubuntu, Py 3.11,\npip install, pytest)", "#fde9c9"),
    (6.1, "Actions: deploy job\n(hub-sync with\nHF_TOKEN secret)", "#fde9c9"),
    (8.1, "HF Space\n(CPU basic: build\n+ serve Gradio)", "#d9f2d9"),
]
for x, label, color in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.0), 1.75, 1.4, boxstyle="round,pad=0.05",
                                fc=color, ec="#333", lw=1))
    ax.text(x + 0.875, 1.7, label, ha="center", va="center", fontsize=8.5)
for x in [1.85, 3.85, 5.85, 7.85]:
    ax.add_patch(FancyArrowPatch((x + 0.02, 1.7), (x + 0.23, 1.7),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.4))
ax.text(5.0, 0.55, "deploy runs only if tests pass and the event is a push "
        "(pull requests are tested but never deployed)",
        ha="center", fontsize=8, style="italic", color="#444")
ax.text(5.0, 2.85, "CI/CD pipeline for the credit-default Space",
        ha="center", fontsize=11, weight="bold")
fig.tight_layout()
PIPE_FIG = os.path.join(FIG, "fig1_pipeline.png")
fig.savefig(PIPE_FIG)
plt.close(fig)

# ------------------------------------------------------------------ document
doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)


def h(text, level=1):
    return doc.add_heading(text, level=level)


def body(text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def figure(path, caption, width=6.2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.italic = True
        run.font.size = Pt(9)


def table(headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    for j, head in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = head
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.rows[i + 1].cells[j]
            cell.text = str(val)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9)
    doc.add_paragraph()


def caption(text):
    p = doc.add_paragraph(text)
    p.runs[0].italic = True


def reference(text):
    p = doc.add_paragraph(text)
    fmt = p.paragraph_format
    fmt.left_indent = Inches(0.5)
    fmt.first_line_indent = Inches(-0.5)
    fmt.space_after = Pt(8)


doc.add_heading("From Notebook to Endpoint: Deploying the Credit-Risk Network "
                "with Hugging Face Spaces and GitHub Actions", level=0)
for line in ["AAI 6610 – Applied Machine Learning · Module 10 Assignment",
             "Dylan Miller · Northeastern University"]:
    p = doc.add_paragraph(line)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph("Live Space: https://huggingface.co/spaces/Dylbot/AAI6610-Module10\n"
                      "GitHub repository: https://github.com/dy1b0t/aai6610-module10")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ------------------------------------------------------------- 1. Intro
h("1. Introduction")
body(
    "Up to this point, every model I have built in this course has lived in "
    "one place: a Python script on my laptop that loads a spreadsheet, "
    "trains, prints metrics and saves a few figures. That is enough to answer "
    "a research question, but a model does not do anything useful until "
    "someone else can hand it an input and get a prediction back. This "
    "assignment is about closing that gap. I took the tuned neural network "
    "from Module 8, which predicts whether a credit-card client will default "
    "next month, and deployed it as a public web app on a Hugging Face Space "
    "running on the free CPU-basic hardware tier (Hugging Face, n.d.-c). I "
    "then set up a continuous integration and continuous deployment (CI/CD) "
    "pipeline in GitHub Actions, so every change I push is tested "
    "automatically and only reaches the Space if those tests pass (Hugging "
    "Face, n.d.-a)."
)
body(
    "I have laid the report out in the same order I did the work. Section 2 "
    "recaps the model and why I chose it. Section 3 covers the deployment: "
    "packaging the model, building the interface, writing tests, and wiring "
    "up the pipeline, along with the problems I hit. Section 4 is the "
    "critical part, covering what this setup cannot do, what I would change, "
    "and what kind of situation would make it worth paying for something "
    "heavier. Section 5 sums up what I took away from it."
)

# ------------------------------------------------------------- 2. Models
h("2. Overview of Models, Training and Tuning")
body(
    "Everything I have done since Module 2 uses the UCI Default of Credit "
    "Card Clients dataset (Yeh, 2016). It covers 30,000 credit-card holders "
    "in Taiwan from 2005, each described by 23 features: credit limit, sex, "
    "education, marital status, age, six months of repayment status (PAY_0 "
    "through PAY_6), six months of bill amounts and six months of payment "
    "amounts. The label says whether the client defaulted the following "
    "month. Only 22.1% of clients did, so the classes are imbalanced, and a "
    "model that always says \u201cno default\u201d already gets 0.779 accuracy. "
    "Every model I have trained shares the same cleaning steps (folding the "
    "undocumented EDUCATION and MARRIAGE codes into an \u201cother\u201d group) "
    "and the same seed-locked, stratified 80/20 split of 24,000 training and "
    "6,000 test clients."
)
body(
    "Over the midterm and Module 8 I compared logistic regression, a depth-5 "
    "decision tree, a 200-tree random forest, and a multilayer perceptron "
    "built in PyTorch (Paszke et al., 2019), which I trained in a standard "
    "and a cost-sensitive version. In Module 8 I re-scored all of them on "
    "the default class, then ran an 18-configuration grid search over "
    "learning rate, dropout and architecture for the neural network, "
    "picking the winner by validation F1 instead of accuracy. Table 1 shows "
    "how the main candidates finished."
)
caption("Table 1. Module 8 test-set results (6,000 clients; default class = positive).")
rows = []
for name in ["Logistic Regression", "Random Forest", "Neural Network (standard)",
             "Neural Network (cost-sensitive)", "Neural Network (tuned)"]:
    r = R[name]
    rows.append([name, f"{r['accuracy']:.3f}", f"{r['precision']:.3f}",
                 f"{r['recall']:.3f}", f"{r['f1']:.3f}", f"{r['roc_auc']:.3f}"])
table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], rows)
body(
    "The model I deployed is the tuned network: four hidden layers "
    "(64-32-16-8), each with batch normalization and ReLU, dropout of 0.5, "
    "trained with Adam at a learning rate of 1e-4 with early stopping on "
    "validation loss. It does not win on every metric, since the "
    "cost-sensitive network still catches more defaulters, but it is what my "
    "Module 8 tuning actually produced. It balances accuracy (0.813) and F1 "
    "(0.513) well, and its weights were already saved. It is also tiny: the "
    "weights file is about 29 KB and one prediction takes well under a "
    "millisecond on a CPU, so free hardware is more than enough. Picking the "
    "neural network over one of the scikit-learn models also made the "
    "exercise more realistic, because it meant shipping PyTorch as a "
    "dependency, and that turned out to be one of the more interesting "
    "problems."
)

# ------------------------------------------------------------- 3. Deployment
h("3. Overview of Model Deployment and CI/CD Pipeline Creation")
h("3.1 Packaging the model", level=2)
body(
    "My first problem came up before I had written any app code. In Module 8 "
    "I saved the network's weights but not the StandardScaler the network "
    "was trained behind. The network never saw raw NT-dollar amounts; it "
    "only ever saw standardized values. Without the scaler, the saved "
    "weights cannot be used to make predictions. Sculley et al. (2015) "
    "describe exactly this kind of hidden technical debt: the model is not "
    "just the weights file but the weights plus every preprocessing step "
    "that shaped the inputs. I fixed it with a short export script that "
    "rebuilds the Module 8 pipeline exactly (same cleaning, same split, same "
    "seed), refits the scaler on the 24,000 training rows only, and writes "
    "its means and scales to a JSON file next to the weights. I chose JSON "
    "over a pickle on purpose. It is readable, it shows up cleanly in Git "
    "diffs, and it avoids the security and version-compatibility risks of "
    "unpickling Python objects."
)
body(
    "To make sure the packaged model really was the same model, the export "
    "script runs all 6,000 test clients through the new inference code. It "
    "matched Module 8 exactly, with accuracy of 0.8130 and F1 of 0.5134, so "
    "the deployed app and the numbers in my reports describe the same "
    "thing. The script also saves ten real test clients (five predicted "
    "defaulters and five predicted non-defaulters) along with their "
    "predicted probabilities, which the test suite uses as a regression "
    "fixture."
)
h("3.2 Building the app", level=2)
body(
    "Spaces supports a few different SDKs. I used Gradio because it turns a "
    "Python function into a web form and a REST API with very little code "
    "(Abid et al., 2019; Hugging Face, n.d.-d). The project is split into "
    "two files. model.py holds the network class and a small "
    "CreditRiskPredictor wrapper that loads the artifacts, standardizes the "
    "input, runs the forward pass, applies the sigmoid and compares the "
    "probability to the 0.5 threshold. app.py builds the interface: a row "
    "for credit limit and demographics, six dropdowns for repayment status "
    "with plain-English labels (\u201c2: 2 months late\u201d instead of a bare "
    "number), and six inputs each for bill and payment amounts. The output "
    "is a probability bar for default versus no default plus a one-line "
    "decision. Three example clients (typical, chronically late, and pays in "
    "full) let a visitor try the model in one click, and the page says "
    "plainly that it is a course project, not a lending tool. Keeping the "
    "model logic out of the UI file paid off later, because the tests can "
    "import the predictor without starting a web server."
)
body(
    "Hugging Face reads a Space's settings from a YAML block at the top of "
    "README.md, which sets the SDK, the Gradio version (sdk_version), the "
    "Python version and the app file (Hugging Face, n.d.-b). Gradio itself "
    "is installed by the platform from sdk_version, so requirements.txt only "
    "lists the extra packages the app needs."
)
h("3.3 Testing", level=2)
body(
    "A deployment pipeline is only as good as the checks that gate it, so I "
    "wrote a pytest suite of 15 tests before touching GitHub Actions. The "
    "tests check that the artifacts load with the expected 23 features; that "
    "each of the ten fixture clients gets the same probability (within 1e-4) "
    "and label it got in Module 8; that every probability falls between 0 "
    "and 1; that a client who is three months late every month scores as "
    "riskier than the same client paying on time; that input with the wrong "
    "number of features raises an error rather than quietly returning a "
    "prediction; and that the Gradio app imports and its scoring function "
    "returns a well-formed result. The behavioural check and the input check "
    "follow the kind of model and infrastructure tests Breck et al. (2017) "
    "recommend for production readiness. The fixture tests matter most. If "
    "anyone (most likely me) swaps in new weights or changes the "
    "preprocessing, any drift in the predictions fails the build before it "
    "reaches users."
)
h("3.4 The CI/CD pipeline", level=2)
figure(PIPE_FIG, "Figure 1. The CI/CD pipeline from local change to live Space.")
body(
    "The Hugging Face documentation now points to the official "
    "huggingface/hub-sync GitHub Action. It mirrors the repository's files "
    "to the Space using the hf command-line tool, replacing the older "
    "approach of git-pushing straight to the Space's remote (Hugging Face, "
    "n.d.-a). I used it, but the documented example only deploys: every "
    "push to main gets synced whether or not anything was tested. That is "
    "continuous deployment with no continuous integration, so I split my "
    "workflow into two jobs. My GitHub repository holds the whole Module 10 "
    "project (the export script, the report generator and the app), but "
    "only the credit-risk-space folder belongs on the Space. The test job "
    "therefore works inside that folder, and the deploy step uses the "
    "action's subdirectory option so that only the app's files are mirrored "
    "(Hugging Face, n.d.-e). The test job runs on an Ubuntu runner with "
    "Python 3.11, installs the pinned dependencies plus the same Gradio "
    "version the Space uses, and runs pytest. The deploy job declares "
    "needs: test, so GitHub only starts it after the test job succeeds "
    "(GitHub, n.d.-a). It is also skipped for pull requests, so a proposed "
    "change gets tested but never shipped before it is merged. For "
    "authentication, a Hugging Face access token with the write role is "
    "stored as an encrypted repository secret called HF_TOKEN, and the Space "
    "ID is stored as a repository variable (GitHub, n.d.-b), so neither the "
    "token nor my username appears in the workflow file. A "
    "workflow_dispatch trigger also lets me rerun the pipeline by hand from "
    "the Actions tab."
)
body(
    "The setup itself had four steps: create a free Gradio Space on CPU "
    "basic; create a Hugging Face access token with the write role; create "
    "the GitHub repository and add the secret and the variable; and push. "
    "The first push ran both jobs, the Space built from requirements.txt, "
    "and the app came up at its public URL. From then on, each commit showed "
    "up in the Actions tab as a green or red run, and only the green ones "
    "made it to the Space."
)
h("3.5 Challenges and lessons learned", level=2)
body(
    "Beyond the missing scaler, four other problems taught me something. "
    "The first was about repository hygiene. My first commit was made from "
    "the project folder with no .gitignore, so it swept in the whole local "
    "virtual environment, more than 41,000 files including PyTorch "
    "libraries larger than GitHub's 100 MB per-file limit, along with the "
    "Word copy of this report. The push failed. I added a .gitignore for "
    "the environment, caches and documents, and rewrote the commit so "
    "those files never appear in the history. At the same time I found "
    "that I had put the workflow file inside the app folder, where GitHub "
    "would never see it, because Actions only reads workflows from "
    ".github/workflows at the repository root (GitHub, n.d.-a). Moving it "
    "fixed that. "
    "The second was the size of the dependencies. A plain \u201cpip install "
    "torch\u201d on Linux pulls the CUDA build, which brings in several "
    "gigabytes of NVIDIA libraries that a CPU-only Space can never use. That "
    "slows down every Space build and every CI run. Pointing "
    "requirements.txt at PyTorch's CPU-only wheel index fixed it and cut the "
    "download to a small fraction of that size. The lesson for me is that "
    "on free hardware, the environment can be a bigger problem than the "
    "model: the model here is 29 KB and the framework is hundreds of "
    "megabytes."
)
body(
    "The third was a front-end bug my tests completely missed. In my first "
    "version, the repayment, bill and payment inputs all used the same month "
    "names as labels (\u201cSep\u201d, \u201cAug\u201d and so on). Python had no "
    "problem with that and all 15 tests passed, but when I opened the app "
    "in a browser, the examples table crashed with a JavaScript "
    "duplicate-key error because Gradio uses the labels as column keys. "
    "Adding prefixes (\u201cBill Sep\u201d, \u201cPaid Sep\u201d) fixed it. The "
    "takeaway is that a green pipeline only proves what the tests check, "
    "not that the product works. My suite covers the model and the Python "
    "callback but nothing that happens in the browser."
)
body(
    "The fourth was keeping versions consistent. The Space installs Gradio "
    "from sdk_version in the README, while CI installs it from the workflow "
    "file, and the two have to match or CI ends up testing a different app "
    "from the one users get. I pinned both to the same version, and pinned "
    "PyTorch as well, so a new upstream release cannot quietly change "
    "behaviour between a passing test run and a deployment. Finally, one "
    "modelling lesson came from actually using the app instead of reading a "
    "metrics table. The \u201cchronically late\u201d example client, who is two "
    "months behind for six months straight and paying nothing, only gets a "
    "55% default probability. That fits what Module 8 found: the tuned "
    "network is conservative at the 0.5 threshold and catches fewer than "
    "half of the real defaulters. Seeing it happen to one concrete client "
    "makes the weakness much more obvious than a recall of 0.446 ever did."
)

# ------------------------------------------------------------- 4. Limits
h("4. Limitations, Potential Improvements and Required Justifications")
h("4.1 Limitations of the deployment environment", level=2)
body(
    "The free CPU-basic tier gives a Space 2 vCPUs and 16 GB of RAM at no "
    "cost (Hugging Face, n.d.-c), which is far more than this model needs. "
    "The real limits lie elsewhere. On free hardware a Space goes to sleep "
    "after a stretch with no traffic, so the next visitor has to wait "
    "through a cold start while it restarts (Hugging Face, n.d.-c). There is "
    "no service-level agreement, no autoscaling and only one replica, so a "
    "single heavy user or a platform outage takes the whole service down. "
    "The Space is public, with no authentication or rate limiting of my "
    "own. That is fine for a demo built on public data, but it would be "
    "unacceptable for anything that handles real customers' financial "
    "records. The configuration reference notes that persistent storage is "
    "no longer offered (Hugging Face, n.d.-b), so I keep no lasting log of "
    "the requests the model receives. That leaves me no way to tell whether "
    "real inputs look anything like the 2005 training data. And the only "
    "way to roll back is to revert a commit and let the pipeline redeploy."
)
h("4.2 Limitations of the CI/CD pipeline", level=2)
body(
    "My pipeline is really CI/CD for code, not for models. It checks that "
    "the committed model still behaves the way it did in Module 8, but it "
    "does not retrain, compare a new candidate against the current model, "
    "or record which data and hyperparameters produced the weights. Google "
    "Cloud (n.d.) describes MLOps maturity in levels, and in those terms "
    "(and in the wider survey by Kreuzberger et al., 2023) this is roughly "
    "level 0 with automated deployment bolted on: training is manual and "
    "offline, and only serving is automated. The model artifacts are stored "
    "directly in the Git repository, which works at 29 KB but would not "
    "work for a large model. There is no staging environment either, so the "
    "first time a change runs on Hugging Face hardware, it is already live. "
    "Nothing checks after the sync that the Space actually started and can "
    "answer a request. And as the label bug showed, the tests stop at the "
    "edge of the Python code."
)
h("4.3 Improvements", level=2)
body(
    "A number of improvements would be cheap and would stay within the free "
    "tier:"
)
for item in [
    "Add a post-deploy job that waits for the Space to rebuild, calls its "
    "/predict endpoint with the gradio_client library, and fails the run if "
    "the live app does not give the right answer.",
    "Add a browser test (Playwright, for example) that loads the page and "
    "clicks an example. That would have caught the duplicate-label bug.",
    "Deploy pull requests to a second, private staging Space, and only "
    "promote changes to the public Space once they are merged.",
    "Move the weights and scaler into a versioned Hugging Face model "
    "repository with a model card (Mitchell et al., 2019), so the Space "
    "loads a specific model revision and the model can be versioned "
    "separately from the UI code.",
    "Add a metric gate: a CI step that scores the committed model on a "
    "held-out set and blocks the deployment if F1 drops below the current "
    "0.513.",
]:
    doc.add_paragraph(item, style="List Bullet")
body(
    "On the modelling side, Module 8 showed that the decision threshold and "
    "class weighting matter more than the hyperparameters. A lender who "
    "mainly cares about catching defaulters would be better served by "
    "exposing the threshold as a documented setting, or by deploying the "
    "cost-sensitive network instead."
)
h("4.4 When more expensive deployment is justified", level=2)
body(
    "For a class project with a tiny model and the occasional visitor, "
    "paying for more would just be wasted money. That changes in a few "
    "specific situations:"
)
for item in [
    "Real credit decisions. If the model were making them, regulation and "
    "risk would drive the design. A lender needs authentication, encryption "
    "of personal financial data, audit logs of every prediction, "
    "explanations for adverse decisions, and a documented model-risk "
    "process, and a public demo Space offers none of these. That would "
    "justify a private deployment on a managed platform such as SageMaker, "
    "Azure Machine Learning or Vertex AI, or a containerized service on "
    "Kubernetes inside the company's own network.",
    "Real traffic. Scoring every account in a nightly batch, or scoring "
    "applications live at checkout, shifts the requirements to throughput, "
    "guaranteed latency and uptime. That justifies autoscaling, multiple "
    "replicas behind a load balancer, and paid always-on hardware with no "
    "cold starts.",
    "Regular retraining. Customer behaviour drifts, and credit data "
    "certainly does over economic cycles. If the model had to be retrained "
    "on a schedule, a full MLOps stack would pay for itself: a feature "
    "store, experiment tracking, a model registry, automated retraining, "
    "canary or shadow deployments, and drift monitoring (Google Cloud, n.d.; "
    "Kreuzberger et al., 2023).",
    "A much larger model. A deep network over transaction sequences, or a "
    "language model reading application text, would need GPU hardware just "
    "to keep latency acceptable, and Spaces lists paid GPU tiers for "
    "exactly that (Hugging Face, n.d.-c).",
]:
    doc.add_paragraph(item, style="List Bullet")
body(
    "In each of these cases the justification is a concrete requirement "
    "(compliance, scale, how often the model changes, or compute), not "
    "complexity for its own sake. Without one of them, the simplest "
    "deployment that works is the right one."
)

# ------------------------------------------------------------- 5. Conclusion
h("5. Conclusions")
body(
    "This assignment took my tuned Module 8 network from a weights file on "
    "my laptop to a public web app, with a tested, automated path from "
    "commit to production. The final setup is a Gradio Space on the free "
    "CPU tier, a 15-test pytest suite that pins the deployed model to its "
    "Module 8 predictions, and a two-job GitHub Actions workflow that only "
    "syncs to Hugging Face once the tests pass. The most useful lessons were "
    "not really about Hugging Face. A model is its weights plus its "
    "preprocessing, and the two have to be saved and versioned together. On "
    "small hardware, the software environment can cause more trouble than "
    "the model itself. Passing tests only prove what they actually check. "
    "And putting a model in front of people exposes its weaknesses in a way "
    "a metrics table does not. This deployment is the right size for a "
    "course project: it is free, reproducible and easy to change. It is not "
    "production infrastructure for a lender, though. The gap between the "
    "two (security, monitoring, retraining and scale) is exactly what "
    "heavier, more expensive MLOps tooling is for, when the stakes justify "
    "it."
)

# ------------------------------------------------------------- References
h("References")
for ref in [
    "Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., & Zou, J. (2019). "
    "Gradio: Hassle-free sharing and testing of ML models in the wild "
    "[Preprint]. arXiv. https://arxiv.org/abs/1906.02569",
    "Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML "
    "test score: A rubric for ML production readiness and technical debt "
    "reduction. 2017 IEEE International Conference on Big Data. IEEE. "
    "https://doi.org/10.1109/BigData.2017.8258038",
    "GitHub. (n.d.-a). Using jobs in a workflow. GitHub Docs. "
    "https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-jobs",
    "GitHub. (n.d.-b). Using secrets in GitHub Actions. GitHub Docs. "
    "https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets",
    "Google Cloud. (n.d.). MLOps: Continuous delivery and automation "
    "pipelines in machine learning. Cloud Architecture Center. "
    "https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning",
    "Hugging Face. (n.d.-a). Managing Spaces with GitHub Actions. Hugging Face "
    "Hub documentation. https://huggingface.co/docs/hub/en/spaces-github-actions",
    "Hugging Face. (n.d.-b). Spaces configuration reference. Hugging Face Hub "
    "documentation. https://huggingface.co/docs/hub/en/spaces-config-reference",
    "Hugging Face. (n.d.-c). Spaces overview. Hugging Face Hub documentation. "
    "https://huggingface.co/docs/hub/en/spaces-overview",
    "Hugging Face. (n.d.-d). Gradio Spaces. Hugging Face Hub documentation. "
    "https://huggingface.co/docs/hub/en/spaces-sdks-gradio",
    "Hugging Face. (n.d.-e). GitHub Actions. Hugging Face Hub documentation. "
    "https://huggingface.co/docs/hub/en/repositories-github-actions",
    "Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine learning "
    "operations (MLOps): Overview, definition, and architecture. IEEE Access, "
    "11. https://doi.org/10.1109/ACCESS.2023.3262138",
    "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., "
    "Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model "
    "cards for model reporting. Proceedings of the Conference on Fairness, "
    "Accountability, and Transparency. ACM. "
    "https://doi.org/10.1145/3287560.3287596",
    "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … "
    "Chintala, S. (2019). PyTorch: An imperative style, high-performance deep "
    "learning library. Advances in Neural Information Processing Systems, 32. "
    "https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html",
    "Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., "
    "Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden "
    "technical debt in machine learning systems. Advances in Neural "
    "Information Processing Systems, 28. "
    "https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html",
    "Yeh, I.-C. (2016). Default of credit card clients [Data set]. UCI "
    "Machine Learning Repository. https://doi.org/10.24432/C55S3H",
]:
    reference(ref)

out = os.path.join(HERE, "AAI6610_Module10_DylanMiller.docx")
doc.save(out)

# word count of the body (sections 1-5, excluding title block and references)
words, counting = 0, False
for p in doc.paragraphs:
    if p.text.startswith("1. Introduction"):
        counting = True
    if p.text == "References":
        counting = False
    if counting:
        words += len(p.text.split())
print("Saved", out, "| body word count:", words)
