# DermaAssist

## Project layout

- `app.py` - Gradio web application.
- `main.py` - command-line prediction entry point.
- `predictor.py`, `gemini_assess.py`, and `triage.py` - application pipeline modules.
- `models/best_model.pth` - ResNet18 model weights.
- `examples/` - local example images used by the web app.
- `data/` - source dataset assets and attribution files.
- `scripts/` - optional cache and evaluation utilities.
- `tests/` - automated triage tests.

## Run locally

Install the dependencies, then start the web app:

```bash
pip install -r requirements.txt
python app.py
```

To run one image through the ResNet18 predictor:

```bash
python main.py examples/test1.png
```

Copy `.env.example` to `.env` and replace the placeholder with your Gemini API
key before using the Gemini assessment step:

```bash
cp .env.example .env
```

Keep `.env` local and never commit it. The application is a research
prototype, not a medical device or diagnosis.

## Test locally

From the repository root, install dependencies and run the automated tests:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Run the model-only smoke test with the bundled image:

```bash
python main.py examples/test1.png
```

Run the web app:

```bash
python app.py
```

Open the local URL printed by Gradio, upload an image, answer the five
questions, and click **Assess**. Without `GEMINI_API_KEY`, the app should still
show the ResNet18 result and the triage floor; Gemini's text will be unavailable.