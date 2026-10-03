# DermaAssist

## Project layout

- `app.py` - Gradio web application.
- `main.py` - command-line prediction entry point.
- `predictor.py`, `gemini_assess.py`, and `triage.py` - application pipeline modules.
- `models/best_model.pth` - ResNet18 model weights.
- `examples/` - local example images used by the web app.
- `data/` - source dataset assets and attribution files.

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

Set `GEMINI_API_KEY` in `.env` before using the Gemini assessment step. The
application is a research prototype, not a medical device or diagnosis.