# Dermassist

DermaAssist is a prototype that looks at a close-up photo of a skin spot and suggests how urgently it should be seen by a doctor.

Two models give an opinion. Gemini gives a diagnosis, an urgency level and general treatment information. A ResNet18 classifier trained on HAM10000 shows its top three guesses alongside. A small set of safety rules can raise Gemini's urgency level but never lowers it.

> **Research prototype. Not a medical device and not a diagnosis.** It can be wrong, and "Nothing flagged" is never an all-clear. Always see a doctor about a spot that worries you.

## Run the web app

You need Python 3 and a Gemini API key (get one free at https://aistudio.google.com/apikey).

1. Create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Add your API key:

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and set `GEMINI_API_KEY=your-key`. `.env` is ignored by git, so the key is never committed.

3. Start the app:

   ```bash
   python app.py
   ```

4. Open http://127.0.0.1:7860 in your browser.

To use the app, upload a close-up photo of a skin spot, or pick one of the example images. Answer the five yes/no questions and click **Assess**.

Without an API key, or if the Gemini call fails, the app still runs. It shows the ResNet18 result and the safety rules' urgency level only.

### Privacy

- The app runs on your own machine and is only reachable from it.
- Uploaded images are sent to Google Gemini for analysis.
- Nothing is saved: the app does not store uploads or log your answers.

### What the result shows

| Part | Meaning |
| --- | --- |
| Urgency level | **Prompt review** (see a doctor soon), **Routine review** (book a skin check), **Nothing flagged** (not an all-clear), or **Cannot assess** (the image isn't a usable close-up of a skin spot). |
| Reasons | Why that level was chosen. A "Raised by safety rule" line means a rule raised Gemini's level. |
| Gemini's opinion | Diagnosis, confidence and visible features. |
| Image model (ResNet18) | Its top three of seven conditions. A warning appears when it disagrees with Gemini. |
| Treatment | How the condition is usually treated, in general terms only. |

<!-- Tri: add the results caveats here (validation set shares lesions with training, so the 77% is optimistic; urgency cutoffs in thresholds.json are set by hand, not tuned on data). -->

## For developers

See [CONTRACTS.md](CONTRACTS.md) for the interfaces between `app.py`, `predictor.py`, `gemini_assess.py` and `triage.py`.
