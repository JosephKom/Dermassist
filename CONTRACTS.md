# Module contracts

`app.py` calls three modules that are built in parallel. They must stick to the
signatures below so that they fit together without changes. If you need to change
one, open a PR that edits this file first and tag everyone.

| Module | Owner | Function |
| --- | --- | --- |
| `predictor.py` | Tri | `predict(image)` |
| `gemini_assess.py` | Joseph | `assess(image, probabilities, answers)` |
| `triage.py` | Antonio | `floor(level, probabilities, answers)` |
| `app.py` | Khang | Calls the three in that order |

## Shared values

### Urgency levels

These are exact strings. Use them as they are everywhere, including in the Gemini JSON schema enum.

| Level | Meaning |
| --- | --- |
| `"Cannot assess"` | The image is not a usable close-up of a skin lesion. |
| `"Prompt review"` | See a doctor soon. |
| `"Routine review"` | Book a skin check. |
| `"Nothing flagged"` | Nothing was flagged. This is never an all-clear. |

Order from lowest to highest: `Nothing flagged` < `Routine review` < `Prompt review`.
`Cannot assess` is outside that order.

### Class codes

There are seven HAM10000 codes, in lowercase: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

### Answers

```python
answers = {"grown": bool, "changed": bool, "bled": bool, "itched": bool, "hurt": bool}
```

## Functions

### `predict(image) -> dict[str, float]` (`predictor.py`)

- `image` is a `PIL.Image.Image` in RGB.
- Returns all seven class codes mapped to probabilities that sum to 1.
- Loads the model once, when the module is imported, not on every call.
- Model weights: `skin-disease-detection-main1/best_model.pth`.

### `assess(image, probabilities, answers) -> dict | None` (`gemini_assess.py`)

- `probabilities` is the output of `predict`, and `answers` is shown above.
- Returns `None`, or raises, when the call fails, is blocked or has no API key.
  `app.py` handles both cases by showing the ResNet18 result only.
- When it succeeds, it returns a dict with these keys:

| Key | Type | Notes |
| --- | --- | --- |
| `is_skin_lesion` | `bool` | The scope check. |
| `image_issue` | `str \| None` | Why the image cannot be assessed, if it can't. |
| `diagnosis` | `str` | Gemini's own diagnosis, which may differ from ResNet18's. |
| `confidence` | `str` | `"low"`, `"medium"` or `"high"`. |
| `visible_features` | `str` | Colour, border and symmetry. |
| `urgency_level` | `str` | One of the four levels. |
| `urgency_reasons` | `list[str]` | Why it chose that level. |
| `treatment_info` | `str` | General information only, with no doses or product names. |

- Reads the API key from the `GEMINI_API_KEY` environment variable. `app.py`
  loads `.env` before importing this module, and `.env.example` lists the variable.
- Cached demo replies for the example images are served from inside `assess`,
  so `app.py` doesn't need to know about them.

### `floor(level, probabilities, answers) -> tuple[str, str | None]` (`triage.py`)

- `level` is Gemini's `urgency_level`, or `None` when Gemini failed. `None` means
  "start from `Nothing flagged`".
- Returns `(final_level, rule_fired)`. `rule_fired` is a short sentence naming the
  rule that raised the level, or `None` if nothing raised it.
- It may raise the level but never lowers it.
- Cancer score is `probabilities["mel"] + probabilities["bcc"]`, and the cutoff is read from `thresholds.json`.

## Open question

- When Gemini returns `is_skin_lesion: False`, `app.py` currently shows
  `Cannot assess` and skips `floor`. Should a red-flag answer still escalate in
  that case? (Antonio to decide.)
