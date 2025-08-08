# Debt Collection Compliance Checker (Streamlit)

Analyze conversations between debt collection agents and borrowers for:
- Profanity Detection (Q1)
- Privacy & Compliance Violation (Q2)
- Call Quality Metrics: Overtalk % and Silence % (Q3)

This app supports two approaches:
- Pattern Matching (regex-based)
- Machine Learning (constrained-vocabulary Logistic Regression with lexicon fallback)

It also includes Comparative Analysis for both Q1 and Q2, showing both approaches side-by-side with a recommendation.

## Project Structure

- `streamlit_app.py`: Streamlit UI and orchestration
- `src/`
  - `loaders.py`: Load YAML/JSON calls
  - `textnorm.py`: Text normalization helpers
  - `regex_rules.py`: Regex detectors for profanity and privacy/compliance
  - `ml_models.py`: ML models aligned to vocabularies, plus lexicon fallback for stability
  - `metrics.py`: Overtalk and silence computations
  - `visualizations.py`: Charts for Q3
  - `utils.py`: Speaker normalization, call-id extraction
- `data/samples/`: Example call JSONs with random call IDs
- `report/technical_report.md`: Recommendations and analysis notes
- `requirements.txt`: Python dependencies

## Run Locally

1. Python 3.10+ recommended
2. Create and activate a virtual environment
3. Install dependencies:
\`\`\`
pip install -r requirements.txt
\`\`\`
4. Launch the app:
\`\`\`
streamlit run streamlit_app.py
\`\`\`

5. In the UI:
   - Choose Approach (Pattern Matching or Machine Learning)
   - Choose Entity (Profanity Detection or Privacy and Compliance Violation)
   - Upload a call file (YAML or JSON). Sample JSON files are in `data/samples/`.

The app shows:
- A Yes/No flag for the selected entity using your chosen approach
- Comparative Analysis panel showing both approaches and a recommendation
- Visualizations of overtalk % and silence % for the call (Q3)

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and deploy the repo.
3. Set Python version to 3.10+ and ensure `requirements.txt` is detected.
4. No secrets needed (LLM approach is not implemented).
