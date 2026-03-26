# RunRight UAE — Analytics Platform

## What's Included

| File | Purpose |
|------|---------|
| `app.py` | Streamlit analytics app (8 pages) |
| `requirements.txt` | Python dependencies |
| `RunRight_Scorer.html` | Standalone HTML scoring tool (no server needed) |
| `RunRight_UAE_Survey_Raw.csv` | Raw survey data (2,000 respondents) |
| `RunRight_UAE_Survey_Encoded.csv` | Encoded/feature-engineered data |
| `enriched.csv` | Scored & segmented full dataset |
| `arm_rules.csv` | 61 association rules |
| `precomputed.json` | Pre-computed chart data |
| `models/` | Trained sklearn models (pkl files) |

## Deploy to Streamlit Cloud

1. Push all files to a GitHub repo
2. Go to share.streamlit.io → New app → Select repo
3. Set main file path: `app.py`
4. Deploy — all dependencies install automatically from `requirements.txt`

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## HTML Scorer

Open `RunRight_Scorer.html` in any browser — no installation needed.
- Drag & drop a CSV of new customers
- Models run 100% in-browser (50-tree RF embedded as JSON, ~1.8MB)
- Download scored results as CSV

## Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| RF Classifier | AUC-ROC | 0.594 |
| RF Classifier | Accuracy | 0.620 |
| RF Regressor | R² | 0.967 |
| RF Regressor | MAE | AED 109 |
| K-Means | Silhouette | 0.078 |
| ARM | Rules | 61 |

## Customer Tiers

| Tier | Personas | Focus |
|------|----------|-------|
| Tier 1 | Trail & Ultra Specialist, Serious Age-Grouper | Invest heavily — highest LTV |
| Tier 2 | Wellness Professional, Social Community Runner | Scale — growth engine |
| Tier 3 | Aspirational Beginner, Casual Lifestyle Runner | Freemium — nurture & upgrade |
