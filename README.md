# Phishing Detection with Stylometric Features & SHAP

A cross-lingual phishing email detection system combining stylometric features (POS, dependency, morphology, urgency/authority markers) with SHAP explainability.

## Dataset

- **SpaPhish**: 1,395 real emails (731 phishing, 664 legitimate), triple-annotated with persuasion labels
- **Multilingual smoke-test**: 15 synthetic rows (EN/ES/PT) for pipeline validation
- [Detailed provenance](https://github.com/yourusername/yourrepo)

## Features

- **Stylometric extraction**: POS bigrams, dependency labels, morphology tags
- **Urgency/authority markers**: Cross-lingual dictionaries (EN/ES/PT)
- **SHAP explainability**: Identify which features drive classification
- **Cross-lingual support**: spaCy models for EN, ES, PT

## Pipeline

1. `normalize_datasets.py` – Merge real + synthetic data
2. `stylometric_features.py` – Extract linguistic features
3. `shap_simple.py` – TF-IDF + SHAP analysis
4. `run_pipeline.py` – Full stylometric LightGBM pipeline

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn shap matplotlib lightgbm spacy
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download pt_core_news_sm

# Run SHAP analysis
python shap_simple.py
