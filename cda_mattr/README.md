# Constructional Diversity Analyzer (Streamlit)

This repo contains a Streamlit app that reproduces the original CDA logic and runs on Streamlit Cloud.

## Repository Layout
```
.
├─ app.py
├─ requirements.txt
└─ Data/
   └─ dictionaries/
      ├─ ditransitive_verb.txt
      ├─ intransitive_motion.txt
      └─ predicative_complement.txt
```

> If you place `app.py` inside a subfolder (e.g., `cda_mattr/app.py`), **keep `Data/` beside that `app.py` file**.  
> The app now resolves paths relative to the script file location, so nested folders are okay.

## Run Locally
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Deploy on Streamlit Cloud
- Repository: select this repo
- Main file path: `app.py` (or `subdir/app.py` if you nested it)
- The app will download `en_core_web_sm` if missing.