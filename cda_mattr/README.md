# Constructional Diversity Analyzer — English UI (Streamlit)

This repository contains a Streamlit app that reproduces your original CDA logic with a fully English interface.
It supports **drag & drop** of multiple `.txt` files or **one `.zip`** containing a folder of `.txt` files (recursively).
Lexicon files have been renamed and moved to `Data/lexicons/`.

## Project Layout
```
.
├─ app.py
├─ requirements.txt
└─ Data/
   └─ lexicons/
      ├─ ditransitive_verbs.txt
      ├─ intransitive_motion_verbs.txt
      └─ predicative_complements.txt
```

## Local Run
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Streamlit Cloud
- Select this repository
- Main file path: `app.py`
- The app downloads `en_core_web_sm` if missing
