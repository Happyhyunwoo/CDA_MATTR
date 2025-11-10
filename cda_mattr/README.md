# CDA MATTR (Streamlit) — Folder/ZIP Upload

This app analyzes English texts with the same logic as the original script, now supporting:
- Multiple `.txt` file uploads, **or**
- A **single ZIP file** containing a folder of `.txt` files (processed recursively)

## Usage
1) Upload either multiple `.txt` files **or** one `.zip` that contains your folder of `.txt` files.
2) Click **Analyze** to get `constructional_diversity.csv`.

## Notes
- Max 2,000 `.txt` files will be processed (sorted by path).
- ZIP support allows nested subfolders; only `.txt` are included.
- Put the dictionaries at:
```
Data/dictionaries/ditransitive_verb.txt
Data/dictionaries/intransitive_motion.txt
Data/dictionaries/predicative_complement.txt
```
- If `app.py` is in a subfolder, keep `Data/` beside that subfolder.
