
from __future__ import division
from operator import itemgetter
import sys, os, io, csv, traceback
import numpy
import more_itertools
from collections import Counter
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

import streamlit as st

# NLTK
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# spaCy
import spacy
from spacy.cli import download as spacy_download

# ========================
# Constants / Limits
# ========================
MAX_FILES = 2000
CONSTRUCTION_TYPES = [
    "there", "attributive", "simple_intransitive", "intransitive_motion",
    "intransitive_resultative", "passive", "simple_transitive", "caused_motion",
    "ditransitive", "transitive_resultative", "phrasal_verb", "n/a"
]
CSV_HEADERS = [
    "file_name", "total_num_sentences", "total_num_words",
    "token_frequency_of_constructions", "type_frequency_of_constructions",
    "log_transformed_type_frequency_of_constructions",
    "MATTR"
] + CONSTRUCTION_TYPES \
  + [f"{ct}_prop" for ct in CONSTRUCTION_TYPES] \
  + [f"arcsine_transformed_{ct}_prop" for ct in CONSTRUCTION_TYPES]

# ========================
# Resource paths
# ========================
def resource_candidates(relative_path: str):
    rel = relative_path.replace("\\", "/").lstrip("/")
    here = Path(__file__).resolve().parent
    yield (here / rel).as_posix()
    yield (Path(os.getcwd()).resolve() / rel).as_posix()
    yield (here.parent / rel).as_posix()

def resource_path(relative_path: str):
    for cand in resource_candidates(relative_path):
        if os.path.isfile(cand):
            return cand
    return (Path(__file__).resolve().parent / relative_path).as_posix()

def load_lexicon(filepath):
    p = resource_path(filepath)
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.warning(f"Lexicon file not found: {p}")
        return []
    except Exception as e:
        st.warning(f"Error loading lexicon: {p} — {e}")
        return []

@st.cache_resource(show_spinner=False)
def load_spacy(model_name: str = "en_core_web_sm", use_blank: bool = False):
    if use_blank:
        return spacy.blank("en")
    try:
        return spacy.load(model_name)
    except Exception as e:
        # Second attempt: download once
        try:
            spacy_download(model_name)
            return spacy.load(model_name)
        except Exception as e2:
            raise RuntimeError(f"Failed to load spaCy model '{model_name}': {e2}")

# ========================
# Core logic
# ========================
def classify_construction(vac_string, verb, vac_dep_list, vac_dep_without_verb,
                        vac_lem_dep_list, vac_pos_dep_list, token, ditransitive_verb, 
                        pred_intransitive_motion):
    if "_" not in vac_string:
        if verb not in ["be", "have", "do"]:
            return "simple_intransitive"
        return None

    if vac_dep_without_verb.startswith(("npadvmod_verb", "advmod_verb", "npadvmod_punct_verb", "advmod_punct_verb")):
        return None

    if "auxpass" in vac_dep_list or ("get-verb" in vac_lem_dep_list and "acomp" in vac_dep_list):
        return "passive"

    if "dobj" not in vac_dep_list:
        if "be-verb" in vac_lem_dep_list:
            if "prep" in vac_dep_list:
                prep_index = vac_dep_list.index("prep")
                be_index = vac_lem_dep_list.index("be-verb")
                if be_index + 1 == prep_index:
                    return "attributive"
                elif "there-expl" in vac_lem_dep_list:
                    return "there"
                else:
                    return "attributive"
            elif "there-expl" in vac_lem_dep_list:
                return "there"
            elif any(x in vac_dep_list for x in ["attr", "acomp", "advmod"]):
                return "attributive"
        elif "prep" in vac_dep_list:
            if verb in pred_intransitive_motion:
                return "intransitive_motion"
            return "simple_intransitive"
        elif any(x in vac_dep_list for x in ["acomp", "advcl", "advmod"]):
            if "verb_acomp" in vac_dep_without_verb or "verb_oprd" in vac_dep_without_verb:
                if verb in ["become", "look"]:
                    return "simple_intransitive"
                return "intransitive_resultative"
            return "simple_intransitive"
        elif "ccomp" in vac_dep_list or "xcomp" in vac_dep_list:
            if "be-verb_xcomp" in vac_dep_list:
                return "attributive"
            elif any(f"{v}-verb_xcomp" in vac_dep_list for v in ["become", "appear", "seem"]):
                return "simple_intransitive"
            return "simple_transitive"
        else:
            return "simple_intransitive"
    else:
        if any(p in vac_dep_without_verb for p in ["dobj_acomp", "dobj_oprd"]) or "ADJ-ccomp" in vac_pos_dep_list:
            if token.lemma_ in ["keep", "drive"] or token.lemma_ in ditransitive_verb:
                return "transitive_resultative"
            return "transitive_resultative"
        elif "prt" in vac_dep_list:
            return "phrasal_verb"
        elif "dative" in vac_dep_list:
            if token.lemma_ in ditransitive_verb:
                if any(x in vac_lem_dep_list for x in ["to-dative", "for-dative"]):
                    return "caused_motion"
                return "ditransitive"
            return "caused_motion"
        elif any(p in vac_lem_dep_list for p in ["to-prep", "for-prep", "onto-prep", "into-prep", "from-prep"]):
            if token.lemma_ in ditransitive_verb:
                return "caused_motion"
            return "simple_transitive"
        else:
            return "simple_transitive"
    return None

def calculate_mattr(construction_list, window_size=11):
    if len(construction_list) < window_size:
        return f"Text must include at least {window_size} constructions."
    elif len(construction_list) == 1:
        return 1.0
    else:
        windows = list(more_itertools.windowed(construction_list, n=window_size, step=1))
        scores = [len(set(w))/window_size for w in windows if w and None not in w]
        return sum(scores)/len(scores) if scores else 0.0

def write_results_row(writer, file_name, text_tokenized, words,
                      counter_clause, counters, mattr_value):
    row = {
        "file_name": file_name,
        "total_num_sentences": len(text_tokenized),
        "total_num_words": len(words),
        "token_frequency_of_constructions": counter_clause,
        "type_frequency_of_constructions": sum(1 for v in counters.values() if v > 0),
        "log_transformed_type_frequency_of_constructions": numpy.log10(
            float(sum(1 for v in counters.values() if v > 0)) + 1
        ),
        "MATTR": mattr_value if isinstance(mattr_value, (int, float)) else str(mattr_value)
    }
    for ct in CONSTRUCTION_TYPES:
        count = counters.get(ct, 0)
        prop = count / counter_clause if counter_clause > 0 else 0.0
        row[ct] = count
        row[f"{ct}_prop"] = prop
        row[f"arcsine_transformed_{ct}_prop"] = numpy.arcsin(prop)
    writer.writerow(row)

# ========================
# Streamlit UI + Debug
# ========================
st.set_page_config(page_title="Constructional Diversity Analyzer", layout="wide")
st.title("Constructional Diversity Analyzer — English (Debug Build)")

st.write("Upload .txt files or a .zip of a folder. Click **Run analysis**. This build includes debug output to help diagnose issues.")

mode = st.radio("Choose upload method", ["Select multiple .txt files", "Upload folder (.zip)"], horizontal=True)

# Load lexicons
ditransitive_verbs = load_lexicon('Data/lexicons/ditransitive_verbs.txt')
predicative_complements = load_lexicon('Data/lexicons/predicative_complements.txt')
intransitive_motion_verbs = load_lexicon('Data/lexicons/intransitive_motion_verbs.txt')

window_size = st.number_input("MATTR window size", min_value=2, max_value=101, value=11, step=1, help="Default is 11")
use_blank = st.checkbox("Use blank English pipeline (no download)", value=False, help="If checked, uses spacy.blank('en') to avoid downloading a model.")

if mode == "Select multiple .txt files":
    uploaded_files = st.file_uploader("Drop or select .txt files to analyze", type=["txt"], accept_multiple_files=True)
    uploaded_zip = None
else:
    uploaded_zip = st.file_uploader("Upload a .zip containing your .txt folder", type=["zip"], accept_multiple_files=False)
    uploaded_files = None

run_btn = st.button("Run analysis")

def normalize_text_filename(name: str) -> str:
    return name[:300]

if run_btn:
    try:
        items = []
        st.info("Run button clicked. Validating inputs...")

        if mode == "Select multiple .txt files":
            if not uploaded_files:
                st.error("No .txt files were uploaded.")
                st.stop()
            for uf in uploaded_files:
                try:
                    raw = uf.read()
                    uf.seek(0)
                    text = raw.decode("utf-8-sig", errors="ignore")
                except Exception as e:
                    st.warning(f"Failed to read file: {uf.name} — {e}")
                    continue
                items.append((normalize_text_filename(uf.name), text))
        else:
            if not uploaded_zip:
                st.error("No ZIP file was uploaded.")
                st.stop()
            import zipfile
            with zipfile.ZipFile(uploaded_zip) as zf:
                txt_names = sorted([n for n in zf.namelist() if n.lower().endswith('.txt') and not n.endswith('/')])
                if not txt_names:
                    st.error("No .txt files were found inside the ZIP.")
                    st.stop()
                for name in txt_names:
                    with zf.open(name) as f:
                        raw = f.read()
                    text = raw.decode('utf-8-sig', errors='ignore')
                    items.append((normalize_text_filename(name), text))

        if not items:
            st.error("There is nothing to process after reading inputs.")
            st.stop()

        if len(items) > MAX_FILES:
            st.warning(f"Only the first {MAX_FILES} files will be processed.")
            items = items[:MAX_FILES]

        with st.spinner("Loading NLP model..."):
            nlp = load_spacy("en_core_web_sm", use_blank=use_blank)
            st.success(f"NLP model ready: {nlp.meta.get('name', 'blank')} {nlp.meta.get('version', '')}")

        progress = st.progress(0)
        status = st.empty()
        output_buffer = io.StringIO()
        writer = csv.DictWriter(output_buffer, fieldnames=CSV_HEADERS)
        writer.writeheader()
        per_file_summary = []

        for idx, (name, text) in enumerate(items, start=1):
            status.text(f"Processing {name} ({idx}/{len(items)})")

            counters = {k:0 for k in CONSTRUCTION_TYPES}
            counter_clause = 0
            construction_types = []
            words = text.split()
            text_tokenized = sent_tokenize(text)

            for sent in text_tokenized:
                doc = nlp(sent)
                for token in doc:
                    if token.dep_ == "ROOT" or (token.pos_ == "VERB" and token.pos_ != "AUX"):
                        vac_dep = [[f"{token.lemma_}-verb", token.i]]
                        vac_dep_wo_verb = [["verb", token.i]]
                        vac_lem_dep = [[f"{token.lemma_}-verb", token.i]]
                        vac_pos_dep = [[f"{token.pos_}-{token.dep_}", token.i]]
                        vac_l = [[token.text, token.i]]

                        for child in token.children:
                            vac_dep.append([child.dep_, child.i])
                            vac_dep_wo_verb.append([child.dep_, child.i])
                            vac_lem_dep.append([f"{child.lemma_}-{child.dep_}", child.i])
                            vac_pos_dep.append([f"{child.pos_}-{child.dep_}", child.i])
                            vac_l.append([child.text, child.i])

                        for lst in [vac_dep, vac_dep_wo_verb, vac_lem_dep, vac_pos_dep, vac_l]:
                            lst.sort(key=lambda x: x[1])

                        vac_dep_list = [x[0] for x in vac_dep]
                        vac_dep_wo_verb_list = [x[0] for x in vac_dep_wo_verb]
                        vac_lem_dep_list = [x[0] for x in vac_lem_dep]
                        vac_pos_dep_list = [x[0] for x in vac_pos_dep]
                        vac_l_final = [x[0] for x in vac_l]

                        vac_dep_without_verb = "_".join(vac_dep_wo_verb_list)
                        vac_lemma = "_".join(vac_lem_dep_list)
                        vac_string = "_".join(vac_l_final)

                        construction = classify_construction(
                            vac_string, token.lemma_, vac_dep_list,
                            vac_dep_without_verb, vac_lem_dep_list,
                            vac_pos_dep_list, token, ditransitive_verbs,
                            intransitive_motion_verbs
                        )

                        if construction:
                            counter_clause += 1
                            if construction not in CONSTRUCTION_TYPES:
                                construction = 'n/a'
                            construction_types.append(construction)

            counts = Counter(construction_types)
            counters.update(counts)

            mattr_value = calculate_mattr(construction_types, window_size=window_size)
            write_results_row(
                writer, name, text_tokenized, words,
                counter_clause, counters, mattr_value
            )

            per_file_summary.append({
                "file_name": name,
                "num_sentences": len(text_tokenized),
                "num_words": len(words),
                "num_clauses": counter_clause,
                "MATTR": mattr_value if isinstance(mattr_value, (int, float)) else None
            })

            progress.progress(idx / len(items))

        status.text("Done.")
        csv_bytes = output_buffer.getvalue().encode("utf-8")
        st.download_button(
            label="Download constructional_diversity.csv",
            data=csv_bytes,
            file_name="constructional_diversity.csv",
            mime="text/csv",
        )

        st.subheader("Summary")
        if per_file_summary:
            import pandas as pd
            df = pd.DataFrame(per_file_summary)
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error("An unexpected error occurred during processing.")
        st.exception(e)
        st.text("Traceback:")
        st.code(traceback.format_exc())
