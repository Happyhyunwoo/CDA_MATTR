from __future__ import division
from operator import itemgetter
import sys
import os
import io
import time
import numpy
import csv
import more_itertools
from collections import Counter
from itertools import islice

import streamlit as st

# NLTK
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# spaCy
import spacy
from spacy.lang.en import English
from spacy.util import is_package
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
# Helpers
# ========================
def ensure_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        # Try to download inside the app runtime (e.g., Streamlit Cloud)
        try:
            spacy_download(model_name)
            return spacy.load(model_name)
        except Exception as e:
            st.error(f"spaCy 모델 '{model_name}'을(를) 설치할 수 없습니다: {e}")
            st.stop()

def resource_path(relative_path):
    base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_dictionary(filepath):
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.warning(f"사전 파일을 찾을 수 없습니다: {filepath}")
        return []
    except Exception as e:
        st.warning(f"사전 로딩 중 오류: {filepath} — {e}")
        return []

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
# Streamlit UI
# ========================
st.set_page_config(page_title="Constructional Diversity Analyzer", layout="wide")
st.title("Constructional Diversity Analyzer (Streamlit)")

st.write("원본 코드와 동일한 규칙으로 영어 문장을 분석합니다. 최대 2,000개의 .txt 파일을 업로드할 수 있습니다. 결과는 CSV로 내려받을 수 있습니다.")

# Load dictionaries (from repo path)
ditransitive_verb = load_dictionary(resource_path('Data/dictionaries/ditransitive_verb.txt'))
pred_comp = load_dictionary(resource_path('Data/dictionaries/predicative_complement.txt'))
pred_intransitive_motion = load_dictionary(resource_path('Data/dictionaries/intransitive_motion.txt'))

# Load spaCy model
nlp = ensure_spacy_model("en_core_web_sm")

uploaded_files = st.file_uploader("분석할 .txt 파일들을 업로드하세요 (여러 개 선택 가능)", type=["txt"], accept_multiple_files=True)

window_size = st.number_input("MATTR 윈도 크기", min_value=2, max_value=101, value=11, step=1, help="기본값 11")

run_btn = st.button("분석 시작")

if run_btn:
    if not uploaded_files:
        st.warning("먼저 텍스트 파일을 업로드하세요.")
        st.stop()

    if len(uploaded_files) > MAX_FILES:
        st.warning(f"파일은 최대 {MAX_FILES}개까지만 분석합니다. 업로드한 앞에서부터 {MAX_FILES}개만 처리합니다.")
        files_to_process = uploaded_files[:MAX_FILES]
    else:
        files_to_process = uploaded_files

    progress = st.progress(0)
    status = st.empty()

    output_buffer = io.StringIO()
    writer = csv.DictWriter(output_buffer, fieldnames=CSV_HEADERS)
    writer.writeheader()

    per_file_summary = []

    for idx, uf in enumerate(files_to_process, start=1):
        file_name = uf.name
        status.text(f"Processing {file_name} ({idx}/{len(files_to_process)})")
        try:
            raw = uf.read()
            # Reset file pointer in case of re-reads
            uf.seek(0)
            text = raw.decode("utf-8-sig", errors="ignore")
        except Exception as e:
            st.warning(f"파일 읽기 오류: {file_name} — {e}")
            progress.progress(idx / len(files_to_process))
            continue

        counters = {
            'there': 0, 'attributive': 0, 'simple_intransitive': 0,
            'intransitive_motion': 0, 'intransitive_resultative': 0,
            'passive': 0, 'simple_transitive': 0, 'caused_motion': 0,
            'ditransitive': 0, 'transitive_resultative': 0,
            'phrasal_verb': 0, 'n/a': 0
        }

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
                        lst.sort(key=itemgetter(1))

                    vac_dep_list = [x[0] for x in vac_dep]
                    vac_dep_wo_verb_list = [x[0] for x in vac_dep_wo_verb]
                    vac_lem_dep_list = [x[0] for x in vac_lem_dep]
                    vac_pos_dep_list = [x[0] for x in vac_pos_dep]
                    vac_l_final = [x[0] for x in vac_l]

                    vac_dep_string = "_".join(vac_dep_list)
                    vac_dep_without_verb = "_".join(vac_dep_wo_verb_list)
                    vac_lemma = "_".join(vac_lem_dep_list)
                    vac_string = "_".join(vac_l_final)

                    construction = classify_construction(
                        vac_string, token.lemma_, vac_dep_list,
                        vac_dep_without_verb, vac_lem_dep_list,
                        vac_pos_dep_list, token, ditransitive_verb,
                        pred_intransitive_motion
                    )

                    if construction:
                        counter_clause += 1
                        counters[construction] = counters.get(construction, 0) + 1
                        construction_types.append(construction)

        mattr_value = calculate_mattr(construction_types, window_size=window_size)
        write_results_row(
            writer, file_name, text_tokenized, words,
            counter_clause, counters, mattr_value
        )

        per_file_summary.append({
            "file_name": file_name,
            "num_sentences": len(text_tokenized),
            "num_words": len(words),
            "num_clauses": counter_clause,
            "MATTR": mattr_value if isinstance(mattr_value, (int, float)) else None
        })

        progress.progress(idx / len(files_to_process))

    status.text("완료되었습니다.")

    csv_bytes = output_buffer.getvalue().encode("utf-8")
    st.download_button(
        label="constructional_diversity.csv 다운로드",
        data=csv_bytes,
        file_name="constructional_diversity.csv",
        mime="text/csv",
    )

    st.subheader("요약")
    if per_file_summary:
        import pandas as pd
        df = pd.DataFrame(per_file_summary)
        st.dataframe(df, use_container_width=True)
