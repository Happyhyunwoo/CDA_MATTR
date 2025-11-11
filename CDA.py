from __future__ import division
from operator import itemgetter
import streamlit as st
import numpy
import csv
import more_itertools
import os
import io
import zipfile

import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

import spacy
from spacy.lang.en import English

# Streamlit app configuration
st.set_page_config(page_title="Constructional Diversity Analyzer", page_icon="📝", layout="wide")

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("⚠️ Spacy model 'en_core_web_sm' not found.")
        st.info("Please install it by running: `python -m spacy download en_core_web_sm`")
        st.stop()

nlp = load_spacy_model()

# Default dictionaries (you can modify these)
DEFAULT_DITRANSITIVE_VERBS = """give
tell
show
send
offer
teach
sell
lend
bring
pass
throw
hand
pay
write
read
buy
cook
make
build
find
get
leave
owe
promise
wish""".split('\n')

DEFAULT_INTRANSITIVE_MOTION = """go
come
walk
run
move
travel
fly
drive
swim
climb
jump
fall
rise
return
arrive
enter
leave
depart
proceed""".split('\n')

DEFAULT_PREDICATIVE_COMPLEMENT = """be
become
seem
appear
remain
stay
look
sound
feel
smell
taste
prove
turn
grow
get""".split('\n')

def classify_construction(vac_string, verb, vac_dep_list, vac_dep_without_verb,
                        vac_lem_dep_list, vac_pos_dep_list, token, ditransitive_verb, 
                        pred_intransitive_motion):
    """Classify the construction type based on the dependencies and patterns"""
    
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
    """Calculate Moving Average Type-Token Ratio"""
    if len(construction_list) < window_size:
        return f"Text must include at least {window_size} constructions."
    elif len(construction_list) == 1:
        return 1.0
    else:
        windows = list(more_itertools.windowed(construction_list, n=window_size, step=1))
        scores = [len(set(w))/window_size for w in windows if w and None not in w]
        return sum(scores)/len(scores) if scores else 0.0

def process_text_file(text, file_name, ditransitive_verb, pred_intransitive_motion):
    """Process a single text file and return results"""
    words = text.split()
    
    counters = {
        'there': 0, 'attributive': 0, 'simple_intransitive': 0,
        'intransitive_motion': 0, 'intransitive_resultative': 0,
        'passive': 0, 'simple_transitive': 0, 'caused_motion': 0,
        'ditransitive': 0, 'transitive_resultative': 0,
        'phrasal_verb': 0, 'n/a': 0
    }
    
    counter_clause = 0
    construction_types = []
    clause_details = []  # For detailed view
    
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
                
                vac_dep = "_".join(vac_dep_list)
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
                    clause_details.append({
                        'sentence': sent,
                        'construction': construction,
                        'pattern': vac_lemma
                    })
    
    mattr_value = calculate_mattr(construction_types)
    
    return {
        'file_name': file_name,
        'total_sentences': len(text_tokenized),
        'total_words': len(words),
        'counter_clause': counter_clause,
        'counters': counters,
        'construction_types': construction_types,
        'mattr_value': mattr_value,
        'clause_details': clause_details
    }

def create_csv_row(result):
    """Create a CSV row from processing results"""
    construction_types = [
        "there", "attributive", "simple_intransitive", "intransitive_motion",
        "intransitive_resultative", "passive", "simple_transitive", "caused_motion",
        "ditransitive", "transitive_resultative", "phrasal_verb", "n/a"
    ]
    
    counter_clause = result['counter_clause']
    counters = result['counters']
    
    row = {
        "file_name": result['file_name'],
        "total_num_sentences": result['total_sentences'],
        "total_num_words": result['total_words'],
        "token_frequency_of_constructions": counter_clause,
        "type_frequency_of_constructions": sum(1 for v in counters.values() if v > 0),
        "log_transformed_type_frequency_of_constructions": numpy.log10(
            float(sum(1 for v in counters.values() if v > 0)) + 1
        ),
        "MATTR": result['mattr_value'] if isinstance(result['mattr_value'], (int, float)) else str(result['mattr_value'])
    }
    
    for ct in construction_types:
        count = counters.get(ct, 0)
        prop = count / counter_clause if counter_clause > 0 else 0
        
        row[ct] = count
        row[f"{ct}_prop"] = prop
        row[f"arcsine_transformed_{ct}_prop"] = numpy.arcsin(prop)
    
    return row

# Main Streamlit UI
st.title("📝 Constructional Diversity Analyzer")
st.markdown("Analyze the constructional diversity of text files using syntactic dependency parsing.")

# Sidebar for dictionary configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Dictionaries")
    use_default = st.checkbox("Use default dictionaries", value=True)
    
    if not use_default:
        st.info("Upload custom dictionary files (one word per line)")
        ditrans_file = st.file_uploader("Ditransitive Verbs", type=['txt'])
        motion_file = st.file_uploader("Intransitive Motion Verbs", type=['txt'])
        
        if ditrans_file:
            ditransitive_verb = [line.decode('utf-8').strip() for line in ditrans_file if line.strip()]
        else:
            ditransitive_verb = DEFAULT_DITRANSITIVE_VERBS
            
        if motion_file:
            pred_intransitive_motion = [line.decode('utf-8').strip() for line in motion_file if line.strip()]
        else:
            pred_intransitive_motion = DEFAULT_INTRANSITIVE_MOTION
    else:
        ditransitive_verb = DEFAULT_DITRANSITIVE_VERBS
        pred_intransitive_motion = DEFAULT_INTRANSITIVE_MOTION
    
    st.subheader("MATTR Window Size")
    mattr_window = st.slider("Window size for MATTR calculation", 5, 20, 11)

# File upload section
st.header("📁 Upload Text Files")
uploaded_files = st.file_uploader(
    "Choose one or more .txt files",
    type=['txt'],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✓ Loaded {len(uploaded_files)} file(s)")
    
    # Show file list
    with st.expander("📋 View uploaded files"):
        for file in uploaded_files:
            st.text(f"• {file.name}")
    
    # Process button
    if st.button("🚀 Analyze Constructions", type="primary"):
        all_results = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")
            
            try:
                text = uploaded_file.read().decode('utf-8-sig')
                
                result = process_text_file(
                    text,
                    uploaded_file.name,
                    ditransitive_verb,
                    pred_intransitive_motion
                )
                
                all_results.append(result)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        progress_bar.empty()
        status_text.empty()
        
        if all_results:
            st.success(f"✓ Successfully analyzed {len(all_results)} file(s)")
            
            # Create summary statistics
            st.header("📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_constructions = sum(r['counter_clause'] for r in all_results)
                st.metric("Total Constructions", total_constructions)
            with col2:
                avg_mattr = numpy.mean([r['mattr_value'] for r in all_results if isinstance(r['mattr_value'], (int, float))])
                st.metric("Average MATTR", f"{avg_mattr:.4f}")
            with col3:
                total_sentences = sum(r['total_sentences'] for r in all_results)
                st.metric("Total Sentences", total_sentences)
            with col4:
                total_words = sum(r['total_words'] for r in all_results)
                st.metric("Total Words", total_words)
            
            # Individual file results
            st.header("📄 Individual File Results")
            
            for result in all_results:
                with st.expander(f"📝 {result['file_name']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentences", result['total_sentences'])
                    with col2:
                        st.metric("Words", result['total_words'])
                    with col3:
                        st.metric("Constructions", result['counter_clause'])
                    
                    st.subheader("Construction Distribution")
                    
                    # Create dataframe for this file
                    construction_data = []
                    for ct, count in result['counters'].items():
                        if count > 0:
                            prop = count / result['counter_clause'] if result['counter_clause'] > 0 else 0
                            construction_data.append({
                                'Construction Type': ct,
                                'Count': count,
                                'Proportion': f"{prop:.4f}"
                            })
                    
                    if construction_data:
                        st.dataframe(construction_data, use_container_width=True)
                    
                    st.metric("MATTR Score", 
                             f"{result['mattr_value']:.4f}" if isinstance(result['mattr_value'], (int, float)) 
                             else result['mattr_value'])
                    
                    # Show sample constructions
                    if result['clause_details']:
                        with st.expander("View sample constructions (first 10)"):
                            for detail in result['clause_details'][:10]:
                                st.markdown(f"**{detail['construction']}**: {detail['pattern']}")
                                st.caption(f"_{detail['sentence']}_")
                                st.divider()
            
            # Create CSV output
            st.header("💾 Download Results")
            
            headers = [
                "file_name", "total_num_sentences", "total_num_words",
                "token_frequency_of_constructions", "type_frequency_of_constructions",
                "log_transformed_type_frequency_of_constructions", "MATTR"
            ]
            
            construction_types = [
                "there", "attributive", "simple_intransitive", "intransitive_motion",
                "intransitive_resultative", "passive", "simple_transitive", "caused_motion",
                "ditransitive", "transitive_resultative", "phrasal_verb", "n/a"
            ]
            
            headers.extend(construction_types)
            headers.extend([f"{ct}_prop" for ct in construction_types])
            headers.extend([f"arcsine_transformed_{ct}_prop" for ct in construction_types])
            
            # Create CSV in memory
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            
            for result in all_results:
                row = create_csv_row(result)
                writer.writerow(row)
            
            csv_data = output.getvalue()
            
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="constructional_diversity.csv",
                mime="text/csv",
                type="primary"
            )

else:
    st.info("👆 Please upload one or more .txt files to begin analysis.")
    
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### Instructions:
        1. **Upload text files** (.txt format)
        2. **Configure dictionaries** (optional) in the sidebar
        3. **Click 'Analyze Constructions'** to process
        4. **Review results** for each file
        5. **Download CSV** with detailed statistics
        
        ### Construction Types:
        - **Simple Intransitive**: Verb with no object (e.g., "She runs")
        - **Simple Transitive**: Verb with direct object (e.g., "She reads books")
        - **Ditransitive**: Verb with two objects (e.g., "She gave him a book")
        - **Passive**: Passive voice construction (e.g., "The book was read")
        - **Caused Motion**: Motion caused by action (e.g., "She threw the ball to him")
        - **Attributive**: Copular constructions (e.g., "She is happy")
        - **Resultative**: Action with result (e.g., "She painted the wall red")
        - **Phrasal Verb**: Verb with particle (e.g., "She turned on the light")
        - **Intransitive Motion**: Motion verb (e.g., "She walked to the store")
        - **There**: Existential there (e.g., "There is a book")
        
        ### MATTR:
        Moving Average Type-Token Ratio measures the diversity of construction types 
        using a sliding window approach. Higher scores indicate greater diversity.
        """)
    
    with st.expander("📋 Example text format"):
        st.markdown("""
        Your text files should contain plain English text. For example:
        
        ```
        The cat sat on the mat. Mary gave John a book. 
        The students were reading in the library. 
        She walked to the store and bought some milk.
        ```
        
        The analyzer will automatically identify and classify syntactic constructions.
        """)
