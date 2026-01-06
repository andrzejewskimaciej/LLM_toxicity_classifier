import streamlit as st
import ollama
import json
import plotly.graph_objects as go
from transformers import pipeline

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Hybrid AI Toxicity Classifier", page_icon="☣️", layout="wide"
)

st.title("☣️ Hybrid AI Toxicity Classifier")
st.markdown(
    """
This tool combines two distinct AI models for efficient moderation:
1.  **Toxic-BERT:** Instant statistical scoring (0-100%) for 6 categories.
2.  **Llama 3.2 (Ollama):** Contextual analysis and reasoning (triggered only for suspicious content).
"""
)

# Inicjalizacja Session State (Pamięć podręczna aplikacji)
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "user_text_cache" not in st.session_state:
    st.session_state.user_text_cache = ""

# --- 2. MODEL FUNCTIONS ---


@st.cache_resource
def load_bert_model():
    """Loads the Toxic-BERT model into memory."""
    with st.spinner("Loading Toxic-BERT model into RAM..."):
        return pipeline("text-classification", model="unitary/toxic-bert", top_k=None)


def analyze_with_ollama(text):
    """Sends a request to the local Llama 3.2 model via Ollama."""
    prompt = f"""
    You are a content moderation AI. Analyze the following text for toxicity.
    Text: "{text}"
    
    Return a valid JSON object with the following fields:
    - "is_ironic": boolean (true if sarcasm/irony is detected)
    - "justification": string (Explain why it is toxic or safe in English)
    - "deciding_fragments": list of strings (specific quotes from the text)
    
    Return ONLY JSON.
    """

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        return json.loads(response["message"]["content"])
    except Exception as e:
        st.error(f"Ollama Error: {e}")
        return None


def complain_about_decision(text, initial_decision_is_toxic):
    """
    Sends the text to Llama 3.2 to argue with the decision.
    Converted from Google GenAI to Ollama.
    """
    prompt = f"""
    You have just analyzed the following text for toxicity levels.
    Text: "{text}"
    Your answer was that this text IS {"" if initial_decision_is_toxic else "NOT"} toxic.
    I do not agree with that. Reconsider your decision and justify your new response.
    
    Return a JSON object with this exact field:
    - "new_decision": string (Your new reasoning in the text's original language)
    
    Return ONLY JSON.
    """

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0},
        )
        # Parsowanie wyniku
        data = json.loads(response["message"]["content"])
        # Zwracamy obiekt przypominający strukturę Pydantic z Twojego przykładu
        return type(
            "obj",
            (object,),
            {"new_decision": data.get("new_decision", "No decision provided")},
        )
    except Exception as e:
        st.error(f"Ollama Complaint Error: {e}")
        return None


def plot_metrics(scores):
    """Generates a Plotly Bar Chart based on BERT scores."""
    labels = [item["label"] for item in scores]
    values = [item["score"] for item in scores]
    colors = ["#ff4b4b" if v > 0.5 else "#09ab3b" for v in values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1%}" for v in values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Toxic-BERT Confidence Scores",
        xaxis_range=[0, 1],
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# --- 3. USER INTERFACE ---

with st.sidebar:
    st.header("⚙️ Configuration")
    threshold = st.slider(
        "Llama Trigger Threshold",
        0.0,
        1.0,
        0.40,
        0.05,
        help="Llama 3.2 will only run if BERT detects toxicity above this level.",
    )
    st.info(f"ℹ️ Llama triggers if BERT > {threshold:.0%}")
    st.caption("Powered by Local AI")

col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("1. Input Text")
    user_text = st.text_area(
        "Enter content to analyze:",
        height=150,
        placeholder="e.g., You are absolutely useless...",
    )
    # Przycisk uruchamia logikę i zapisuje do STANU
    analyze_btn = st.button("Analyze Text", type="primary", width="stretch")

# Logika uruchamiania analizy
if analyze_btn and user_text:
    # Zapisujemy tekst do cache, żeby był dostępny dla przycisku skargi
    st.session_state.user_text_cache = user_text

    # 1. BERT
    classifier = load_bert_model()
    bert_results = classifier(user_text)[0]
    max_score = max([item["score"] for item in bert_results])

    # 2. Llama (opcjonalnie)
    ollama_result = None
    if max_score > threshold:
        with st.spinner("Llama is reading the context..."):
            ollama_result = analyze_with_ollama(user_text)

    # ZAPISUJEMY WYNIK DO SESJI
    st.session_state.analysis_result = {
        "bert_results": bert_results,
        "max_score": max_score,
        "ollama_result": ollama_result,
        "is_toxic_flag": max_score > 0.5,  # Flaga dla funkcji skargi
    }

# --- WYŚWIETLANIE WYNIKÓW (Oparte na Stanie) ---
if st.session_state.analysis_result:
    data = st.session_state.analysis_result

    with col_results:
        st.subheader("2. Analysis Results")
        st.plotly_chart(plot_metrics(data["bert_results"]), width="stretch")

        # Wyświetlanie wyników Llama (jeśli była uruchomiona)
        if data["ollama_result"]:
            res = data["ollama_result"]
            st.markdown("### 🧠 Llama 3.2 Insights")

            irony = "**Not**" if not res.get("is_ironic") else ""
            st.info(f"💡 Sarcasm / Irony {irony} Detected")
            st.markdown(f"**Reasoning:**\n> {res.get('justification')}")

            frags = res.get("deciding_fragments", [])
            if frags:
                st.markdown("**Flagged Keywords:**")
                for f in frags:
                    st.code(f, language="text")
        else:
            # Jeśli Llama się nie uruchomiła (niski wynik BERT)
            st.success(f"✅ Content appears safe (Max score: {data['max_score']:.1%}).")
            st.caption("Llama 3.2 analysis skipped.")

        st.markdown("---")

        # --- SEKCJA SKARGI ---
        # Teraz jest bezpieczna, bo jest w bloku zależnym od session_state
        complaint_btn = st.button(
            "Complain about the answer", type="secondary", width="stretch"
        )

        if complaint_btn:
            with st.spinner("Llama is reconsidering..."):
                # Pobieramy tekst z cache sesji
                text_to_argue = st.session_state.user_text_cache
                is_toxic = data["is_toxic_flag"]

                argue_response = complain_about_decision(text_to_argue, is_toxic)

            if argue_response:
                st.markdown("### 🧠 AI Complaint Response")
                st.info(argue_response.new_decision)

elif not st.session_state.analysis_result and analyze_btn and not user_text:
    st.warning("Please enter some text.")
