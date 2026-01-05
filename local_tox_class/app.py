import streamlit as st
import ollama
import json
import plotly.graph_objects as go
from transformers import pipeline

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


@st.cache_resource
def load_bert_model():
    """
    Loads the Toxic-BERT model into memory.
    Cached to avoid reloading on every interaction.
    """
    with st.spinner("Loading Toxic-BERT model into RAM..."):
        # return_all_scores=True ensures we get the full vector of 6 categories
        return pipeline(
            "text-classification", model="unitary/toxic-bert", return_all_scores=True
        )


def analyze_with_ollama(text):
    """
    Sends a request to the local Llama 3.2 model via Ollama.
    Expects a structured JSON response.
    """
    prompt = f"""
    You are a content moderation AI. Analyze the following text for toxicity.
    Text: "{text}"
    
    Return a valid JSON object with the following fields:
    - "is_ironic": boolean (true if sarcasm/irony is detected)
    - "justification": string (Explain why it is toxic or safe in English)
    - "deciding_fragments": list of strings (specific quotes from the text)
    
    Return ONLY JSON. Do not include markdown formatting like ```json.
    """

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            format="json",  # Forces JSON mode (Ollama feature)
            options={"temperature": 0.0},  # Deterministic output
        )
        return json.loads(response["message"]["content"])
    except Exception as e:
        st.error(f"Ollama Connection Error: {e}. Is Ollama running?")
        return None


def plot_metrics(scores):
    """
    Generates a Plotly Bar Chart based on BERT scores.
    """
    # Scores come as a list of dicts: [{'label': 'toxic', 'score': 0.9}, ...]
    labels = [item["label"] for item in scores]
    values = [item["score"] for item in scores]

    # Color logic: Red if > 0.5, Green if < 0.5
    colors = ["#ff4b4b" if v > 0.5 else "#09ab3b" for v in values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",  # Horizontal bars
                marker_color=colors,
                text=[f"{v:.1%}" for v in values],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Toxic-BERT Confidence Scores",
        xaxis_title="Probability (0.0 - 1.0)",
        xaxis_range=[0, 1],
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# --- 3. USER INTERFACE ---

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Slider to control when the "Heavy" model (Llama) kicks in
    threshold = st.slider(
        "Llama Trigger Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Llama 3.2 will only run if BERT detects toxicity above this level.",
    )

    st.info(
        f"ℹ️ Contextual analysis (Llama) will trigger only if BERT score > {threshold:.0%}"
    )
    st.markdown("---")
    st.caption("Powered by Local AI")

# Main Layout: Two columns
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("1. Input Text")
    user_text = st.text_area(
        "Enter content to analyze:",
        height=150,
        placeholder="e.g., You are absolutely useless...",
    )
    analyze_btn = st.button("Analyze Text", type="primary", width="stretch")

# Application Logic
if analyze_btn and user_text:
    # A. Run BERT (Fast)
    classifier = load_bert_model()
    bert_results = classifier(user_text)[
        0
    ]  # [0] because pipeline returns a list of lists

    # Find the maximum toxicity score among all categories
    max_score = max([item["score"] for item in bert_results])

    with col_results:
        st.subheader("2. Analysis Results")

        # Display BERT Chart
        st.plotly_chart(plot_metrics(bert_results), width="stretch")

        # B. Check Threshold & Run Llama (Slow)
        if max_score > threshold:
            st.warning(
                f"⚠️ Potential toxicity detected ({max_score:.1%}). Starting Llama 3.2 for contextual analysis..."
            )

            with st.spinner("Llama is reading the context..."):
                ollama_result = analyze_with_ollama(user_text)

            if ollama_result:
                st.markdown("### 🧠 Llama 3.2 Insights")

                # Irony Badge
                if ollama_result.get("is_ironic"):
                    st.info("💡 Sarcasm / Irony Detected")

                # Justification
                st.markdown(f"**Reasoning:**\n> {ollama_result.get('justification')}")

                # Highlighted Fragments
                frags = ollama_result.get("deciding_fragments", [])
                if frags:
                    st.markdown("**Flagged Keywords:**")
                    for f in frags:
                        st.code(f, language="text")
        else:
            # Safe Path
            st.success(f"✅ Content appears safe (Max score: {max_score:.1%}).")
            st.caption("Llama 3.2 analysis skipped to save resources.")

elif analyze_btn and not user_text:
    st.warning("Please enter some text to analyze.")
