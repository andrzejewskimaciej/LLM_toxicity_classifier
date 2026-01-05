import streamlit as st
import os
import plotly.graph_objects as go
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Toxicity Classifier", page_icon="🛡️", layout="wide")


# --- 2. DATA MODEL (PYDANTIC) ---
class ToxicityAnalysis(BaseModel):
    toxicity: float = Field(..., description="General toxicity score (0-1).")
    severe_toxicity: float = Field(..., description="Severe toxicity score (0-1).")
    obscene: float = Field(..., description="Obscenity score (0-1).")
    threat: float = Field(..., description="Threat score (0-1).")
    insult: float = Field(..., description="Insult score (0-1).")
    identity_attack: float = Field(..., description="Identity attack score (0-1).")
    sexual_explicit: float = Field(..., description="Sexually explicit score (0-1).")

    deciding_fragments: List[str] = Field(
        ...,
        description="List of specific text fragments/quotes that influenced the score.",
    )
    ambiguous_fragments: List[str] = Field(
        ..., description="Fragments difficult to classify (e.g., sarcasm, irony)."
    )
    justification: str = Field(
        ..., description="Detailed reasoning for the classification in English."
    )


# --- 3. HELPER FUNCTIONS ---


@st.cache_resource
def get_client(api_key):
    """Initializes the Google GenAI client."""
    return genai.Client(api_key=api_key)


def analyze_text(client, text):
    """Sends the text to Gemini 3 Flash for structured analysis."""
    prompt = f"""
    Analyze the following text for toxicity levels.
    Text: "{text}"
    
    Return the result strictly adhering to the JSON schema.
    Provide the justification in text's original language.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ToxicityAnalysis,
                temperature=0.0,  # Deterministic output
            ),
        )
        return response.parsed
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def create_radar_chart(data: ToxicityAnalysis):
    """Generates a Radar Chart using Plotly."""
    categories = [
        "Toxicity",
        "Severe",
        "Obscene",
        "Threat",
        "Insult",
        "Identity",
        "Sexual",
    ]
    values = [
        data.toxicity,
        data.severe_toxicity,
        data.obscene,
        data.threat,
        data.insult,
        data.identity_attack,
        data.sexual_explicit,
    ]

    # Close the polygon for the radar chart
    categories = [*categories, categories[0]]
    values = [*values, values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name="Score",
                line_color="#FF4B4B",
            )
        ],
        layout=go.Layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            height=400,
        ),
    )
    return fig


# --- 4. USER INTERFACE ---

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")

    # # Check for API Key in environment or input
    env_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key_input = st.text_input("Google API Key", type="password", value=env_key)

    if not api_key_input:
        st.warning("Please enter your Google API Key to proceed.")
        st.stop()

    client = get_client(api_key_input)
    st.success("API Key connected!")
    st.markdown("---")
    st.info("Powered by **Gemini 3.0 Flash**")
    # st.caption("Mode: Structured Output (Strict JSON)")

# Main Layout
st.title("🛡️ AI Toxicity Classifier")
st.markdown(
    "Analyze text for toxic content, threats, and insults using Google's Gemini 3 Flash model."
)

# col_input, col_result = st.columns([1, 1], gap="large")

# # Left Column: Input
# with col_input:
#     st.subheader("Input Text")
#     user_text = st.text_area(
#         "Paste the content below:",
#         height=250,
#         placeholder="e.g., You are absolutely useless and I hate your opinion...",
#     )

#     analyze_btn = st.button("Analyze Text", type="primary", width='stretch')

# # Right Column: Analysis Results
# with col_result:
#     if analyze_btn and user_text:
#         with st.spinner("Gemini is analyzing context and nuances..."):
#             result = analyze_text(client, user_text)

#         if result:
#             st.subheader("Analysis Results")

#             # 1. Radar Chart
#             st.plotly_chart(create_radar_chart(result), width='stretch')

#             # 2. Key Metrics
#             m1, m2, m3 = st.columns(3)
#             m1.metric(
#                 "General Toxicity",
#                 value=f"{result.toxicity:.1%}",
#                 delta_color="inverse",
#             )
#             m2.metric(
#                 "Severe Toxicity",
#                 value=f"{result.severe_toxicity:.1%}",
#                 delta_color="inverse",
#             )
#             m3.metric(
#                 "Insult Score", value=f"{result.insult:.1%}", delta_color="inverse"
#             )

#             # 3. Detailed Report (Expander)
#             with st.expander("🔍 Full Report & Justification", expanded=True):
#                 st.markdown("### 🧠 AI Justification")
#                 st.info(result.justification)

#                 st.markdown("---")

#                 c1, c2 = st.columns(2)
#                 with c1:
#                     st.markdown("**🚩 Decisive Fragments:**")
#                     if result.deciding_fragments:
#                         for frag in result.deciding_fragments:
#                             st.error(f'"{frag}"')
#                     else:
#                         st.caption("None identified.")

#                 with c2:
#                     st.markdown("**🤔 Ambiguous / Ironic:**")
#                     if result.ambiguous_fragments:
#                         for frag in result.ambiguous_fragments:
#                             st.warning(f'"{frag}"')
#                     else:
#                         st.caption("None identified.")

#     elif analyze_btn and not user_text:
#         st.warning("Please enter some text to analyze.")
#     else:
#         # Placeholder state before analysis
#         st.info(
#             "👋 Enter text on the left and click 'Analyze Text' to see the breakdown."
#         )

st.subheader("Input Text")
user_text = st.text_area(
    "Paste the content below:",
    height=250,
    placeholder="e.g., You are absolutely useless and I hate your opinion...",
)

analyze_btn = st.button("Analyze Text", type="primary", width="stretch")

if analyze_btn and user_text:
    with st.spinner("Gemini is analyzing context and nuances..."):
        result = analyze_text(client, user_text)

    if result:
        st.subheader("Analysis Results")

        # 1. Radar Chart
        st.plotly_chart(create_radar_chart(result), width="stretch")

        # 2. Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "General Toxicity",
            value=f"{result.toxicity:.1%}",
            delta_color="inverse",
        )
        m2.metric(
            "Severe Toxicity",
            value=f"{result.severe_toxicity:.1%}",
            delta_color="inverse",
        )
        m3.metric("Insult Score", value=f"{result.insult:.1%}", delta_color="inverse")

        # 3. Detailed Report (Expander)
        with st.expander("🔍 Full Report & Justification", expanded=True):
            st.markdown("### 🧠 AI Justification")
            st.info(result.justification)

            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🚩 Decisive Fragments:**")
                if result.deciding_fragments:
                    for frag in result.deciding_fragments:
                        st.error(f'"{frag}"')
                else:
                    st.caption("None identified.")

            with c2:
                st.markdown("**🤔 Ambiguous / Ironic:**")
                if result.ambiguous_fragments:
                    for frag in result.ambiguous_fragments:
                        st.warning(f'"{frag}"')
                else:
                    st.caption("None identified.")

elif analyze_btn and not user_text:
    st.warning("Please enter some text to analyze.")
else:
    # Placeholder state before analysis
    st.info("👋 Enter text on the left and click 'Analyze Text' to see the breakdown.")
