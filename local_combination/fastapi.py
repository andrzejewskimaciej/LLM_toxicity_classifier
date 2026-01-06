import json
import ollama
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

# --- 1. GLOBAL STATE & LIFESPAN MANAGEMENT ---

# Global variable for the BERT model
bert_classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager.
    Logic before 'yield' runs on startup.
    Logic after 'yield' runs on shutdown.
    """
    # --- STARTUP LOGIC ---
    global bert_classifier
    print("⚙️ [STARTUP] Loading Toxic-BERT model...")
    try:
        bert_classifier = pipeline(
            "text-classification", model="unitary/toxic-bert", top_k=None
        )
        print("✅ [STARTUP] Toxic-BERT model is ready.")
    except Exception as e:
        print(f"❌ [STARTUP ERROR] Could not load BERT model: {e}")

    yield  # Application runs here

    # --- SHUTDOWN LOGIC ---
    print("🛑 [SHUTDOWN] Cleaning up resources...")
    bert_classifier = None


# --- 2. API CONFIGURATION ---

app = FastAPI(
    title="Hybrid Toxicity Classifier API",
    description="API combining Toxic-BERT (fast statistical scoring) and Llama 3.2 (contextual reasoning).",
    version="1.0",
    lifespan=lifespan,  # Link the lifespan handler here
)

# --- 3. DATA MODELS (Pydantic) ---


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="The content to be analyzed.")
    threshold: float = Field(
        0.40,
        description="Confidence threshold (0.0-1.0). If BERT score > threshold, Llama is triggered.",
    )


class BertScore(BaseModel):
    label: str
    score: float


class LlamaAnalysis(BaseModel):
    is_ironic: bool
    justification: str
    deciding_fragments: List[str]


class AnalyzeResponse(BaseModel):
    bert_scores: List[BertScore]
    max_score: float
    is_toxic_flag: bool
    llama_analysis: Optional[LlamaAnalysis] = None


# --- 4. HELPER FUNCTIONS ---


def run_ollama_analysis(text: str) -> Optional[dict]:
    """Sends a request to the local Llama 3.2 model for a JSON analysis."""
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
        print(f"❌ Ollama Error: {e}")
        return None


# --- 5. API ENDPOINTS ---


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Main hybrid endpoint.
    1. Runs Toxic-BERT (Fast).
    2. If max_score > threshold -> Runs Llama 3.2 (Slow/Contextual).
    """
    if not bert_classifier:
        raise HTTPException(status_code=503, detail="BERT model is not loaded yet.")

    # STEP 1: Fast BERT Analysis
    # The pipeline returns a list of lists of dicts [[{'label':..., 'score':...}]]
    bert_raw = bert_classifier(request.text)[0]

    # Convert to Pydantic models
    bert_scores = [
        BertScore(label=item["label"], score=item["score"]) for item in bert_raw
    ]

    # Find the maximum toxicity score
    max_score = max(item.score for item in bert_scores)
    is_toxic = max_score > 0.5

    # STEP 2: Conditional Llama Analysis
    llama_result = None

    if max_score > request.threshold:
        print(
            f"⚠️ Trigger: Score {max_score:.2f} > {request.threshold}. Starting Llama..."
        )
        raw_llama = run_ollama_analysis(request.text)

        if raw_llama:
            llama_result = LlamaAnalysis(
                is_ironic=raw_llama.get("is_ironic", False),
                justification=raw_llama.get(
                    "justification", "No justification provided."
                ),
                deciding_fragments=raw_llama.get("deciding_fragments", []),
            )
    else:
        print(f"✅ Safe: Score {max_score:.2f}. Skipping Llama.")

    # Return combined result
    return AnalyzeResponse(
        bert_scores=bert_scores,
        max_score=max_score,
        is_toxic_flag=is_toxic,
        llama_analysis=llama_result,
    )


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": bert_classifier is not None}


if __name__ == "__main__":
    import uvicorn

    print("Start serwera na http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
