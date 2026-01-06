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
        # top_k=None returns scores for all labels
        # The pipeline automatically handles batch inputs (list of strings) efficiently
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
    title="Hybrid Toxicity Classifier API (Batch Support)",
    description="API combining Toxic-BERT and Llama 3.2. Supports batch processing.",
    version="1.1",
    lifespan=lifespan,
)

# --- 3. DATA MODELS (Pydantic) ---


# Input model now accepts a LIST of strings
class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to be analyzed.")
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


# Structure for a SINGLE text result
class SingleAnalysisResult(BaseModel):
    text: str  # Return the original text for reference
    bert_scores: List[BertScore]
    max_score: float
    is_toxic_flag: bool
    llama_analysis: Optional[LlamaAnalysis] = None


# API Response Wrapper
class BatchAnalyzeResponse(BaseModel):
    results: List[SingleAnalysisResult]
    total_processed: int


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


@app.post("/analyze-batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    """
    Batch hybrid endpoint.
    1. Runs Toxic-BERT on ALL texts at once (vectorized/batched operation).
    2. Iterates through results.
    3. If specific text max_score > threshold -> Runs Llama 3.2 (Sequentially).
    """
    if not bert_classifier:
        raise HTTPException(status_code=503, detail="BERT model is not loaded yet.")

    if not request.texts:
        raise HTTPException(
            status_code=400, detail="Input list 'texts' cannot be empty."
        )

    # STEP 1: Batch BERT Analysis
    # Passing a list of strings to the pipeline is much faster than a loop
    # returns: List[List[Dict]] -> One list of scores per input text
    print(f"🚀 Processing batch of {len(request.texts)} texts with BERT...")
    batch_bert_raw = bert_classifier(request.texts)

    final_results = []

    # STEP 2: Process results and conditionally trigger Llama
    for i, text in enumerate(request.texts):
        # Extract BERT results for current text
        bert_raw = batch_bert_raw[i]

        # Convert to Pydantic models
        bert_scores = [
            BertScore(label=item["label"], score=item["score"]) for item in bert_raw
        ]

        # Find the maximum toxicity score
        max_score = max(item.score for item in bert_scores)
        is_toxic = max_score > 0.5

        # Conditional Llama Analysis
        llama_result = None

        if max_score > request.threshold:
            print(
                f"⚠️ [Text {i+1}] Trigger: Score {max_score:.2f} > {request.threshold}. Starting Llama..."
            )
            # Llama is processed sequentially here (could be parallelized with more complexity)
            raw_llama = run_ollama_analysis(text)

            if raw_llama:
                llama_result = LlamaAnalysis(
                    is_ironic=raw_llama.get("is_ironic", False),
                    justification=raw_llama.get(
                        "justification", "No justification provided."
                    ),
                    deciding_fragments=raw_llama.get("deciding_fragments", []),
                )
        else:
            # Debug log for safe text (optional, can be spammy for large batches)
            # print(f"✅ [Text {i+1}] Safe: Score {max_score:.2f}. Skipping Llama.")
            pass

        # Build single result object
        result_obj = SingleAnalysisResult(
            text=text,
            bert_scores=bert_scores,
            max_score=max_score,
            is_toxic_flag=is_toxic,
            llama_analysis=llama_result,
        )
        final_results.append(result_obj)

    # Return combined batch result
    return BatchAnalyzeResponse(
        results=final_results, total_processed=len(final_results)
    )


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": bert_classifier is not None}


# for testing purpouses
if __name__ == "__main__":
    import uvicorn

    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
