import os
import dotenv
import json
import time
import tempfile
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

dotenv.load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_ID = "gemini-3-flash-preview"

if not API_KEY:
    raise ValueError("❌ API Key missing! Set environmental variable GOOGLE_API_KEY.")

client = genai.Client(api_key=API_KEY)

app = FastAPI(title="Gemini Batch Toxicity API")

# --- 1. MODELE DANYCH (PYDANTIC) ---


# Schemat odpowiedzi modelu
class ToxicityAnalysis(BaseModel):
    toxicity: float = Field(..., description="General toxicity score (0-1).")
    severe_toxicity: float = Field(..., description="Severe toxicity score (0-1).")
    obscene: float = Field(..., description="Obscenity score (0-1).")
    threat: float = Field(..., description="Threat score (0-1).")
    insult: float = Field(..., description="Insult score (0-1).")
    identity_attack: float = Field(..., description="Identity attack score (0-1).")
    sexual_explicit: float = Field(..., description="Sexually explicit score (0-1).")
    deciding_fragments: List[str] = Field(..., description="Decisive text fragments.")
    justification: str = Field(
        ..., description="Justification in text's original language."
    )


# Model pojedynczego komentarza wejściowego
class CommentInput(BaseModel):
    id: str = Field(..., description="Unique ID for the comment (to map results back)")
    text: str = Field(..., description="Content to analyze")


# Model requestu do API
class BatchRequest(BaseModel):
    comments: List[CommentInput]


# Model pojedynczego wyniku (połączony)
class AnalyzedComment(BaseModel):
    id: str
    text: str
    analysis: ToxicityAnalysis | None = None
    error: str | None = None


# Model odpowiedzi API
class BatchResponse(BaseModel):
    results: List[AnalyzedComment]
    total_processed: int
    status: str


# --- 2. LOGIKA BATCH API ---


def create_jsonl_content(comments: List[CommentInput]) -> str:
    """Tworzy zawartość pliku JSONL dla Batch API."""
    schema_dict = ToxicityAnalysis.model_json_schema()
    jsonl_lines = []

    for comment in comments:
        prompt_text = f"""Analyze toxicity for: "{comment.text}". Return JSON based on schema. Justification in text's original language."""

        request_body = {
            "custom_id": comment.id,
            "method": "models.generateContent",
            "request": {
                "model": f"models/{MODEL_ID}",
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "response_json_schema": schema_dict,
                },
            },
        }
        jsonl_lines.append(json.dumps(request_body))

    return "\n".join(jsonl_lines)


def run_google_batch_process(jsonl_file_path: str) -> str:
    """Wysyła plik, uruchamia zadanie i zwraca nazwę pliku wynikowego (URI)."""

    # 1. Upload
    print("⬆️ Wysyłanie pliku wsadowego do Google...")
    batch_input_file = client.files.upload(
        file=jsonl_file_path, config={"mime_type": "application/json"}
    )

    # Czekanie na aktywację pliku
    while batch_input_file.state.name == "STATE_PROCESSING":
        time.sleep(1)
        batch_input_file = client.files.get(name=batch_input_file.name)

    # 2. Start Job
    print(f"🚀 Uruchamianie zadania Batch (Model: {MODEL_ID})...")
    batch_job = client.batches.create(
        model=MODEL_ID,
        src=batch_input_file.name,
    )

    print(f"⏳ Job ID: {batch_job.name}. Oczekiwanie na wyniki...")

    # 3. Polling
    while True:
        job_status = client.batches.get(name=batch_job.name)
        state = job_status.state.name

        if state == "JOB_STATE_SUCCEEDED":
            print("✅ Zadanie zakończone!")
            return job_status.dest.file_name
        elif state == "JOB_STATE_FAILED":
            raise RuntimeError(f"Batch Job Failed: {job_status.error}")
        elif state in ["JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            raise RuntimeError(f"Batch Job Stopped: {state}")

        # Czekaj 5 sekund przed kolejnym sprawdzeniem
        print("waitin " + state)
        time.sleep(5)


def parse_results(
    output_file_name: str, original_map: Dict[str, str]
) -> List[AnalyzedComment]:
    """Pobiera wyniki z Google i łączy je z oryginalnymi tekstami."""

    # Pobierz zawartość pliku wyjściowego
    file_content = client.files.download(file=output_file_name)
    # Dekodowanie bajtów do stringa
    text_content = file_content.decode("utf-8")

    results_map: Dict[str, Any] = {}

    # Parsowanie linii JSONL z odpowiedzi
    for line in text_content.strip().split("\n"):
        if not line:
            continue
        try:
            item = json.loads(line)
            custom_id = item.get("custom_id")  # Używamy .get dla bezpieczeństwa

            # --- POPRAWKA ---
            # W Twoim JSONie 'candidates' są bezpośrednio w 'response',
            # nie ma klucza 'body'.
            response_data = item.get("response", {})
            candidates = response_data.get("candidates", [])

            if candidates:
                # Ścieżka: candidates[0] -> content -> parts[0] -> text
                # Pobieramy tekst, który jest stringiem JSON
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    raw_json_text = parts[0].get("text", "{}")

                    # Czasami model może dodać znaczniki markdown, czyścimy je
                    raw_json_text = (
                        raw_json_text.replace("```json", "").replace("```", "").strip()
                    )

                    analysis_data = json.loads(raw_json_text)
                    results_map[custom_id] = analysis_data
                else:
                    results_map[custom_id] = {"error": "Empty parts in response"}
            else:
                # Sprawdzamy czy nie ma błędu w samej odpowiedzi (np. filtr bezpieczeństwa)
                error_info = response_data.get("error", "No candidates returned")
                results_map[custom_id] = {"error": str(error_info)}

        except Exception as e:
            print(f"Błąd parsowania linii: {e}")
            if custom_id:
                results_map[custom_id] = {"error": str(e)}

    # Łączenie wyników (Merge)
    final_output = []
    for cid, text in original_map.items():
        # cid musi być stringiem, bo custom_id w JSONL jest stringiem
        cid = str(cid)
        analysis = results_map.get(cid)

        # Sprawdzamy czy analiza to sukces czy błąd
        if analysis and "error" not in analysis:
            final_output.append(AnalyzedComment(id=cid, text=text, analysis=analysis))
        else:
            error_msg = (
                analysis.get("error")
                if analysis
                else "No result found (Job failed or ID mismatch)"
            )
            final_output.append(
                AnalyzedComment(id=cid, text=text, error=str(error_msg))
            )

    return final_output


# --- 3. ENDPOINT API ---


@app.post("/analyze-batch", response_model=BatchResponse)
async def analyze_batch_endpoint(request: BatchRequest):
    """
    Przyjmuje listę komentarzy, wysyła do Google Batch API i zwraca wyniki.
    UWAGA: To zapytanie może trwać długo (zależnie od obciążenia Google).
    """

    if not request.comments:
        raise HTTPException(status_code=400, detail="Lista komentarzy jest pusta.")

    # Mapa ID -> Tekst do późniejszego złączenia
    original_map = {c.id: c.text for c in request.comments}

    # Tworzenie pliku tymczasowego
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as temp_file:
        jsonl_content = create_jsonl_content(request.comments)
        temp_file.write(jsonl_content)
        temp_file_path = temp_file.name

    try:
        # 1. Uruchomienie procesu w Google
        output_file_name = run_google_batch_process(temp_file_path)

        # 2. Parsowanie i łączenie wyników
        results = parse_results(output_file_name, original_map)

        return BatchResponse(
            results=results, total_processed=len(results), status="completed"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Sprzątanie pliku lokalnego
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


# for testing purpouses
if __name__ == "__main__":
    import uvicorn

    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
