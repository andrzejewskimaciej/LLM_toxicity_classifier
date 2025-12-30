# 📘 Inteligentny Klasyfikator Toksyczności z Wyjaśnialnością  
---

## 1. Wstęp i Założenia Architektoniczne

Celem projektu jest stworzenie **systemu klasyfikacji komentarzy internetowych (PL/EN)**, który nie tylko ocenia poziom toksyczności, ale również **rozumie kontekst i wyjaśnia swoją decyzję (Explainable AI)**.

### Kluczowe filary projektu

#### 🔁 Dual-Pipeline – dwa niezależne silniki

**Cloud Engine (LLM – Gemini)**  
- Oparty o Google Gemini API  
- Priorytety:
  - głębokie rozumienie semantyki
  - wykrywanie ironii i sarkazmu
  - bogata wyjaśnialność (JSON)
  - rozróżnienie wulgaryzmów od kontekstu decyzyjnego

**Local Engine (Open Source – HuggingFace)**  
- Model offline, uruchamiany lokalnie  
- Priorytety:
  - prywatność
  - szybkość
  - brak wysyłania danych do chmury
  - prosty, liczbowy wynik

---

#### 🐳 Pełna konteneryzacja (MLOps-ready)
- Każdy komponent w osobnym kontenerze
- Orkiestracja przez **docker-compose**

#### 🔍 Wyjaśnialność
- Identyfikacja:
  - słów problematycznych (wulgaryzmy)
  - fragmentów decyzyjnych (ironia, groźby, cytaty)

#### 📊 Ewaluacja naukowa
- Metryki: **MAE, F1-score, Irony Recall**
- Analiza błędów (Failure Analysis)

---

## 2. Etap 1: Przygotowanie Danych (Data Engineering)

### 🎯 Cel
Stworzenie pliku **dataset_benchmark.csv** 
---

### Źródła danych

#### 🇵🇱 BAN-PL (GitHub)
- 20 × komentarze toksyczne  
- 20 × komentarze neutralne  
- 10 × komentarze graniczne (zgłoszone, ale nie zbanowane)

#### 🇬🇧 Civil Comments (Kaggle)
- toxicity > 0.8 → toksyczne  
- toxicity < 0.1 → neutralne  
- toxicity ≈ 0.5 → graniczne

#### Pozostałe

---

### Format CSV

| Kolumna | Opis |
|------|------|
| id | unikalny identyfikator |
| text | treść komentarza |
| lang | `"pl"` lub `"en"` |
| expected_score | wartość 0.0–1.0 |
| is_irony | TRUE / FALSE |
| contains_profanity | TRUE / FALSE |

---

## 3. Etap 2: Backend Cloud (Gemini API)

### Technologia
- Python
- FastAPI
- google-generativeai

---

### System Prompt (JSON-only)
```text
Jesteś ekspertem moderacji treści.
Przeanalizuj tekst i zwróć WYŁĄCZNIE JSON:

{
"toxicity_score": 0.0-1.0,
"attributes": {
"is_irony": true/false,
"is_joke": true/false,
"is_threat": true/false
},
"fragments": {
"problematic_words": [],
"decisive_spans": []
},
"reasoning": "Wyjaśnienie krok po kroku"
}
```
## 4. Etap 3: Backend Local (HuggingFace)

### Technologia
- Python
- FastAPI
- transformers
- torch

---

### Modele

| Język | Model |
|----|----|
| PL | herbert-base-cased / toxic-bert-pl |
| EN | unitary/toxic-bert |

Model wybierany dynamicznie na podstawie języka.

---

### Optymalizacja Dockerowa
- Skrypt `model_loader.py`
- Modele pobrane **na etapie budowania obrazu**

---

## 5. Etap 4: Frontend (GUI)

### Układ strony

**Góra:**  
- Pole tekstowe do wprowadzania komentarza

**Środek:**  
- Checkboxy – Ground Truth użytkownika:
  - [ ] To jest toksyczne
  - [ ] To jest żart / ironia

**Dół:**  
- Lewa kolumna – Local Model (ProgressBar)
- Prawa kolumna – Cloud Model (ProgressBar + wyjaśnienie)

---

### Podświetlanie (JavaScript)

- `problematic_words` → `<span class="highlight-red">`
- `decisive_spans` → `<span class="highlight-yellow">`

---

### Feedback użytkownika

Jeżeli:
- model zwraca `is_irony = false`
- użytkownik zaznaczy „To jest żart”

➡️ komunikat: ⚠️ Model nie wykrył Twojej intencji żartu!

---

## 7. Etap 5: Ewaluacja i Eksperymenty

### evaluate_models.py
- iteracja po `dataset_benchmark.csv`
- zapytania:
  - Gemini (3 różne prompty)
  - Local model
- zapis wyników do `results.csv`

---

### Prompt Engineering – Cloud

1. Zero-shot  
2. Persona (językoznawca / moderator)  
3. Few-shot (3 przykłady ironii)

➡️ wybór najlepszego promptu na podstawie **F1-score**

---

### Metryki

- **MAE (Mean Absolute Error)**
- **Binary F1-score (threshold = 0.5)**
- **Irony Recall**

---

### Failure Analysis

Plik: `failure_analysis.md`

| Text | Expected | Predicted | Reasoning | Comment |
|----|----|----|----|----|

---

## 8. Finalny Produkt

### 📦 Deliverables

#### Repozytorium
- `docker-compose up` uruchamia całość

#### Demo Web
- porównanie Local vs Cloud
- paski procentowe
- kolorowe podświetlenia

#### Raport (PDF / MD)
- opis zbiorów danych
- wykres jakości vs koszt
- wpływ prompt engineeringu
- **5 Success Cases**
- **5 Failure Cases**

---

## 🎓 Efekt końcowy

Projekt łączy:
- LLM + klasyczne NLP
- Explainable AI
- Docker / MLOps
- rygor analizy naukowej
