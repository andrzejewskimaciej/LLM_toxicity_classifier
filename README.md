# 🛡️ AI Toxicity Classifier

## Overview

The **AI Toxicity Classifier** is a robust, automated tool designed to detect and categorize toxic content in text. Its primary goal is to identify harmful behaviors such as **threats, insults, obscenity, and identity attacks**, going beyond simple keyword filtering to understand **context and nuance** (including irony and satire).

---

## 🎯 Target Audience

- **Content Moderators** – Automate the initial triage of reported content  
- **Trust & Safety Teams** – Ensure online safety and regulatory compliance  
- **Platform Owners** – Scale moderation for forums, comment sections, and social applications  

---

## 🏗️ Architecture

The project follows a clear separation of concerns and provides both programmatic access and a visual interface:

- **API (FastAPI)**  
  A high-performance HTTP server handling single and batch requests.  
  Returns structured JSON with toxicity scores and AI-generated reasoning.

- **UI (Streamlit)**  
  An interactive web dashboard for human moderators:
  - Test model behavior  
  - View radar charts  
  - Inspect contextual justifications  

---

## 🚀 Two Approaches

The system offers two architectural variants depending on budget, privacy, and infrastructure needs.

---

### 1. ☁️ Cloud-Based (Google Gemini)

- **Model**: Gemini 3.0 Flash (via API)
- **Performance**: ~85% accuracy
- **Capabilities**:
  - Deep contextual understanding
  - Strong handling of sarcasm and cultural nuance
  - Native multilingual support (English & Polish)

**Pros**
- High precision  
- No local hardware or maintenance  

**Cons**
- API costs after free tier  
- Data leaves your infrastructure  

---

### 2. 🖥️ Local Hybrid (Toxic-BERT + Llama 3.2)

A fully open-source, on-premise pipeline combining two models:

- **Toxic-BERT**  
  - Lightweight, fast classifier  
  - Categorizes text into **6 toxicity types** (0–100% scores)

- **Llama 3.2 (via Ollama)**  
  - Triggered only when BERT flags suspicious content  
  - Provides reasoning and irony/satire checks  

- **Performance**: ~72% accuracy
- **Language Support**:
  - Optimized for English  
  - Empirically performs quite well on Polish content  

**Pros**
- 100% free  
- Full data privacy (offline)  
- Efficient on consumer-grade CPUs  

---

## 🛠️ Implementation Details

- **Language**: Python (entire stack)
- **Backend**: FastAPI
  - Request validation (Pydantic)
  - Batch processing
  - Model orchestration
- **Frontend**: Streamlit
  - Real-time interaction
  - Plotly-based visualizations
- **Containerization**:
  - Docker for all services
  - Docker Compose for orchestration
  - Consistent environments across deployments  

---

## 📦 Deployment (Docker)

The project is fully containerized for easy deployment.  
All configuration files are located in the `docker-files` directory.

### Prerequisites

- Docker  
- Docker Compose  
- Google API Key *(required for Cloud/Gemini approach)*  

---

### ▶️ How to Run

#### 1. Set Environment Variable (for Cloud version)

Export your Google API key before starting the containers:

```bash
# Linux / macOS
export GOOGLE_API_KEY=your_key_here

# Windows PowerShell
$Env:GOOGLE_API_KEY="your_key_here"

# Windows CMD
set GOOGLE_API_KEY=your_key_here
```

#### 2. Run the Chosen Environment

* Cloud (Gemini) version:
  ```
  docker-compose -f docker-files/docker-compose.gemini.yml up --build
  ```

* Local (Hybrid) version:
  ```
  docker-compose -f docker-files/docker-compose.local.yml up --build
  ```

*Note: The local version will automatically download the required LLM models on the first launch. Additionaly, both containers take up some space and time to set up.*

### 🌐 Access the Application

* **Web UI**: http://localhost:8501

* **API Docs (Swagger)**: http://localhost:8000/docs