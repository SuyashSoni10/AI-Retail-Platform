# AI-Powered Smart Retail & Customer Intelligence Platform

An integrated, end-to-end smart retail checkout and customer analytics platform. The application combines Computer Vision, Natural Language Processing (NLP), and statistical telemetry to deliver high-quality client tracking, automatic grocery/retail checkout scanning, customer satisfaction analytics, and an interactive FAQ chatbot.

---

## 🚀 Key Features

1. **👤 Loyalty & Face Kiosk:** Detects returning loyalty program members automatically using face classification (webcam snapshot or photo upload).
2. **📦 Checkout Product Scanner:** Classifies images of products (using transfer learning on MobileNetV2) and registers items for automatic checkout.
3. **💬 Sentiment Analyzer:** Evaluates text review sentiments into Positive, Neutral, or Negative classes using TF-IDF and Logistic Regression.
4. **🤖 FAQ Support Chatbot:** Employs an intent-matching neural text classifier to chat with customers and resolve support queries.
5. **📈 Manager Telemetry Dashboard:** Visualizes total visits, returning loyalty visitors, dwell times, and product categories sales distributions.

---

## 📁 Directory Architecture

- **`app/`**: Core API backend server logic.
  - **`main.py`**: Initializes the FastAPI app, manages middleware, routes endpoints, and exposes analytics.
  - **`schemas.py`**: Pydantic input/output validation structures.
  - **`routers/`**: Exposes REST endpoints (`vision.py`, `nlp.py`, `chatbot.py`).
  - **`services/`**: Encapsulates model inference workflows (`cv_service.py`, `nlp_service.py`, `chatbot_service.py`).
  - **`models/`**: Folder for trained binary weights (`*.h5`, `*.pkl`).
- **`data/`**: Raw dataset configuration templates and FAQ intents (`intents.json`).
- **`notebooks/`**: Modular Jupyter notebooks demonstrating model training pipelines.
- **`tests/`**: Automated pytest validation scripts for regression checks.
- **`dashboard.py`**: Streamlit frontend interface.

---

## 🔧 Installation & Set Up

### 1. Environment Activation
Ensure Python 3.10+ is installed on your system. Navigate to the project root directory and set up a virtual environment:
```powershell
# Create virtual environment
python -m venv virtual_env

# Activate on Windows (PowerShell)
.\virtual_env\Scripts\Activate.ps1
```

### 2. Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 3. Model Training (Optional)
If you wish to train the models from scratch on real datasets (RPC, LFW, Amazon Reviews):
* Load the notebooks inside the `notebooks/` directory.
* Upload the notebooks to Kaggle, attach the relevant datasets, and run with a **GPU T4** accelerator enabled.
* Download the output files and place them in the `app/models/` directory.

---

## 🏃 Running the Application

Start the backend server and the frontend dashboard in separate terminal tabs:

### Step A: Start the FastAPI Backend
```powershell
.\virtual_env\Scripts\python.exe -m uvicorn app.main:app --reload
```
* **API Home:** `http://127.0.0.1:8000`
* **Swagger API Documentation:** `http://127.0.0.1:8000/docs` (inspect endpoints and run mock tests directly through the browser).

### Step B: Start the Streamlit Web Dashboard
```powershell
.\virtual_env\Scripts\streamlit.exe run dashboard.py
```
* **Web Interface:** `http://localhost:8501`

---

## 🧪 Running Automated Tests

A comprehensive endpoint regression test suite is provided. To execute the tests:
```powershell
.\virtual_env\Scripts\python.exe -m pytest
```
*All endpoints are validated using mock file payloads and text inputs to verify correct status code responses.*
