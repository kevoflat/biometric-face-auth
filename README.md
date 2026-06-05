# 👁️ Biometric Face Recognition System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-orange)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**A full-stack biometric authentication system for Student & Employee Identification.**  
Real-time face registration, matching, and access logging — fully local, no cloud required.

[Features](#-features) · [Tech Stack](#-tech-stack) · [Installation](#-installation) · [Usage](#-usage) · [Architecture](#-architecture) · [Configuration](#-configuration)

</div>

---

## 🎯 Overview

A production-ready biometric face recognition system built with modern Python tools. Enroll students or employees with a single face capture, then authenticate them in real-time using state-of-the-art **ArcFace embeddings** — all running locally with zero cloud dependencies and zero data leaving your machine.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📸 **Face Registration** | Enroll users via webcam or photo upload with personal details |
| 🔐 **Real-time Authentication** | Face matching with confidence score and visual gauge |
| 👥 **User Management** | View, search, and delete registered users |
| 📋 **Access Logs** | Full audit trail of all GRANTED / DENIED attempts |
| 📥 **CSV Export** | Download access logs for reporting |
| 🔒 **Privacy-first** | All processing is local — no data sent to external servers |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Face Recognition | InsightFace (ArcFace / ONNX) | 512-d face embeddings + cosine similarity |
| Backend API | FastAPI + Uvicorn | REST endpoints for register / authenticate / logs |
| Frontend | Streamlit | Interactive web UI with webcam support |
| Database | SQLite | User data, face embeddings, access logs |
| Image Processing | OpenCV + Pillow | Image decoding and preprocessing |

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Webcam (for live capture) or face image files

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/kevoflat/biometric-face-auth.git
cd biometric-face-auth

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** On first use, InsightFace automatically downloads the `buffalo_sc` model (~80 MB) and caches it at `~/.insightface/`. Subsequent runs are instant.

---

## 🚀 Usage

You need **two terminals running simultaneously**.

**Terminal 1 — Backend API:**
```bash
cd biometric/backend
python main.py
# API running at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

**Terminal 2 — Frontend UI:**
```bash
cd biometric/frontend
streamlit run app.py
# App running at http://localhost:8501
```

Open `http://localhost:8501` in your browser.

### Workflow

1. **Register** — Enter name, ID, role, department → capture face → click Register
2. **Authenticate** — Capture face → click Verify Identity → see GRANTED/DENIED + confidence
3. **Users** — Browse all enrolled users, delete if needed
4. **Access Logs** — View authentication history, export to CSV

---

## 🏗️ Architecture

```
Browser (Streamlit UI :8501)
         │
         │  HTTP POST (base64 image + metadata)
         ▼
FastAPI Backend (:8000)
         │
         ├── /register      → extract_embedding() → store in SQLite
         ├── /authenticate  → extract_embedding() → find_match() → log result
         ├── /users         → read users table
         ├── /logs          → read access_logs table
         └── /delete        → remove user + embeddings
         │
         ▼
face_engine.py
  ├── InsightFace (buffalo_sc)
  │     ├── det_500m.onnx     — face detection
  │     └── w600k_mbf.onnx    — ArcFace recognition (512-d embedding)
  ├── L2-normalise embedding
  └── Cosine similarity → best match vs threshold
         │
         ▼
SQLite (data/biometric.db)
  ├── users            (id, name, student_id, role, department, ...)
  ├── face_embeddings  (user_id, embedding JSON, created_at)
  └── access_logs      (user_id, status, confidence, timestamp)
```

---

## ⚙️ Configuration

Tune recognition sensitivity in `biometric/backend/face_engine.py`:

```python
COSINE_THRESHOLD = 0.35  # default — adjust to your environment

# Higher (e.g. 0.45) → stricter, fewer false positives
# Lower  (e.g. 0.25) → more lenient, fewer false negatives
```

For faster inference at the cost of detection accuracy:
```python
app.prepare(ctx_id=0, det_size=(160, 160))  # smaller → faster on CPU
```

---

## 📁 Project Structure

```
biometric-face-auth/
├── biometric/
│   ├── backend/
│   │   ├── main.py          # FastAPI app & all endpoints
│   │   ├── face_engine.py   # InsightFace embedding extraction & matching
│   │   ├── database.py      # SQLite CRUD operations
│   │   └── data/            # Database file (gitignored)
│   └── frontend/
│       └── app.py           # Streamlit multi-tab UI
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔐 Privacy & Security

- All face embeddings are stored locally in SQLite — never uploaded anywhere
- The `data/` directory (containing the `.db` file) is gitignored to prevent accidental biometric data commits
- Embeddings are 512-d numerical vectors — not raw images

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ by **[Kelvin Mwangi](https://github.com/kevoflat)**

⭐ Star this repo if you found it useful!

</div>
