# 🎓 AI-Powered Answer Sheet Validator

> Upload a photo of a student's answer sheet → OCR extracts the text → Groq API AI grades it and returns structured feedback.

---

## 📋 What It Does

| Step | Technology | Description |
|------|-----------|-------------|
| 1 | **Tesseract OCR** | Reads handwritten/printed text from the uploaded image |
| 2 | **Groq API** | Compares student answers against the answer key and scores them |
| 3 | **Flask** | Serves the web UI and REST `/grade` endpoint |
| 4 | **MLflow** | Logs every grading experiment (score, subject, filename) |
| 5 | **DVC** | Versions training data and model artifacts |

---

## 🛠️ Tech Stack

- **Backend:** Python 3.11, Flask 3.0, Werkzeug 3.0
- **OCR:** pytesseract 0.3.13, Pillow 10.4
- **AI:** Groq SDK (groq 0.x)
- **MLOps:** MLflow 2.15, DVC 3.51
- **Testing:** pytest 8.3, pytest-cov 5.0
- **CI/CD:** GitHub Actions
- **Containerisation:** Docker (multi-stage build)

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract#installing-tesseract) installed and on your `PATH`
- A [Groq Cloud](https://console.groq.com/) API key

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/answer-sheet-validator.git
cd answer-sheet-validator

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set GROQ_API_KEY=<your_real_key>

# 5. Run the Flask development server
flask run
# Open http://127.0.0.1:5000
```

> ⚠️ **SECURITY — Never share your `.env` file.**  
> It is listed in `.gitignore` and must never be committed to Git or shared with anyone. Rotate your API key immediately if it is accidentally exposed.

---

## 🧪 Running Tests

```bash
# Run all tests with coverage report
pytest tests/ --cov=model --cov-report=term-missing --cov-fail-under=70 -v
```

Coverage must stay above **70%** (enforced in CI).

---

## 🐳 Running with Docker

```bash
# Build the image
docker build -t answer-sheet-validator .

# Run the container (pass your API key at runtime — never bake it into the image)
docker run -p 5000:5000 \
  -e GROQ_API_KEY=your_real_key \
  answer-sheet-validator

# Open http://localhost:5000
```

---

## 📊 MLflow Dashboard

Every grading request automatically logs:
- Subject name
- Image filename
- Score (0–100)

```bash
# Launch the MLflow UI
mlflow ui

# Open http://127.0.0.1:5000 (or :8080 if Flask is already running)
mlflow ui --port 8080
```

---

## 📁 Project Structure

```
answer-sheet-validator/
├── app/                    # Flask application
│   ├── __init__.py         # App factory
│   ├── main.py             # Routes & upload handling
│   └── templates/
│       └── index.html      # Teacher upload UI
├── model/
│   ├── __init__.py
│   └── grader.py           # OCR + Groq grading pipeline
├── tests/
│   └── test_grader.py      # Unit tests (fully mocked)
├── data/answer_keys/       # Sample JSON answer keys
├── .github/workflows/      # GitHub Actions CI
├── .env.example            # Environment variable template
├── .gitignore
├── Dockerfile              # Multi-stage production image
├── requirements.txt
└── README.md
```

---

## 🔒 Security Notes (OWASP Top 10 Aware)

- **No hardcoded credentials** — all secrets loaded from environment variables
- **`.env` is gitignored** — it can never be accidentally committed
- **File upload validation** — only PNG/JPG accepted, max 5 MB, temp files deleted after processing
- **Non-root Docker user** — container runs as an unprivileged system user
- **Input sanitisation** — filenames are sanitised via `werkzeug.utils.secure_filename`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: describe your change"`
4. Push and open a Pull Request against `main`

All PRs must pass CI (lint + test coverage ≥ 70%).
