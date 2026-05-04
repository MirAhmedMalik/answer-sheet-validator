# =============================================================================
# model/grader.py — AI Grading Engine (Groq / LLaMA 3 Edition)
# Pipeline: validate → Tesseract OCR → Groq LLaMA3 grading → MLflow logging.
# Gemini replaced with Groq API (groq Python library) to work in all regions.
# SECURITY: OWASP-aware — no hardcoded keys, validated extensions, no PII logs.
# =============================================================================

import json
import logging
import os
import re

import mlflow
import pytesseract
from dotenv import load_dotenv
from groq import Groq
from PIL import Image

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
MODEL_NAME: str               = "llama-3.1-8b-instant"
MLFLOW_EXPERIMENT: str        = "answer-sheet-validation"

# ── Ensure Tesseract is in PATH (Windows only — Linux/Docker uses system PATH) ─
import sys as _sys
if _sys.platform == "win32":
    _TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
    if _TESSERACT_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")
pytesseract.pytesseract.tesseract_cmd = "tesseract"

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Groq client (key from env — never hardcoded) ──────────────────────────────
_GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
_groq_client: Groq | None = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None


# =============================================================================
# Validation
# =============================================================================

def validate_image_path(image_path: str) -> None:
    """
    Validate that the image file exists and has an allowed extension.

    Args:
        image_path: Path to the candidate image file.

    Raises:
        FileNotFoundError: File does not exist at the given path.
        ValueError:        File extension is not in ALLOWED_EXTENSIONS.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image file not found: '{image_path}'. "
            "Provide a valid path to a .jpg, .jpeg, or .png file."
        )
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


# =============================================================================
# OCR
# =============================================================================

def extract_text(image_path: str) -> str:
    """
    Run Tesseract OCR on an image and return the extracted text.

    Args:
        image_path: Path to a valid image file.

    Returns:
        Stripped string of text found in the image.
    """
    with Image.open(image_path) as img:
        text: str = pytesseract.image_to_string(img)
    return text.strip()


# =============================================================================
# Prompt builder
# =============================================================================

def build_prompt(question: str, correct_answer: str, student_text: str) -> str:
    """
    Build a structured grading prompt for the LLaMA3 model.

    Args:
        question:       The exam question being assessed.
        correct_answer: The expected model answer.
        student_text:   OCR-extracted text from the student's sheet.

    Returns:
        A formatted prompt string instructing the model to return only JSON.
    """
    return (
        "You are an expert, highly accurate academic grader.\n\n"
        "First, analyze the QUESTION and RUBRIC below to automatically deduce the subject (e.g., Math, Physics, English, History). "
        "Adapt your grading strictness to that subject (e.g., Math requires correct final numbers or steps, History/English focus on concepts and semantic meaning).\n\n"
        f"QUESTION:\n{question}\n\n"
        f"TEACHER'S GRADING RUBRIC / CORRECT ANSWER:\n{correct_answer}\n\n"
        f"STUDENT'S WRITTEN ANSWER (extracted via OCR, may contain typos):\n{student_text}\n\n"
        "INSTRUCTIONS FOR EVALUATION:\n"
        "1. Understand the student's core conceptual meaning. Ignore minor spelling mistakes or OCR artifacts (like random punctuation or misread letters).\n"
        "2. SEMANTIC MATCHING: DO NOT penalize the student if they use different words or synonyms than the teacher. If the underlying logic, physics, math end-result, or conceptual meaning is the same as the rubric, award full points!\n"
        "3. GRADING & SCORING (Maximum 10 Points):\n"
        "   - IF THE RUBRIC GIVES SPECIFIC POINTS: You MUST strictly add up the points exactly according to the teacher's point distribution (e.g., if they say '4 pts for X, 6 pts for Y', score strictly out of those rules).\n"
        "   - IF NO POINT BREAKDOWN IS GIVEN: Use your own expert judgment to assign a fair score out of 10 based on how closely the student's meaning matches the teacher's expected answer (10 = perfectly accurate, 5 = partially correct, 0 = entirely wrong).\n"
        "4. Write 2-3 sentences of clear feedback explaining exactly what they got right, and where they went wrong.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "Respond with ONLY valid JSON. Absolutely no markdown fences (like ```json), no intro text, no trailing text.\n"
        'Exactly like this: {"score": <integer>, "feedback": "<string>"}'
    )


# =============================================================================
# Groq API call
# =============================================================================

def call_groq(prompt: str) -> dict:
    """
    Send the grading prompt to Groq (LLaMA3) and parse the JSON response.

    Args:
        prompt: Fully constructed grading prompt string.

    Returns:
        Dict with keys ``score`` (int) and ``feedback`` (str).

    Raises:
        RuntimeError: GROQ_API_KEY environment variable is not set.
        ValueError:   Response cannot be parsed as valid JSON.
    """
    if not _groq_client:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file: GROQ_API_KEY=your_key_here"
        )

    chat_completion = _groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an academic grader. "
                    "Always respond with valid JSON only — no markdown, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,   # Low temp for consistent, deterministic grading
        max_tokens=512,
    )

    raw: str = chat_completion.choices[0].message.content or ""

    # Extract JSON even if model wraps it in markdown fences
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(
            f"Groq returned a response with no JSON object: {raw!r}"
        )

    return json.loads(json_match.group())


# =============================================================================
# MLflow logging
# =============================================================================

def log_to_mlflow(question_length: int, score: int) -> None:
    """
    Log one grading session to MLflow.

    Params : question_length, model_name
    Metrics: score
    Note   : Raw student text intentionally excluded (student privacy).

    Args:
        question_length: Character count of the exam question.
        score:           Integer score assigned by LLaMA3 (0-10).
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_param("question_length", question_length)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_metric("score", score)


# =============================================================================
# Public API
# =============================================================================

def grade_answer(
    image_path: str = "",
    question: str = "",
    correct_answer: str = "",
    student_text_input: str = "",
) -> dict:
    """
    Full grading pipeline: validate → OCR → Groq grade → MLflow log → result.
    If student_text_input is provided, bypasses image validation and OCR completely.
    """
    if student_text_input:
        student_text = student_text_input
    else:
        # Step 1 — Validate (raises on bad path / extension)
        validate_image_path(image_path)

        # Step 2 — OCR
        try:
            student_text = extract_text(image_path)
        except Exception as exc:
            logger.error("OCR failed: %s", exc)
            return {"error": f"OCR failed: {exc}", "extracted_text": "", "score": 0, "feedback": ""}

        if not student_text:
            return {
                "error": "OCR returned no text. Check image quality or try a clearer photo.",
                "extracted_text": "",
                "score": 0,
                "feedback": "",
            }

    # Step 3 — Groq grading
    try:
        prompt        = build_prompt(question, correct_answer, student_text)
        groq_result   = call_groq(prompt)
        score: int    = int(groq_result["score"])
        feedback: str = str(groq_result["feedback"])
    except Exception as exc:
        logger.error("Groq grading failed: %s", exc)
        return {
            "error": f"Grading failed: {exc}",
            "extracted_text": student_text,
            "score": 0,
            "feedback": "",
        }

    # Step 4 — MLflow (non-fatal)
    try:
        log_to_mlflow(question_length=len(question), score=score)
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): %s", exc)

    # Step 5 — Return
    return {
        "extracted_text": student_text,
        "score": score,
        "feedback": feedback,
    }
