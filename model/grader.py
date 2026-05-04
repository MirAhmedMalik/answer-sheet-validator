# model/grader.py — AI grading engine.
# Pipeline: image → Tesseract OCR → Groq LLaMA3 grading → MLflow logging.
# No hardcoded credentials. Student text is never logged (privacy).

import json
import logging
import os
import re
import sys

import mlflow
import pytesseract
from dotenv import load_dotenv
from groq import Groq
from PIL import Image

# Load environment variables from .env (no-op in production where env is set directly)
load_dotenv()

# Model and experiment names — change here to switch models globally
ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
MODEL_NAME: str = "llama-3.1-8b-instant"
MLFLOW_EXPERIMENT: str = "answer-sheet-validation"

# On Windows, Tesseract is usually not on PATH automatically — add it if needed.
# Linux and Docker find it via the system PATH, so this block is skipped there.
if sys.platform == "win32":
    _TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
    if _TESSERACT_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")
pytesseract.pytesseract.tesseract_cmd = "tesseract"

logger = logging.getLogger(__name__)

# Build the Groq client once at import time. If the key is missing, the client
# is None and call_groq() raises a clear RuntimeError instead of crashing here.
_GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
_groq_client: Groq | None = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None


def validate_image_path(image_path: str) -> None:
    """
    Check that the file exists and has an allowed image extension.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: Extension is not .jpg, .jpeg, or .png.
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


def extract_text(image_path: str) -> str:
    """Run Tesseract OCR on an image and return the extracted text (stripped)."""
    with Image.open(image_path) as img:
        text: str = pytesseract.image_to_string(img)
    return text.strip()


def build_prompt(question: str, correct_answer: str, student_text: str) -> str:
    """
    Build the structured grading prompt sent to LLaMA3.

    Returns a string that instructs the model to return only a JSON object
    with 'score' (int, 0-10) and 'feedback' (str) keys.
    """
    return (
        "You are an expert, highly accurate academic grader.\n\n"
        "First, analyze the QUESTION and RUBRIC below to automatically deduce the subject "
        "(e.g., Math, Physics, English, History). "
        "Adapt your grading strictness to that subject (e.g., Math requires correct final "
        "numbers or steps, History/English focus on concepts and semantic meaning).\n\n"
        f"QUESTION:\n{question}\n\n"
        f"TEACHER'S GRADING RUBRIC / CORRECT ANSWER:\n{correct_answer}\n\n"
        f"STUDENT'S WRITTEN ANSWER (extracted via OCR, may contain typos):\n{student_text}\n\n"
        "INSTRUCTIONS FOR EVALUATION:\n"
        "1. Understand the student's core conceptual meaning. Ignore minor spelling mistakes "
        "or OCR artifacts (like random punctuation or misread letters).\n"
        "2. SEMANTIC MATCHING: DO NOT penalize the student if they use different words or "
        "synonyms than the teacher. If the underlying logic, physics, math end-result, or "
        "conceptual meaning is the same as the rubric, award full points!\n"
        "3. GRADING & SCORING (Maximum 10 Points):\n"
        "   - IF THE RUBRIC GIVES SPECIFIC POINTS: You MUST strictly add up the points exactly "
        "according to the teacher's point distribution "
        "(e.g., if they say '4 pts for X, 6 pts for Y', score strictly out of those rules).\n"
        "   - IF NO POINT BREAKDOWN IS GIVEN: Use your own expert judgment to assign a fair "
        "score out of 10 based on how closely the student's meaning matches the teacher's "
        "expected answer (10 = perfectly accurate, 5 = partially correct, 0 = entirely wrong).\n"
        "4. Write 2-3 sentences of clear feedback explaining exactly what they got right, "
        "and where they went wrong.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "Respond with ONLY valid JSON. Absolutely no markdown fences (like ```json), "
        "no intro text, no trailing text.\n"
        'Exactly like this: {"score": <integer>, "feedback": "<string>"}'
    )


def call_groq(prompt: str) -> dict:
    """
    Send the grading prompt to Groq (LLaMA3) and return the parsed JSON response.

    Returns:
        Dict with 'score' (int) and 'feedback' (str).

    Raises:
        RuntimeError: GROQ_API_KEY is not set.
        ValueError: Response contains no parseable JSON object.
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
        temperature=0.2,  # Low temperature keeps scores consistent across repeated runs
        max_tokens=512,
    )

    raw: str = chat_completion.choices[0].message.content or ""

    # The model occasionally wraps output in ```json fences — strip them with regex
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Groq returned a response with no JSON object: {raw!r}")

    return json.loads(json_match.group())


def log_to_mlflow(question_length: int, score: int) -> None:
    """
    Log one grading session to MLflow.

    We log question_length and model_name as params, and score as a metric.
    Raw student text is intentionally excluded to protect student privacy.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_param("question_length", question_length)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_metric("score", score)


def grade_answer(
    image_path: str = "",
    question: str = "",
    correct_answer: str = "",
    student_text_input: str = "",
) -> dict:
    """
    Run the full grading pipeline and return a result dict.

    If student_text_input is provided, OCR is skipped entirely and that text
    is graded directly. Otherwise the image at image_path is OCR'd first.

    Returns a dict with keys: extracted_text, score, feedback.
    On failure, returns a dict with an 'error' key instead.
    """
    if student_text_input:
        student_text = student_text_input
    else:
        # Validate path and extension before touching any external service
        validate_image_path(image_path)

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

    # Send to Groq for AI grading
    try:
        prompt = build_prompt(question, correct_answer, student_text)
        groq_result = call_groq(prompt)
        score: int = int(groq_result["score"])
        feedback: str = str(groq_result["feedback"])
    except Exception as exc:
        logger.error("Groq grading failed: %s", exc)
        return {
            "error": f"Grading failed: {exc}",
            "extracted_text": student_text,
            "score": 0,
            "feedback": "",
        }

    # MLflow logging is non-fatal — a tracking failure never blocks the grading result
    try:
        log_to_mlflow(question_length=len(question), score=score)
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): %s", exc)

    return {
        "extracted_text": student_text,
        "score": score,
        "feedback": feedback,
    }
