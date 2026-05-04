# app/main.py — Flask routes and upload handling.
# GET /       — serves the upload page.
# POST /grade — receives the file/text, runs OCR + AI grading, returns JSON.

import logging
import os
import uuid

from flask import Blueprint, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from model.grader import grade_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
UPLOAD_FOLDER: str = "uploads"
MAX_QUESTION_LEN: int = 500
MAX_ANSWER_LEN: int = 2000

main_bp = Blueprint("main", __name__)


def validate_upload(file) -> tuple[bool, str]:
    """
    Validate a Werkzeug FileStorage upload.

    Checks the filename is present, the extension is allowed, and that
    Pillow can actually open it (catches renamed non-image files).

    Returns:
        (True, "") on success, or (False, error_message) on failure.
    """
    if not file or file.filename == "":
        return False, "No file selected."

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Invalid file type '{ext}'. "
            "Only .jpg, .jpeg, and .png images are accepted."
        )

    # Verify real image content — don't just trust the extension (OWASP A03)
    try:
        file.stream.seek(0)
        with Image.open(file.stream) as img:
            img.verify()
        file.stream.seek(0)  # rewind so the file can be saved afterwards
    except UnidentifiedImageError:
        return False, "Uploaded file is not a valid image."
    except Exception:
        return False, "Could not read uploaded file. Please try again."

    return True, ""


def cleanup_file(path: str) -> None:
    """
    Delete a temporary file from disk.

    Called in a finally block so student data is never left on the server,
    even if grading raises an exception.
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info("Cleaned up temporary file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete temporary file '%s': %s", path, exc)


def _add_security_headers(response):
    """Add OWASP-recommended security headers to every HTTP response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


def _validate_text_field(
    value: str | None,
    field_name: str,
    max_len: int,
) -> tuple[str, str]:
    """
    Strip and validate a required text form field.

    Returns:
        (cleaned_value, "") on success, or ("", error_message) on failure.
    """
    if not value or not value.strip():
        return "", f"'{field_name}' is required."
    cleaned = value.strip()
    if len(cleaned) > max_len:
        return "", (
            f"'{field_name}' must be {max_len} characters or fewer "
            f"(received {len(cleaned)})."
        )
    return cleaned, ""


@main_bp.route("/", methods=["GET"])
def index():
    """Serve the teacher upload page."""
    logger.info("GET / — upload page requested")
    return render_template("index.html")


@main_bp.route("/grade", methods=["POST"])
def grade():
    """
    Accept a multipart form upload, run OCR + AI grading, return JSON.

    Form fields:
        file           — image (.jpg/.jpeg/.png, max 5 MB)
        question       — exam question (required, max 500 chars)
        correct_answer — model answer / rubric (required, max 2000 chars)
        student_text_input — optional raw text; when provided, image is skipped

    Returns JSON with 'extracted_text', 'score', 'feedback' on success,
    or 'error' with an appropriate HTTP status on failure.
    """
    logger.info("POST /grade — grading request received")
    file_path: str = ""

    try:
        question, q_err = _validate_text_field(
            request.form.get("question"), "question", MAX_QUESTION_LEN
        )
        if q_err:
            return jsonify({"error": q_err}), 400

        correct_answer, a_err = _validate_text_field(
            request.form.get("correct_answer"), "correct_answer", MAX_ANSWER_LEN
        )
        if a_err:
            return jsonify({"error": a_err}), 400

        student_text_input = request.form.get("student_text_input", "").strip()
        uploaded_file = request.files.get("file")

        if student_text_input:
            if len(student_text_input) > 5000:
                return jsonify({"error": "Raw text input must be 5000 characters or fewer."}), 400
            file_path = ""
        else:
            is_valid, file_error = validate_upload(uploaded_file)
            if not is_valid:
                logger.warning("Upload validation failed: %s", file_error)
                return jsonify({"error": file_error}), 400

            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            safe_name = f"{uuid.uuid4().hex}_{secure_filename(uploaded_file.filename)}"
            file_path = os.path.join(UPLOAD_FOLDER, safe_name)
            uploaded_file.save(file_path)
            logger.info("File saved temporarily: %s", safe_name)

        result = grade_answer(
            image_path=file_path,
            question=question,
            correct_answer=correct_answer,
            student_text_input=student_text_input,
        )

        if "error" in result:
            logger.error("Grading returned error: %s", result["error"])
            return jsonify({"error": result["error"]}), 500

        logger.info("Grading complete — score: %s/10", result.get("score"))
        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return jsonify({"error": str(exc)}), 400

    except Exception:
        # Never let internal details leak to the client (OWASP A05)
        logger.exception("Unexpected error during grading")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500

    finally:
        cleanup_file(file_path)
