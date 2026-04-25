# =============================================================================
# app/main.py — Flask Web Application for AI Answer Sheet Validator (Phase 3)
# Provides two routes: GET / (upload page) and POST /grade (grading endpoint).
# SECURITY: OWASP A01/A03/A05 aware — validated uploads, security headers,
# no hardcoded secrets, no stack traces exposed to client.
# =============================================================================

import logging
import os
import uuid

from flask import Blueprint, Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from model.grader import grade_answer

# ── Logging setup (INFO level, no print statements) ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
UPLOAD_FOLDER: str = "uploads"
MAX_QUESTION_LEN: int = 500
MAX_ANSWER_LEN: int = 2000

# ── Blueprint ─────────────────────────────────────────────────────────────────
main_bp = Blueprint("main", __name__)


# =============================================================================
# Helper functions
# =============================================================================

def validate_upload(file) -> tuple[bool, str]:
    """
    Validate an uploaded file object for extension and real image content.

    Checks:
        1. Filename is not empty.
        2. File extension is in ALLOWED_EXTENSIONS (.jpg/.jpeg/.png).
        3. File can be opened by Pillow (confirms it is a real image, not a
           renamed non-image file — OWASP A03 defence-in-depth).

    Args:
        file: A Werkzeug ``FileStorage`` object from ``request.files``.

    Returns:
        A tuple ``(is_valid: bool, error_message: str)``.
        ``error_message`` is an empty string when ``is_valid`` is True.
    """
    if not file or file.filename == "":
        return False, "No file selected."

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Invalid file type '{ext}'. "
            "Only .jpg, .jpeg, and .png images are accepted."
        )

    # Verify actual image content with Pillow (not just trusting the extension)
    try:
        file.stream.seek(0)
        with Image.open(file.stream) as img:
            img.verify()  # Raises if not a valid image
        file.stream.seek(0)  # Rewind for later saving
    except UnidentifiedImageError:
        return False, "Uploaded file is not a valid image."
    except Exception:
        return False, "Could not read uploaded file. Please try again."

    return True, ""


def cleanup_file(path: str) -> None:
    """
    Delete a temporary uploaded file from disk.

    Student data must never be persisted after grading. This function is
    called in a ``finally`` block to guarantee cleanup even on errors.

    Args:
        path: Absolute or relative path to the file to delete.
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info("Cleaned up temporary file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete temporary file '%s': %s", path, exc)


def _add_security_headers(response):
    """
    Attach OWASP-recommended security headers to every HTTP response.

    Args:
        response: A Flask ``Response`` object.

    Returns:
        The same response object with security headers added.
    """
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

    Args:
        value:      Raw string value from ``request.form``.
        field_name: Human-readable name used in error messages.
        max_len:    Maximum allowed character length.

    Returns:
        A tuple ``(cleaned_value: str, error: str)``.
        ``error`` is empty when validation passes.
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


# =============================================================================
# Routes
# =============================================================================

@main_bp.route("/", methods=["GET"])
def index():
    """
    Serve the teacher upload page.

    Returns:
        Rendered ``index.html`` template with HTTP 200.
    """
    logger.info("GET / — upload page requested")
    return render_template("index.html")


@main_bp.route("/grade", methods=["POST"])
def grade():
    """
    Accept a multipart form upload, run OCR + AI grading, return JSON.

    Form fields expected:
        - ``file``           — image file (.jpg/.jpeg/.png, max 5 MB)
        - ``question``       — exam question (required, max 500 chars)
        - ``correct_answer`` — model answer  (required, max 2000 chars)

    Returns:
        JSON with keys ``extracted_text``, ``score``, ``feedback`` on success.
        JSON with key ``error`` and appropriate HTTP status on failure.
    """
    logger.info("POST /grade — grading request received")
    file_path: str = ""

    try:
        # ── Validate file ──────────────────────────────────────────────── #
        uploaded_file = request.files.get("file")
        is_valid, file_error = validate_upload(uploaded_file)
        if not is_valid:
            logger.warning("Upload validation failed: %s", file_error)
            return jsonify({"error": file_error}), 400

        # ── Validate text fields ───────────────────────────────────────── #
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

        # ── Save file safely ───────────────────────────────────────────── #
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        safe_name = f"{uuid.uuid4().hex}_{secure_filename(uploaded_file.filename)}"
        file_path = os.path.join(UPLOAD_FOLDER, safe_name)
        uploaded_file.save(file_path)
        logger.info("File saved temporarily: %s", safe_name)

        # ── Grade ──────────────────────────────────────────────────────── #
        result = grade_answer(
            image_path=file_path,
            question=question,
            correct_answer=correct_answer,
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
        # Never expose internal details to the client (OWASP A05)
        logger.exception("Unexpected error during grading")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500

    finally:
        cleanup_file(file_path)


# =============================================================================
# App factory
# =============================================================================

def create_app() -> Flask:
    """
    Flask application factory.

    Loads configuration from environment variables, registers the blueprint,
    and attaches the security-header after-request hook.

    Returns:
        A configured Flask application instance.
    """
    app = Flask(__name__, template_folder="templates")

    # Load secret key from environment — NEVER hardcode this value
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(32))

    # Enforce 5 MB upload limit at the Flask level (returns 413 automatically)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

    app.register_blueprint(main_bp)
    app.after_request(_add_security_headers)

    return app


# ── Entry point (dev only — use gunicorn/waitress in production) ──────────────
if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(debug=False, host="0.0.0.0", port=5000)
