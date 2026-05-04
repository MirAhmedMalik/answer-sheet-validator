# =============================================================================
# tests/test_grader.py — Unit tests for model/grader.py
# All external calls (Groq API, OCR, MLflow) are fully mocked.
# Tests never touch the internet or require a real API key.
# Run with: pytest tests/ -v
# =============================================================================

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants matching grader.py — used to keep tests in sync
# ---------------------------------------------------------------------------
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]
INVALID_EXTENSIONS = [".pdf", ".docx", ".txt", ".gif"]

MOCK_OCR_TEXT = "The capital of France is Paris."
MOCK_SCORE = 8
MOCK_FEEDBACK = "Good answer! You correctly identified Paris as the capital of France."
MOCK_GROQ_JSON = json.dumps({"score": MOCK_SCORE, "feedback": MOCK_FEEDBACK})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_pil_image():
    """Return a MagicMock that works as a PIL Image context manager."""
    img = MagicMock()
    img.__enter__ = MagicMock(return_value=img)
    img.__exit__ = MagicMock(return_value=False)
    return img


def _make_groq_mock(response_text: str) -> MagicMock:
    """Build a mock Groq client whose chat completions return response_text."""
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_chat = MagicMock()
    mock_chat.completions.create.return_value = mock_completion

    mock_client = MagicMock()
    mock_client.chat = mock_chat
    return mock_client


# ===========================================================================
# 1. validate_image_path — extension checks
# ===========================================================================

class TestValidateImagePath:
    """Tests for the validate_image_path function."""

    @pytest.mark.parametrize("ext", VALID_EXTENSIONS)
    def test_allowed_extension_accepted(self, ext: str, tmp_path):
        """Confirm that .jpg, .jpeg, and .png pass validation without raising."""
        from model.grader import validate_image_path

        fake_file = tmp_path / f"sheet{ext}"
        fake_file.write_text("dummy")  # File must exist for path check
        # Should NOT raise
        validate_image_path(str(fake_file))

    @pytest.mark.parametrize("ext", INVALID_EXTENSIONS)
    def test_invalid_extension_rejected(self, ext: str, tmp_path):
        """Confirm that non-image extensions (.pdf etc.) raise ValueError."""
        from model.grader import validate_image_path

        fake_file = tmp_path / f"sheet{ext}"
        fake_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file type"):
            validate_image_path(str(fake_file))

    def test_missing_file_raises_file_not_found(self):
        """Confirm FileNotFoundError is raised when file does not exist."""
        from model.grader import validate_image_path

        with pytest.raises(FileNotFoundError, match="Image file not found"):
            validate_image_path("/nonexistent/path/image.jpg")


# ===========================================================================
# 2. extract_text — OCR
# ===========================================================================

class TestExtractText:
    """Tests for the extract_text function."""

    @patch("model.grader.pytesseract.image_to_string",
           return_value=f"  {MOCK_OCR_TEXT}  ")
    @patch("model.grader.Image.open", return_value=_mock_pil_image())
    def test_returns_stripped_text(self, mock_open, mock_ocr):
        """OCR output must be stripped of leading/trailing whitespace."""
        from model.grader import extract_text

        result = extract_text("fake/image.jpg")
        assert result == MOCK_OCR_TEXT


# ===========================================================================
# 3. build_prompt — prompt construction
# ===========================================================================

class TestBuildPrompt:
    """Tests for the build_prompt function."""

    def test_prompt_contains_question_and_answer(self):
        """Prompt must embed the question, correct answer, and student text."""
        from model.grader import build_prompt

        prompt = build_prompt(
            question="What is 2+2?",
            correct_answer="4",
            student_text="The answer is 4.",
        )
        assert "What is 2+2?" in prompt
        assert "4" in prompt
        assert "The answer is 4." in prompt

    def test_prompt_requests_json_output(self):
        """Prompt must instruct the Groq model to return JSON only."""
        from model.grader import build_prompt

        prompt = build_prompt("Q", "A", "S")
        assert "JSON" in prompt
        assert "score" in prompt
        assert "feedback" in prompt


# ===========================================================================
# 4. grade_answer — integration (all external calls mocked)
# ===========================================================================

class TestGradeAnswer:
    """Integration-level tests for the public grade_answer function."""

    @patch("model.grader.log_to_mlflow")
    @patch("model.grader._groq_client",
           new_callable=lambda: _make_groq_mock(MOCK_GROQ_JSON))
    @patch("model.grader.pytesseract.image_to_string",
           return_value=MOCK_OCR_TEXT)
    @patch("model.grader.Image.open", return_value=_mock_pil_image())
    def test_grade_answer_returns_required_keys(
        self, mock_open, mock_ocr, mock_client, mock_log, tmp_path
    ):
        """grade_answer must return a dict with extracted_text, score, feedback."""
        from model.grader import grade_answer

        img = tmp_path / "sheet.jpg"
        img.write_bytes(b"fake-image-data")

        result = grade_answer(
            image_path=str(img),
            question="What is the capital of France?",
            correct_answer="Paris",
        )

        assert "extracted_text" in result, "Missing key: extracted_text"
        assert "score" in result, "Missing key: score"
        assert "feedback" in result, "Missing key: feedback"

    @patch("model.grader.log_to_mlflow")
    @patch("model.grader._groq_client",
           new_callable=lambda: _make_groq_mock(MOCK_GROQ_JSON))
    @patch("model.grader.pytesseract.image_to_string",
           return_value=MOCK_OCR_TEXT)
    @patch("model.grader.Image.open", return_value=_mock_pil_image())
    def test_score_is_within_range(
        self, mock_open, mock_ocr, mock_client, mock_log, tmp_path
    ):
        """Score must be an integer between 0 and 10 inclusive."""
        from model.grader import grade_answer

        img = tmp_path / "sheet.jpeg"
        img.write_bytes(b"fake-image-data")

        result = grade_answer(
            image_path=str(img),
            question="What is the capital of France?",
            correct_answer="Paris",
        )

        score = result.get("score")
        assert isinstance(score, int), f"Score must be int, got {type(score)}"
        assert 0 <= score <= 10, f"Score {score} is out of valid range [0, 10]"

    @patch("model.grader._groq_client", None)
    def test_grade_answer_no_api_key_returns_error(self, tmp_path):
        """When GROQ_API_KEY is missing, grade_answer must return an error dict."""
        from model.grader import grade_answer

        img = tmp_path / "sheet.png"
        img.write_bytes(b"fake-image-data")

        with patch("model.grader.pytesseract.image_to_string", return_value=MOCK_OCR_TEXT), \
                patch("model.grader.Image.open", return_value=_mock_pil_image()):
            result = grade_answer(
                image_path=str(img),
                question="Test question",
            )

        assert "error" in result

    def test_grade_answer_invalid_extension_raises(self, tmp_path):
        """A .pdf file must raise ValueError before any API call is made."""
        from model.grader import grade_answer

        fake_pdf = tmp_path / "sheet.pdf"
        fake_pdf.write_bytes(b"not-an-image")

        with pytest.raises(ValueError, match="Unsupported file type"):
            grade_answer(image_path=str(fake_pdf), question="Any question")

    def test_grade_answer_missing_file_raises(self):
        """A non-existent path must raise FileNotFoundError."""
        from model.grader import grade_answer

        with pytest.raises(FileNotFoundError):
            grade_answer(image_path="/does/not/exist.jpg", question="Q")

    @patch("model.grader.log_to_mlflow")
    @patch("model.grader._groq_client",
           new_callable=lambda: _make_groq_mock(MOCK_GROQ_JSON))
    @patch("model.grader.Image.open", return_value=_mock_pil_image())
    @patch("model.grader.pytesseract.image_to_string", return_value="")
    def test_empty_ocr_returns_error_dict(
            self, mock_ocr, mock_open, mock_client, mock_log, tmp_path):
        """If OCR returns empty text, grade_answer must return an error dict."""
        from model.grader import grade_answer

        img = tmp_path / "blank.jpg"
        img.write_bytes(b"blank")

        result = grade_answer(image_path=str(img), question="Any question")
        assert "error" in result
        assert result["extracted_text"] == ""
