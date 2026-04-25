# =============================================================================
# app/__init__.py — Flask application factory.
# Initialises the Flask app and registers blueprints/extensions.
# =============================================================================

from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__, template_folder="templates")

    # Import routes here to avoid circular imports
    from app.main import main_bp  # noqa: PLC0415

    app.register_blueprint(main_bp)

    return app
