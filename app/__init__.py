# app/__init__.py — Flask application factory.
# Creates the app, loads config from environment, and wires everything together.

import os

from flask import Flask


def create_app() -> Flask:
    """Create and return a configured Flask application instance."""
    app = Flask(__name__, template_folder="templates")

    # Secret key for session signing — must come from environment in production
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(32))

    # Reject uploads larger than 5 MB before they even reach our route handler
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

    # Import here to avoid circular imports (app package isn't fully loaded yet)
    from app.main import main_bp, _add_security_headers  # noqa: PLC0415

    app.register_blueprint(main_bp)

    # Attach OWASP security headers to every response
    app.after_request(_add_security_headers)

    return app
