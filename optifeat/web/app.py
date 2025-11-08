"""Flask application factory for the OptiFeat dashboard."""
from __future__ import annotations

from flask import Flask

from optifeat.config import SECRET_KEY
from optifeat.storage.database import initialize_database
from optifeat.web.routes import dashboard


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY

    initialize_database()

    app.register_blueprint(dashboard.bp)
    return app


if __name__ == "__main__":
    application = create_app()
    application.run(debug=True)
