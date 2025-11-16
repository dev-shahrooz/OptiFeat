"""Flask application factory for the OptiFeat dashboard."""
from flask import Flask

from optifeat.config import SECRET_KEY
from optifeat.storage.database import initialize_database
from optifeat.web.routes import bp as dashboard_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY

    # ایجاد دیتابیس (اگر وجود نداشته باشد)
    initialize_database()

    # ثبت بلوپرینت داشبورد
    app.register_blueprint(dashboard_bp)
    return app



if __name__ == "__main__":
    application = create_app()
    application.run(debug=True)
