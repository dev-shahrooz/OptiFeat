"""Route registration for the OptiFeat dashboard."""
from __future__ import annotations

from flask import Blueprint

bp = Blueprint("dashboard", __name__, template_folder="templates")

# The actual route handlers are imported to register with the blueprint.
from optifeat.web import views  # noqa: E402  pylint: disable=wrong-import-position
