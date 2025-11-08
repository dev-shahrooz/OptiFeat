"""Flask view functions for the OptiFeat dashboard."""
from __future__ import annotations

import json
from flask import (
    Response,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from optifeat.config import HISTORY_PAGE_SIZE, UPLOAD_DIR
from optifeat.services.pipeline import OptimizationPipeline
from optifeat.storage import database
from optifeat.web.routes import bp


ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/", methods=["GET", "POST"])
@bp.route("/dashboard", methods=["GET", "POST"])
def dashboard() -> Response:
    pipeline_result = None
    history = database.fetch_history(limit=HISTORY_PAGE_SIZE)
    if request.method == "POST":
        uploaded = request.files.get("dataset")
        target_column = request.form.get("target_column") or "target"
        time_budget_raw = request.form.get("time_budget") or "5"

        if not uploaded or uploaded.filename == "":
            flash("لطفاً فایل داده را انتخاب کنید.", "error")
            return render_template(
                "dashboard.html", history=history, result=pipeline_result
            )

        if not allowed_file(uploaded.filename):
            flash("فقط فایل‌های CSV پشتیبانی می‌شوند.", "error")
            return render_template(
                "dashboard.html", history=history, result=pipeline_result
            )

        try:
            time_budget = float(time_budget_raw)
        except ValueError:
            flash("بودجه زمانی باید یک عدد باشد.", "error")
            return render_template(
                "dashboard.html", history=history, result=pipeline_result
            )

        filename = secure_filename(uploaded.filename)
        dataset_path = UPLOAD_DIR / filename
        uploaded.save(dataset_path)

        pipeline = OptimizationPipeline()
        try:
            pipeline_result = pipeline.run(
                dataset_path,
                target_column=target_column,
                time_budget=time_budget,
            )
            flash("بهینه‌سازی با موفقیت انجام شد.", "success")
        except Exception as exc:  # pylint: disable=broad-except
            current_app.logger.exception("Pipeline execution failed: %s", exc)
            flash(f"خطا در اجرای مدل: {exc}", "error")

        history = database.fetch_history(limit=HISTORY_PAGE_SIZE)

    return render_template("dashboard.html", history=history, result=pipeline_result)


@bp.route("/history")
def history_view() -> Response:
    history = database.fetch_history(limit=100)
    return render_template("history.html", history=history)


@bp.route("/downloads/<int:run_id>/features")
def download_features(run_id: int) -> Response:
    run = database.fetch_run(run_id)
    if not run:
        flash("رکورد مورد نظر یافت نشد.", "error")
        return redirect(url_for("dashboard.dashboard"))

    content = json.dumps({"selected_features": run["selected_features"]}, ensure_ascii=False)
    path = UPLOAD_DIR / f"run_{run_id}_features.json"
    path.write_text(content, encoding="utf-8")
    return send_file(path, as_attachment=True)
