
import os
import re
import json
import sqlite3
import logging
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, send_file
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from services.detector import compare_documents
from services.extractor import extract_text
from services.gemini_service import explain_matches
from services.report import create_pdf


# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
DB_PATH = os.path.join(BASE_DIR, "semantiai.db")
SETTINGS_PATH = os.path.join(BASE_DIR, "plagiarism_settings.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "change-this-secret-before-deployment"
)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = 28800

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


# ------------------------------------------------
# DATABASE
# ------------------------------------------------

def db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_a TEXT,
                doc_b TEXT,
                score REAL,
                risk TEXT,
                result_json TEXT,
                created_at TEXT,
                user_id INTEGER
            )
        """)

        # Add user_id to an existing database without deleting old records.
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(comparisons)"
            ).fetchall()
        }

        if "user_id" not in columns:
            conn.execute(
                "ALTER TABLE comparisons ADD COLUMN user_id INTEGER"
            )

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_comparisons_user
            ON comparisons(user_id)
        """)


init_db()


# ------------------------------------------------
# LOGIN PROTECTION
# ------------------------------------------------

def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "info")
            return redirect(
                url_for("login", next=request.path)
            )
        return function(*args, **kwargs)
    return wrapper


def safe_next_url(target):
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("home")


# ------------------------------------------------
# REGISTRATION
# ------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if len(name) < 2 or len(name) > 80:
            flash("Enter a name between 2 and 80 characters.", "danger")
            return render_template("register.html")

        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Enter a valid email address.", "danger")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        try:
            with db() as conn:
                cursor = conn.execute("""
                    INSERT INTO users
                    (name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    name,
                    email,
                    generate_password_hash(password),
                    datetime.now().isoformat(timespec="seconds")
                ))

                user_id = cursor.lastrowid

            session.clear()
            session["user_id"] = user_id
            session["user_name"] = name
            session.permanent = True

            flash("Account created successfully!", "success")
            return redirect(url_for("home"))

        except sqlite3.IntegrityError:
            flash("This email is already registered. Please log in.", "warning")

        except Exception:
            app.logger.exception("Registration error")
            flash("Could not create your account. Try again.", "danger")

    return render_template("register.html")


# ------------------------------------------------
# LOGIN
# ------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with db() as conn:
            user = conn.execute("""
                SELECT * FROM users
                WHERE email = ? COLLATE NOCASE
            """, (email,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):

            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session.permanent = True

            flash("Welcome back!", "success")

            return redirect(
                safe_next_url(request.args.get("next"))
            )

        flash("Incorrect email or password.", "danger")

    return render_template("login.html")


# ------------------------------------------------
# LOGOUT
# ------------------------------------------------

@app.route("/logout", methods=["POST"])
@login_required
def logout():

    session.clear()
    flash("You have been logged out.", "success")

    return redirect(url_for("login"))


# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

DEFAULT_SETTINGS = {
    "similarity_threshold": 75,
    "detect_exact": True,
    "detect_partial": True,
    "detect_paraphrase": True,
    "detect_semantic": True,
    "comparison_level": "sentence",
    "highlight_matches": True,
    "report_matches": True,
    "report_scores": True,
    "report_recommendations": True,
    "delete_after_analysis": True
}


def load_settings():

    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_SETTINGS.copy()

    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as file:
            saved = json.load(file)

        settings = {**DEFAULT_SETTINGS, **saved}

        try:
            settings["similarity_threshold"] = max(
                10, min(100, int(settings["similarity_threshold"]))
            )
        except (ValueError, TypeError):
            settings["similarity_threshold"] = 75

        if settings["comparison_level"] not in (
            "sentence", "paragraph", "document"
        ):
            settings["comparison_level"] = "sentence"

        return settings

    except (OSError, ValueError, TypeError):
        app.logger.exception("Settings could not be loaded")
        return DEFAULT_SETTINGS.copy()


def save_settings(settings):

    temp_path = SETTINGS_PATH + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)

    os.replace(temp_path, SETTINGS_PATH)


# ------------------------------------------------
# FILE HELPERS
# ------------------------------------------------

def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def report_path(item_id):
    return os.path.join(
        REPORT_DIR,
        f"SemantiAi_Report_{item_id}.pdf"
    )


def remove_report(item_id):
    path = report_path(item_id)

    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        app.logger.exception("Could not delete PDF report")


def delete_record(item_id, user_id):

    with db() as conn:
        cursor = conn.execute("""
            DELETE FROM comparisons
            WHERE id = ? AND user_id = ?
        """, (item_id, user_id))

        deleted = cursor.rowcount > 0

    if deleted:
        remove_report(item_id)

    return deleted


# ------------------------------------------------
# HOME
# ------------------------------------------------

@app.route("/")
@login_required
def home():

    with db() as conn:
        total = conn.execute("""
            SELECT COUNT(*) FROM comparisons
            WHERE user_id = ?
        """, (session["user_id"],)).fetchone()[0]

    return render_template("index.html", total=total)


# ------------------------------------------------
# ABOUT
# ------------------------------------------------

@app.route("/about")
@login_required
def about():
    return render_template("about.html")


# ------------------------------------------------
# DOCUMENT COMPARISON
# ------------------------------------------------

@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare():

    if request.method == "GET":
        return render_template("upload.html")

    file_a = request.files.get("document_a")
    file_b = request.files.get("document_b")

    if (
        not file_a or not file_b
        or not file_a.filename or not file_b.filename
    ):
        flash("Please select both documents.", "danger")
        return redirect(url_for("compare"))

    if not allowed_file(file_a.filename) or not allowed_file(file_b.filename):
        flash("Only PDF, DOCX and TXT files are supported.", "danger")
        return redirect(url_for("compare"))

    name_a = secure_filename(file_a.filename)
    name_b = secure_filename(file_b.filename)

    if not name_a or not name_b:
        flash("Invalid filename.", "danger")
        return redirect(url_for("compare"))

    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

    path_a = os.path.join(UPLOAD_DIR, f"{stamp}_A_{name_a}")
    path_b = os.path.join(UPLOAD_DIR, f"{stamp}_B_{name_b}")

    settings = load_settings()

    try:
        file_a.save(path_a)
        file_b.save(path_b)

        text_a = extract_text(path_a)
        text_b = extract_text(path_b)

        if not text_a or not text_a.strip() or not text_b or not text_b.strip():
            raise ValueError(
                "No readable text found in one or both documents."
            )

        result = compare_documents(text_a, text_b)

        try:
            result["gemini_explanation"] = explain_matches(
                result.get("matches", [])[:5]
            )
        except Exception:
            app.logger.exception("Gemini explanation failed")
            result["gemini_explanation"] = (
                "AI explanation unavailable. Please review the matches manually."
            )

        result["doc_a"] = name_a
        result["doc_b"] = name_b

        with db() as conn:
            cursor = conn.execute("""
                INSERT INTO comparisons
                (doc_a, doc_b, score, risk, result_json, created_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                name_a,
                name_b,
                result.get("overall_similarity", 0),
                result.get("risk", "Manual review recommended"),
                json.dumps(result),
                datetime.now().isoformat(timespec="seconds"),
                session["user_id"]
            ))

            result["id"] = cursor.lastrowid

        return render_template("results.html", result=result)

    except ValueError as error:
        flash(str(error), "danger")
        return redirect(url_for("compare"))

    except Exception:
        app.logger.exception("Comparison failed")
        flash("Comparison failed. Check the terminal for details.", "danger")
        return redirect(url_for("compare"))

    finally:
        if settings["delete_after_analysis"]:
            for path in (path_a, path_b):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    app.logger.exception("Could not remove uploaded file")


# ------------------------------------------------
# HISTORY
# ------------------------------------------------

@app.route("/history")
@login_required
def history():

    with db() as conn:
        rows = conn.execute("""
            SELECT id, doc_a, doc_b, score, risk, created_at
            FROM comparisons
            WHERE user_id = ?
            ORDER BY id DESC
        """, (session["user_id"],)).fetchall()

    return render_template("history.html", rows=rows)


@app.route("/history/<int:item_id>")
@login_required
def history_item(item_id):

    with db() as conn:
        row = conn.execute("""
            SELECT result_json FROM comparisons
            WHERE id = ? AND user_id = ?
        """, (item_id, session["user_id"])).fetchone()

    if row is None:
        flash("Comparison not found.", "warning")
        return redirect(url_for("history"))

    try:
        result = json.loads(row["result_json"])
    except (TypeError, json.JSONDecodeError):
        flash("Could not read the saved comparison.", "danger")
        return redirect(url_for("history"))

    result["id"] = item_id

    return render_template("results.html", result=result)


# ------------------------------------------------
# DELETE HISTORY
# ------------------------------------------------

@app.route("/history/delete/<int:item_id>", methods=["POST"])
@login_required
def delete_history(item_id):

    if delete_record(item_id, session["user_id"]):
        flash("Comparison deleted.", "success")
    else:
        flash("Comparison not found.", "warning")

    return redirect(url_for("history"))


@app.route("/history/delete-selected", methods=["POST"])
@login_required
def delete_selected_history():

    ids = request.form.getlist("comparison_ids")

    try:
        ids = list(dict.fromkeys(int(value) for value in ids))
        if any(value <= 0 for value in ids):
            raise ValueError
    except (ValueError, TypeError):
        flash("Invalid selection.", "danger")
        return redirect(url_for("history"))

    count = 0

    for item_id in ids:
        if delete_record(item_id, session["user_id"]):
            count += 1

    flash(f"{count} comparison(s) deleted.", "success")
    return redirect(url_for("history"))


@app.route("/history/delete-all", methods=["POST"])
@login_required
def delete_all_history():

    with db() as conn:
        rows = conn.execute("""
            SELECT id FROM comparisons WHERE user_id = ?
        """, (session["user_id"],)).fetchall()

        conn.execute("""
            DELETE FROM comparisons WHERE user_id = ?
        """, (session["user_id"],))

    for row in rows:
        remove_report(row["id"])

    flash(f"{len(rows)} comparison(s) deleted.", "success")
    return redirect(url_for("history"))


# ------------------------------------------------
# PDF REPORT
# ------------------------------------------------

@app.route("/report/<int:item_id>")
@login_required
def report(item_id):

    with db() as conn:
        row = conn.execute("""
            SELECT result_json FROM comparisons
            WHERE id = ? AND user_id = ?
        """, (item_id, session["user_id"])).fetchone()

    if row is None:
        flash("Report not found.", "warning")
        return redirect(url_for("history"))

    try:
        result = json.loads(row["result_json"])
        result["id"] = item_id

        path = report_path(item_id)
        create_pdf(result, path)

    except Exception:
        app.logger.exception("PDF generation failed")
        flash("Could not generate PDF report.", "danger")
        return redirect(url_for("history"))

    if not os.path.isfile(path):
        flash("PDF report was not created.", "danger")
        return redirect(url_for("history"))

    return send_file(path, as_attachment=True)


# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():

    if request.method == "POST":

        if request.form.get("action") != "plagiarism_settings":
            flash("Invalid settings action.", "warning")
            return redirect(url_for("settings"))

        try:
            threshold = int(request.form.get("similarity_threshold", 75))
        except (ValueError, TypeError):
            threshold = 75

        threshold = max(10, min(100, threshold))

        level = request.form.get("comparison_level", "sentence")

        if level not in ("sentence", "paragraph", "document"):
            level = "sentence"

        data = {
            "similarity_threshold": threshold,
            "detect_exact": "detect_exact" in request.form,
            "detect_partial": "detect_partial" in request.form,
            "detect_paraphrase": "detect_paraphrase" in request.form,
            "detect_semantic": "detect_semantic" in request.form,
            "comparison_level": level,
            "highlight_matches": "highlight_matches" in request.form,
            "report_matches": "report_matches" in request.form,
            "report_scores": "report_scores" in request.form,
            "report_recommendations": "report_recommendations" in request.form,
            "delete_after_analysis": "delete_after_analysis" in request.form
        }

        try:
            save_settings(data)
            flash("Settings saved successfully.", "success")
        except OSError:
            app.logger.exception("Could not save settings")
            flash("Could not save settings.", "danger")

        return redirect(url_for("settings"))

    return render_template(
        "settings.html",
        settings=load_settings()
    )


# ------------------------------------------------
# ERROR HANDLER
# ------------------------------------------------

@app.errorhandler(413)
def too_large(error):
    flash("File too large. Maximum size is 512 MB.", "danger")
    return redirect(url_for("compare"))


# ------------------------------------------------
# RUN
# ------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
