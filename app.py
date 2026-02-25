"""
Flask web server for the Flight Briefer chatbot.
"""

import json
import os
import uuid
from flask import (Flask, request, jsonify, render_template,
                   session, Response, stream_with_context)
import briefer

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))

# ── HTTP Basic Auth ──────────────────────────────────────────────────────────
_AUTH_USER = os.environ.get("BRIEFER_USER", "")
_AUTH_PASS = os.environ.get("BRIEFER_PASS", "")

@app.before_request
def require_auth():
    if not _AUTH_USER:
        return  # no credentials set — allow all (local dev)
    auth = request.authorization
    if auth and auth.username == _AUTH_USER and auth.password == _AUTH_PASS:
        return
    return Response(
        "Flight Watch — Authorization Required",
        401,
        {"WWW-Authenticate": 'Basic realm="Flight Watch"'},
    )

# ── Aircraft list ────────────────────────────────────────────────────────────
_AIRCRAFT_LIST_PATH = os.path.join(os.path.dirname(__file__), "aircraft.json")
with open(_AIRCRAFT_LIST_PATH) as f:
    AIRCRAFT_LIST = json.load(f)


def _get_session_id() -> str:
    if "briefing_id" not in session:
        session["briefing_id"] = str(uuid.uuid4())
    return session["briefing_id"]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    aircraft = (data.get("aircraft") or "").strip()
    departure = (data.get("departure") or "").strip()

    if not message:
        return jsonify({"error": "No message provided."}), 400

    session_id = _get_session_id()

    def generate():
        try:
            for event in briefer.stream_chat(session_id, message, aircraft, departure):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            app.logger.error("Streaming error: %s", str(e))
            error = {"type": "error",
                     "text": "A system error occurred. Please try again."}
            yield f"data: {json.dumps(error)}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/reset", methods=["POST"])
def reset():
    session_id = _get_session_id()
    briefer.reset_session(session_id)
    return jsonify({"status": "ok"})


@app.route("/aircraft")
def aircraft():
    return jsonify(AIRCRAFT_LIST)


@app.route("/env-check")
def env_check():
    """Diagnostic: shows which key env vars are present (no values exposed)."""
    keys = ["FAA_CLIENT_ID", "FAA_CLIENT_SECRET", "NMS_CLIENT_ID", "NMS_CLIENT_SECRET",
            "ANTHROPIC_API_KEY", "FLASK_SECRET_KEY", "BRIEFER_USER"]
    return jsonify({k: bool(os.environ.get(k)) for k in keys})


if __name__ == "__main__":
    app.run(debug=True)
