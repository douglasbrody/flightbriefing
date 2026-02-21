"""
Flask web server for the Flight Briefer chatbot.
"""

import json
import os
import uuid
from flask import Flask, request, jsonify, render_template, session, Response
import briefer

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))

# ── HTTP Basic Auth ──────────────────────────────────────────────────────────
# Set BRIEFER_USER and BRIEFER_PASS env vars to enable password protection.
# If neither is set (e.g. local dev), auth is skipped entirely.
_AUTH_USER = os.environ.get("BRIEFER_USER", "")
_AUTH_PASS = os.environ.get("BRIEFER_PASS", "")

@app.before_request
def require_auth():
    if not _AUTH_USER:
        return  # no credentials configured — allow all (local dev)
    auth = request.authorization
    if auth and auth.username == _AUTH_USER and auth.password == _AUTH_PASS:
        return  # valid credentials
    return Response(
        "Flight Watch — Authorization Required",
        401,
        {"WWW-Authenticate": 'Basic realm="Flight Watch"'},
    )

# Load aircraft list once at startup
_AIRCRAFT_LIST_PATH = os.path.join(os.path.dirname(__file__), "aircraft.json")
with open(_AIRCRAFT_LIST_PATH) as f:
    AIRCRAFT_LIST = json.load(f)


def _get_session_id() -> str:
    """Get or create a unique session ID stored in the Flask session cookie."""
    if "briefing_id" not in session:
        session["briefing_id"] = str(uuid.uuid4())
    return session["briefing_id"]


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

    try:
        response_text = briefer.chat(
            session_id=session_id,
            user_message=message,
            aircraft=aircraft,
            departure=departure,
        )
        return jsonify({"response": response_text})
    except Exception as e:
        app.logger.error("Error in briefer.chat: %s", str(e))
        return jsonify({
            "response": (
                "Flight Watch is temporarily unavailable due to a system error. "
                "Please contact your local FSS at 1-800-WX-BRIEF for a standard weather briefing."
            )
        })


@app.route("/reset", methods=["POST"])
def reset():
    session_id = _get_session_id()
    briefer.reset_session(session_id)
    return jsonify({"status": "ok"})


@app.route("/aircraft")
def aircraft():
    return jsonify(AIRCRAFT_LIST)


if __name__ == "__main__":
    app.run(debug=True)
