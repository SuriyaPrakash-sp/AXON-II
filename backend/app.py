"""
app.py — Flask application entry point

Run:
    cd backend/
    python app.py

Or with gunicorn:
    gunicorn -w 1 -b 0.0.0.0:5000 app:app
"""

from flask import Flask, send_from_directory
from flask_cors import CORS
from pathlib import Path

from routes import api
from model_loader import get_model_bundle


def create_app():
    app = Flask(__name__, static_folder=None)

    # Allow all origins (tighten to specific origin in production)
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register API blueprint — routes: /data, /sensor/N3, /predict, /health
    app.register_blueprint(api)

    # Serve the frontend/dashboard folder from Flask
    # Looks for   project/dashboard/   relative to this file's parent
    FRONTEND_DIR = Path(__file__).parent.parent / "dashboard"

    if FRONTEND_DIR.exists():
        @app.route("/")
        def serve_index():
            return send_from_directory(str(FRONTEND_DIR), "index.html")

        @app.route("/<path:filename>")
        def serve_static(filename):
            return send_from_directory(str(FRONTEND_DIR), filename)
    else:
        @app.route("/")
        def serve_index():
            return "Flask backend running. Frontend folder not found.", 200

    return app


app = create_app()

if __name__ == "__main__":
    # Pre-load model before first request to avoid cold-start delay
    print("Pre-loading model …")
    get_model_bundle()
    print("Starting Flask on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)