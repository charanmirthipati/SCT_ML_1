import base64
import tempfile
from pathlib import Path

import joblib
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from svm_cats_dogs import predict_image


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "svm_cat_dog.joblib"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

app = Flask(__name__)
app.secret_key = "cat-dog-svm-demo"


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    model = load_model()
    result = None
    confidence = None
    image_data = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            flash("Choose an image first.", "error")
            return redirect(url_for("index"))

        if not allowed_file(uploaded_file.filename):
            flash("Use a JPG, PNG, BMP, GIF, or WEBP image.", "error")
            return redirect(url_for("index"))

        if model is None:
            flash("Train the model first by running svm_cats_dogs.py.", "error")
            return redirect(url_for("index"))

        safe_name = secure_filename(uploaded_file.filename)
        suffix = Path(safe_name).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            uploaded_file.save(temp_file.name)
            temp_path = Path(temp_file.name)

        try:
            result, confidence = predict_image(model, temp_path, image_size=64)
            image_bytes = temp_path.read_bytes()
            image_data = base64.b64encode(image_bytes).decode("ascii")
        finally:
            temp_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        model_ready=model is not None,
        result=result,
        confidence=confidence,
        image_data=image_data,
    )


@app.route("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


if __name__ == "__main__":
    app.run(debug=True)
