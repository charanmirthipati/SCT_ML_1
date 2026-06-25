import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as img:
        gray = img.convert("L")
        resized = gray.resize((image_size, image_size))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        return arr.ravel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SVM classifier for cats vs dogs images."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("PetImages"),
        help="Directory that contains Cat/ and Dog/ folders.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Image width/height in pixels after resize.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for test split.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap on images per class for faster training.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("svm_cat_dog.joblib"),
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    return parser.parse_args()


def gather_image_paths(class_dir: Path, max_images: int | None) -> list[Path]:
    paths = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
    paths.sort()
    if max_images is not None:
        return paths[:max_images]
    return paths


def load_dataset(dataset_dir: Path, image_size: int, max_per_class: int | None):
    class_map = {"Cat": 0, "Dog": 1}
    X: list[np.ndarray] = []
    y: list[int] = []
    skipped = 0

    for class_name, label in class_map.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for image_path in gather_image_paths(class_dir, max_per_class):
            try:
                X.append(preprocess_image(image_path, image_size))
                y.append(label)
            except (UnidentifiedImageError, OSError):
                # Some files in PetImages are corrupted; skip them safely.
                skipped += 1

    if not X:
        raise RuntimeError("No valid images found. Check dataset path and image files.")

    return np.asarray(X), np.asarray(y), skipped


def build_model(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    random_state=random_state,
                    dual=False,
                    max_iter=3000,
                ),
            ),
        ]
    )


def predict_image(model: Pipeline, image_path: Path, image_size: int) -> tuple[str, float | None]:
    features = preprocess_image(image_path, image_size)
    prediction = int(model.predict([features])[0])

    label = "Dog" if prediction == 1 else "Cat"
    confidence = None

    svm = model.named_steps["svm"]
    if hasattr(svm, "decision_function"):
        scaled_features = model.named_steps["scaler"].transform([features])
        score = float(svm.decision_function(scaled_features)[0])
        confidence = 1.0 / (1.0 + np.exp(-abs(score)))

    return label, confidence


def main() -> None:
    args = parse_args()

    X, y, skipped = load_dataset(args.dataset_dir, args.image_size, args.max_per_class)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(args.random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Loaded samples: {len(X)}")
    print(f"Skipped corrupted/unreadable files: {skipped}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Cat", "Dog"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, args.model_out)
    print(f"\nSaved model to: {args.model_out}")


if __name__ == "__main__":
    main()
