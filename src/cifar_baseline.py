"""Load, audit, and establish a majority-class baseline for CIFAR-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
TRAIN_BATCHES = tuple(f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6))
TEST_BATCH = "cifar-10-batches-py/test_batch"


def _read_batch(archive: tarfile.TarFile, member_name: str) -> tuple[np.ndarray, np.ndarray]:
    member = archive.extractfile(member_name)
    if member is None:
        raise ValueError(f"Dataset archive is missing {member_name!r}")

    batch = pickle.loads(member.read(), encoding="bytes")
    images = np.asarray(batch[b"data"], dtype=np.uint8)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def load_dataset(archive_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return official training and test images and labels from a CIFAR-10 tarball."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"CIFAR-10 archive not found: {archive_path}")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        train_batches = [_read_batch(archive, name) for name in TRAIN_BATCHES]
        test_images, test_labels = _read_batch(archive, TEST_BATCH)

    train_images = np.concatenate([batch[0] for batch in train_batches])
    train_labels = np.concatenate([batch[1] for batch in train_batches])
    return train_images, train_labels, test_images, test_labels


def audit_dataset(images: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Summarize structural quality, label balance, duplicate images, and pixel range."""
    fingerprints = Counter(hashlib.sha256(image.tobytes()).digest() for image in images)
    duplicate_groups = sum(count > 1 for count in fingerprints.values())
    duplicate_excess = sum(count - 1 for count in fingerprints.values() if count > 1)
    class_counts = Counter(labels.tolist())

    return {
        "image_count": int(images.shape[0]),
        "stored_image_shape": list(images.shape[1:]),
        "image_dimensions": [3, 32, 32],
        "image_dtype": str(images.dtype),
        "target_dtype": str(labels.dtype),
        "target_type": "categorical (10 class labels)",
        "missing_image_values": int(np.count_nonzero(~np.isfinite(images))),
        "missing_targets": 0,
        "invalid_targets": int(
            np.count_nonzero((labels < 0) | (labels >= len(CLASS_NAMES)))
        ),
        "duplicate_image_groups": int(duplicate_groups),
        "duplicate_excess_images": int(duplicate_excess),
        "pixel_min": int(images.min()),
        "pixel_max": int(images.max()),
        "pixels_outside_uint8_range": int(np.count_nonzero((images < 0) | (images > 255))),
        "target_distribution": {
            CLASS_NAMES[label]: int(class_counts.get(label, 0))
            for label in range(len(CLASS_NAMES))
        },
    }


def evaluate_majority_baseline(
    train_labels: np.ndarray, validation_labels: np.ndarray
) -> dict[str, Any]:
    """Fit a majority-class classifier without using image features."""
    train_features = np.zeros((len(train_labels), 1), dtype=np.uint8)
    validation_features = np.zeros((len(validation_labels), 1), dtype=np.uint8)

    model = DummyClassifier(strategy="most_frequent")
    model.fit(train_features, train_labels)
    predictions = model.predict(validation_features)

    precision, recall, f1, _ = precision_recall_fscore_support(
        validation_labels,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(
        validation_labels, predictions, labels=np.arange(len(CLASS_NAMES))
    )
    errors_by_predicted_class = {
        CLASS_NAMES[int(label)]: int(np.count_nonzero(
            (validation_labels != predictions) & (predictions == label)
        ))
        for label in np.unique(predictions)
    }

    return {
        "model": "DummyClassifier(strategy='most_frequent')",
        "predicted_class": CLASS_NAMES[int(model.classes_[0])],
        "accuracy": float(accuracy_score(validation_labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": matrix.tolist(),
        "class_order": list(CLASS_NAMES),
        "misclassified_images_by_predicted_class": errors_by_predicted_class,
    }


def build_results(archive_path: Path) -> dict[str, Any]:
    train_images, train_labels, test_images, test_labels = load_dataset(archive_path)
    all_images = np.concatenate((train_images, test_images))
    all_labels = np.concatenate((train_labels, test_labels))

    train_indices, validation_indices = train_test_split(
        np.arange(len(train_labels)),
        test_size=0.2,
        random_state=42,
        stratify=train_labels,
    )
    baseline = evaluate_majority_baseline(
        train_labels[train_indices], train_labels[validation_indices]
    )

    return {
        "dataset_audit": audit_dataset(all_images, all_labels),
        "official_splits": {
            "training_images": int(len(train_labels)),
            "official_test_images_reserved": int(len(test_labels)),
        },
        "validation_design": {
            "source": "official training split only",
            "train_images": int(len(train_indices)),
            "validation_images": int(len(validation_indices)),
            "strategy": "80/20 stratified holdout",
            "random_state": 42,
        },
        "baseline": baseline,
    }


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=repository_root / "cifar-10-python.tar.gz",
        help="Path to cifar-10-python.tar.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "results" / "baseline_results.json",
        help="Path for the machine-readable audit and baseline results",
    )
    args = parser.parse_args()

    results = build_results(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
