import argparse
import os
import re
import time
import torch

import matplotlib.pyplot as plt
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier
from pathlib import Path
import numpy as np
import pandas as pd
torch.backends.cudnn.benchmark = True  # auto-tunes best convolution algorithms
torch.set_float32_matmul_precision('high')  # improves Tensor Core performance

# NUMPY compatibility shim (fix: "module 'numpy' has no attribute 'int'")

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object

# ✅ Function: Classify result from probability
def classify_label(prob):
    """Convert numeric probability into label and per-video accuracy."""
    if prob >= 0.5:
        return "FAKE", round(prob * 100, 2)
    else:
        return "REAL", round((1 - prob) * 100, 2)

# ✅ Main Script
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict test videos with visualization (fast mode)")
    arg = parser.add_argument
    arg('--weights-dir', type=str, default="weights", help="path to directory with checkpoints")
    arg('--models', nargs='+', required=True, help="checkpoint files")
    arg('--test-dir', type=str, required=True, help="path to directory with videos")
    arg('--output', type=str, default="predictions.csv", help="path to output csv")
    arg('--sequence-length', type=int, default=32, help="Number of frames to extract per video")
    args = parser.parse_args()

    # ✅ Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ✅ Load models
    models = []
    model_paths = [os.path.join(args.weights_dir, model) for model in args.models]
    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
        print(f"Loading state dict: {path}")
        try:
            checkpoint = torch.load(path, map_location=device)
        except TypeError:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
        model.eval()
        if device == "cuda":
            model = model.half()
        models.append(model)

    # ✅ Setup for video reading and prediction
    frames_per_video = args.sequence_length
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    input_size = 380
    strategy = confident_strategy
    stime = time.time()

    # ✅ Load all test videos
    test_videos = sorted([x for x in os.listdir(args.test_dir) if x.endswith(".mp4")])
    print(f"\n🎥 Predicting {len(test_videos)} videos using {frames_per_video} frames each...")

    # ✅ Run predictions
    predictions = predict_on_video_set(
        face_extractor=face_extractor,
        input_size=input_size,
        models=models,
        strategy=strategy,
        frames_per_video=frames_per_video,
        videos=test_videos,
        num_workers=2,
        test_dir=args.test_dir
    )

    # ✅ Convert predictions into readable results
    labels, accuracies = [], []
    for pred in predictions:
        label, acc = classify_label(pred)
        labels.append(label)
        accuracies.append(acc)

    # ✅ Save results as CSV (now showing accuracy instead of confidence)
    submission_df = pd.DataFrame({
        "filename": test_videos,
        "probability": predictions,
        "prediction": labels,
        "accuracy(%)": accuracies
    })
    submission_df.to_csv(args.output, index=False)
    print(f"\n✅ Predictions saved to {args.output}")

    # ✅ Print results summary
    print("\n🎯 Prediction Summary:")
    for f, lbl, acc in zip(test_videos, labels, accuracies):
        print(f"{f:25} → {lbl:<5} (Accuracy: {acc}%)")

    # ✅ Bar chart visualization
    plt.figure(figsize=(8, 5))
    plt.bar(test_videos, predictions, color=['red' if p >= 0.5 else 'green' for p in predictions])
    plt.title("DeepFake Detection Results")
    plt.ylabel("Fake Probability")
    plt.xlabel("Video Filename")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("prediction_graph.png")
    print("\n📊 Saved graph as prediction_graph.png")

    print(f"\n⏱️ Elapsed time: {round(time.time() - stime, 2)} seconds")
