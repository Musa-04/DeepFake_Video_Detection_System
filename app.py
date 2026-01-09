

import os
import re
import uuid
import base64
import shutil
import traceback
import cv2
import numpy as np
import torch
import logging
import sys
import threading
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# --------------------------
# Logging configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("dfdetect")

# --------------------------
# Robust albumentations shim
# --------------------------
_image_compression_impl = None

def _make_wrapper_from_transform(ImageCompression):
    def wrapper(image, quality=50, image_type=None, **kwargs):
        try:
            q = int(max(1, min(100, int(quality))))
        except Exception:
            q = 50
        transform = ImageCompression(quality_lower=q, quality_upper=q, p=1.0)
        result = transform(image=image)
        return result.get("image", image)
    return wrapper

_import_attempts = [
    ("albumentations.augmentations.functional", "image_compression"),
    ("albumentations.functional", "image_compression"),
    ("albumentations.augmentations.transforms", "ImageCompression"),
    ("albumentations", "augmentations.transforms.ImageCompression"),
]

for module_path, attr in _import_attempts:
    try:
        mod = __import__(module_path, fromlist=['*'])
        if hasattr(mod, attr):
            _image_compression_impl = getattr(mod, attr)
            break
        if "." in attr:
            subattr_parts = attr.split(".")
            candidate = mod
            ok = True
            for p in subattr_parts:
                if hasattr(candidate, p):
                    candidate = getattr(candidate, p)
                else:
                    ok = False
                    break
            if ok:
                _image_compression_impl = _make_wrapper_from_transform(candidate)
                break
    except Exception:
        continue

if _image_compression_impl is None:
    try:
        from albumentations.augmentations.transforms import ImageCompression  # type: ignore
        _image_compression_impl = _make_wrapper_from_transform(ImageCompression)
    except Exception:
        _image_compression_impl = None

if _image_compression_impl is None:
    def _image_compression_noop(image, quality=50, image_type=None, **kwargs):
        return image
    _image_compression_impl = _image_compression_noop

image_compression = _image_compression_impl

# -----------------------
# facenet / MTCNN
# -----------------------
from facenet_pytorch.models.mtcnn import MTCNN

# concurrency + transforms
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import Normalize

# model import from your repo (leave as-is)
from training.zoo.classifiers import DeepFakeClassifier

# -----------------------
# App init
# -----------------------
app = Flask(__name__)
CORS(app)

# -----------------------
# NUMPY compat shim (older code expecting np.int etc)
# -----------------------
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object

# -----------------------
# CONFIG
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = "weights"
WEIGHTS_NAME = "model_v1"
INPUT_SIZE = 380

# Sampling defaults
MIN_SAMPLE_FRAMES = 16
MAX_SAMPLE_FRAMES = 64
SAMPLES_PER_SECOND = 1.0

TOP_K_FRAMES = 8
INFERENCE_BATCH = 128

# Storage behavior
SAVE_FRAMES_TEMP = True
KEEP_FRAMES_AFTER_PRED = True
FRAMES_SUBDIR_NAME = "frames"           # annotated frames (existing)
VIDEO_SUBDIR_NAME = "video"             # original video
ALL_FRAMES_SUBDIR_NAME = "all_frames"   # every frame from the video
SAMPLE_FRAMES_SUBDIR_NAME = "sample_frames"  # sampled frames used by pipeline

# New toggles
SAVE_ALL_FRAMES = True
SAVE_SAMPLE_FRAMES = True

# -----------------------
# Uploads cleanup config
# -----------------------
UPLOADS_BASE_DIR = "uploads"
MAX_UPLOAD_AGE_SECONDS = 60 * 60        # 1 hour
CLEANUP_INTERVAL_SECONDS = 10 * 60      # 10 minutes

# -----------------------
# Model loader
# -----------------------
def load_models(weights_dir, weight_names):
    device = DEVICE
    models = []
    model_paths = (
        [os.path.join(weights_dir, name) for name in weight_names]
        if isinstance(weight_names, (list, tuple))
        else [os.path.join(weights_dir, weight_names)]
    )
    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight file not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=device)
        except TypeError:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {re.sub(r"^module\.", "", k): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)
        model.eval()
        if device == "cuda":
            model = model.half()
        models.append(model)
    return models

logger.info("Loading model(s)...")
models = load_models(WEIGHTS_DIR, WEIGHTS_NAME)
logger.info(f"Loaded {len(models)} model(s). Device: {DEVICE}")

# -----------------------
# Normalization util (ImageNet stats)
# -----------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

# -----------------------
# VideoReader & helpers
# -----------------------
class VideoReader:
    def __init__(self, verbose=True, insets=(0, 0)):
        self.verbose = verbose
        self.insets = insets

    def read_frames(self, path, num_frames, jitter=0, seed=None):
        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            capture.release()
            return None

        if num_frames is None:
            # read all frames
            frame_idxs = list(range(frame_count))
        else:
            assert num_frames > 0
            frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)

        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def _read_frames_at_indices(self, path, capture, frame_idxs):
        try:
            frames = []
            idxs_read = []
            for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
                ret = capture.grab()
                if not ret:
                    break
                current = len(idxs_read)
                if frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        break
                    frame = self._postprocess_frame(frame)
                    frames.append(frame)
                    idxs_read.append(frame_idx)
                    # logging progress for each retrieved frame
                    logger.info(f"[VideoReader] Read frame {frame_idx} ({len(idxs_read)}/{len(frame_idxs)}) from {os.path.basename(path)}")
                    if len(idxs_read) == len(frame_idxs):
                        break
            if len(frames) > 0:
                return np.stack(frames), idxs_read
            if self.verbose:
                logger.info("No frames read from movie %s" % path)
            return None
        except Exception as e:
            if self.verbose:
                logger.exception("Exception while reading movie %s: %s" % (path, str(e)))
            return None

    def _postprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.insets[0] > 0:
            W = frame.shape[1]
            p = int(W * self.insets[0])
            frame = frame[:, p:-p, :]
        if self.insets[1] > 0:
            H = frame.shape[0]
            q = int(H * self.insets[1])
            frame = frame[q:-q, :, :]
        return frame

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized

def put_to_center(img, input_size):
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0], start_w: start_w + img.shape[1], :] = img
    return image

# -----------------------
# FaceExtractor (MTCNN)
# -----------------------
class FaceExtractor:
    def __init__(self, video_read_fn):
        self.video_read_fn = video_read_fn
        mtcnn_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device=mtcnn_device)

    def process_videos(self, input_dir, filenames, video_idxs):
        results = []
        for video_idx in video_idxs:
            filename = filenames[video_idx]
            video_path = os.path.join(input_dir, filename)
            result = self.video_read_fn(video_path)
            if result is None:
                continue
            my_frames, my_idxs = result
            for i, frame in enumerate(my_frames):
                h_frame, w_frame = frame.shape[:2]
                logger.info(f"[FaceExtractor] Detecting faces in video={filename} frame={my_idxs[i]} size={w_frame}x{h_frame}")
                img = Image.fromarray(frame.astype(np.uint8))
                img = img.resize(size=[s // 2 for s in img.size])
                batch_boxes, probs = self.detector.detect(img, landmarks=False)
                if batch_boxes is None:
                    logger.info(f"[FaceExtractor] No faces in frame {my_idxs[i]} of {filename}")
                    continue
                faces = []
                scores = []
                boxes = []
                num_faces = 0
                for bbox, score in zip(batch_boxes, probs):
                    if bbox is not None:
                        num_faces += 1
                        xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                        xmin = max(0, xmin); ymin = max(0, ymin)
                        xmax = min(w_frame - 1, xmax); ymax = min(h_frame - 1, ymax)
                        w = xmax - xmin
                        h = ymax - ymin
                        p_h = h // 3
                        p_w = w // 3
                        crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                        faces.append(crop)
                        scores.append(float(score))
                        boxes.append((xmin, ymin, xmax - xmin, ymax - ymin))
                        logger.debug(f"[FaceExtractor] face crop box={(xmin,ymin,xmax,ymax)} score={score:.3f}")
                logger.info(f"[FaceExtractor] Found {num_faces} faces in frame {my_idxs[i]} of {filename}")
                if len(faces) == 0:
                    continue
                frame_dict = {
                    "video_idx": video_idx,
                    "frame_idx": my_idxs[i],
                    "frame_w": w_frame,
                    "frame_h": h_frame,
                    "faces": faces,
                    "scores": scores,
                    "boxes": boxes
                }
                results.append(frame_dict)
        return results

    def process_video(self, video_path):
        input_dir = os.path.dirname(video_path)
        filenames = [os.path.basename(video_path)]
        return self.process_videos(input_dir, filenames, [0])

# -----------------------
# strategy / classify utils
# -----------------------
def confident_strategy(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    if sz == 0:
        return 0.5
    fakes = np.count_nonzero(pred > t)
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)

def classify_label(prob):
    if prob >= 0.5:
        return "FAKE", round(prob * 100, 2)
    else:
        return "REAL", round((1 - prob) * 100, 2)

# -----------------------
# Drawing helper to annotate frames
# -----------------------
def draw_box_and_label(frame_bgr, box, label_text, score, label_col=(0,200,150)):
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    thickness = max(1, int(round(min(frame_bgr.shape[0], frame_bgr.shape[1]) / 200)))
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), label_col, thickness=thickness)
    text = f"{label_text} {round(score * 100, 2)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(frame_bgr.shape[1] / 1200.0, 0.9))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    bg_x1 = x
    bg_y1 = max(0, y - text_h - 2 * pad)
    bg_x2 = x + text_w + 2 * pad
    bg_y2 = max(0, y)
    cv2.rectangle(frame_bgr, (bg_x1, bg_y1), (bg_x2, bg_y2), (10, 10, 10), cv2.FILLED)
    cv2.putText(frame_bgr, text, (x + pad, bg_y2 - pad), font, font_scale, label_col, thickness=thickness, lineType=cv2.LINE_AA)

# -----------------------
# FPS-aware sampling function
# -----------------------
def compute_frames_to_sample(video_path, min_frames=MIN_SAMPLE_FRAMES, max_frames=MAX_SAMPLE_FRAMES, frames_per_second=SAMPLES_PER_SECOND):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return min_frames
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if frame_count <= 0:
            return min_frames

        duration = frame_count / fps
        num = int(np.ceil(duration * frames_per_second))

        if num < min_frames:
            num = min_frames
        if num > max_frames:
            num = max_frames
        num = min(num, frame_count)
        return int(max(1, num))
    except Exception:
        return min_frames

# instantiate video helpers
video_reader = VideoReader()
video_read_fn = lambda path: video_reader.read_frames(
    path,
    num_frames=compute_frames_to_sample(path, min_frames=MIN_SAMPLE_FRAMES, max_frames=MAX_SAMPLE_FRAMES, frames_per_second=SAMPLES_PER_SECOND)
)
face_extractor = FaceExtractor(video_read_fn)

# -----------------------
# Forgery reason generator helpers
# -----------------------
def high_freq_energy(gray_crop):
    # small FFT-based heuristic for high-frequency energy
    try:
        f = np.fft.fft2(gray_crop)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        h, w = magnitude.shape
        # consider outer band as high frequency
        band = magnitude.copy()
        cy, cx = h // 2, w // 2
        band[cy - h//4:cy + h//4, cx - w//4:cx + w//4] = 0
        return float(np.sum(band) / (h*w) + 1e-6)
    except Exception:
        return 0.0

def boundary_artifact_score(face_gray):
    try:
        h, w = face_gray.shape[:2]
        # small border vs inner variance
        border = np.concatenate([
            face_gray[0:3,:].ravel() if h>3 else np.array([]),
            face_gray[-3:,:].ravel() if h>3 else np.array([]),
            face_gray[:,0:3].ravel() if w>3 else np.array([]),
            face_gray[:,-3:].ravel() if w>3 else np.array([])
        ])
        inner = face_gray[3:-3,3:-3].ravel() if (h>8 and w>8) else face_gray.ravel()
        if inner.size == 0 or border.size == 0:
            return 0.0
        b_var = np.var(border)
        i_var = np.var(inner) + 1e-6
        ratio = abs(b_var - i_var) / (i_var + b_var)
        # squash to 0..1
        return float(np.tanh(ratio*2.0))
    except Exception:
        return 0.0

def color_consistency_score(face_rgb, neck_rgb):
    try:
        # compare mean HSV vectors
        face_hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV).mean(axis=(0,1))
        neck_hsv = cv2.cvtColor(neck_rgb, cv2.COLOR_RGB2HSV).mean(axis=(0,1))
        dist = np.linalg.norm(face_hsv - neck_hsv) / (np.linalg.norm([255,255,255]) + 1e-6)
        return float(min(1.0, dist * 2.0))
    except Exception:
        return 0.0

def analyze_video_temporal_for_flow(frames_rgb, frame_face_boxes_map, idxs_read):
    """
    frames_rgb: list/array of sampled frames (RGB)
    frame_face_boxes_map: dict mapping frame_idx -> list of boxes (x,y,w,h)
    idxs_read: list of frame indices corresponding to frames_rgb
    """
    flow_mags = []
    prev_gray = None
    # convert to BGR grayscale array for optical flow
    for pos, fidx in enumerate(idxs_read):
        frame = frames_rgb[pos]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            # calc dense flow across full frame, but we'll sample in face boxes
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # if face box present, sample mean mag inside union of boxes, else use global mean
            boxes = frame_face_boxes_map.get(fidx, [])
            if boxes:
                face_vals = []
                for box in boxes:
                    x,y,w,h = [int(v) for v in box]
                    x2, y2 = min(x+w, mag.shape[1]-1), min(y+h, mag.shape[0]-1)
                    if x < 0 or y < 0 or x>=mag.shape[1] or y>=mag.shape[0]:
                        continue
                    crop = mag[y:y2, x:x2]
                    if crop.size:
                        face_vals.append(np.mean(crop))
                if face_vals:
                    flow_mags.append(float(np.mean(face_vals)))
                else:
                    flow_mags.append(float(np.mean(mag)))
            else:
                flow_mags.append(float(np.mean(mag)))
        prev_gray = gray
    flow_var = float(np.var(flow_mags)) if len(flow_mags) > 0 else 0.0
    flow_mean = float(np.mean(flow_mags)) if len(flow_mags) > 0 else 0.0
    return {"flow_variance": flow_var, "flow_mean": flow_mean, "flow_samples": flow_mags}

def generate_forgery_reasons_from_frame_analyses(frame_analyses, temporal_metrics, thresholds=None, top_k_frames_map=None):
    """
    frame_analyses: list of dicts, each with keys: frame_idx, boundary_score, hf_energy, color_consistency, model_score
    temporal_metrics: dict from analyze_video_temporal_for_flow
    top_k_frames_map: list of (frame_idx, score) tuples used for evidence
    """
    if thresholds is None:
        thresholds = {
            "boundary": 0.45,
            "hf_low": 5000,     # heuristic; depends on FFT scale
            "hf_high": 40000,
            "color_consistency": 0.25,
            "flow_var": 0.6,
            "model_conf": 0.6
        }

    reasons = []
    # model confidence
    model_scores = [f.get("model_score") for f in frame_analyses if f.get("model_score") is not None]
    mean_model = float(np.mean(model_scores)) if model_scores else None
    if mean_model is not None and mean_model >= thresholds["model_conf"]:
        reasons.append({
            "type":"model_confidence",
            "score": mean_model,
            "reason": f"Classifier indicates manipulation (average model score {mean_model:.3f}).",
            "evidence_frames": [int(f[0]) for f in (top_k_frames_map or [])]
        })

    # boundary
    boundary_vals = [f["boundary_score"] for f in frame_analyses]
    if len(boundary_vals) and float(np.mean(boundary_vals)) > thresholds["boundary"]:
        # pick frames with top boundary score for evidence
        sorted_b = sorted(frame_analyses, key=lambda x: x["boundary_score"], reverse=True)[:3]
        reasons.append({
            "type":"boundary_artifact",
            "score": float(np.mean(boundary_vals)),
            "reason": "Visible blending/edge artifacts detected around face boundaries.",
            "evidence_frames": [int(f["frame_idx"]) for f in sorted_b]
        })

    # texture / hf energy (too low -> oversmoothed; too high -> noise)
    hf_vals = [f["hf_energy"] for f in frame_analyses]
    if len(hf_vals):
        hf_mean = float(np.mean(hf_vals))
        if hf_mean < thresholds["hf_low"]:
            sorted_h_low = sorted(frame_analyses, key=lambda x: x["hf_energy"])[:3]
            reasons.append({
                "type":"texture_smooth",
                "score": hf_mean,
                "reason": "Face texture appears overly smooth (lack of high-frequency detail), which is typical for many generated faces.",
                "evidence_frames": [int(f["frame_idx"]) for f in sorted_h_low]
            })
        elif hf_mean > thresholds["hf_high"]:
            sorted_h_hi = sorted(frame_analyses, key=lambda x: x["hf_energy"], reverse=True)[:3]
            reasons.append({
                "type":"texture_noise",
                "score": hf_mean,
                "reason": "Unnatural high-frequency noise present on the face region (possible generation artifacts).",
                "evidence_frames": [int(f["frame_idx"]) for f in sorted_h_hi]
            })

    # color / lighting inconsistency
    color_vals = [f["color_consistency"] for f in frame_analyses]
    if len(color_vals) and float(np.mean(color_vals)) > thresholds["color_consistency"]:
        sorted_c = sorted(frame_analyses, key=lambda x: x["color_consistency"], reverse=True)[:3]
        reasons.append({
            "type":"lighting_mismatch",
            "score": float(np.mean(color_vals)),
            "reason": "Lighting/color on face differs noticeably from surrounding skin/background (possible compositing).",
            "evidence_frames": [int(f["frame_idx"]) for f in sorted_c]
        })

    # temporal (optical flow variance)
    if temporal_metrics.get("flow_variance", 0) > thresholds["flow_var"]:
        reasons.append({
            "type":"temporal_inconsistency",
            "score": float(temporal_metrics.get("flow_variance", 0)),
            "reason": "Temporal inconsistency detected (irregular motion/jitter across frames).",
            "evidence_frames": [int(f[0]) for f in (top_k_frames_map or [])]
        })

    if len(reasons) == 0:
        reasons.append({
            "type":"no_strong_evidence",
            "score": 0.0,
            "reason": "No strong forgery evidence found by heuristics. Rely on model confidence and inspect frames if unsure.",
            "evidence_frames": [int(f[0]) for f in (top_k_frames_map or [])]
        })

    summary = {
        "reasons": reasons,
        "mean_model_score": mean_model,
        "temporal_metrics": temporal_metrics
    }
    return summary

# Fallback builder used when we can't re-read sampled full frames
def _build_fallback_frame_analyses(processed_faces, face_frame_refs, face_boxes, per_face_probs):
    analyses = []
    for face_idx, img in enumerate(processed_faces):
        try:
            face_rgb = img.copy()
            face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
            hfe = high_freq_energy(cv2.resize(face_gray, (128,128)))
            bscore = boundary_artifact_score(cv2.resize(face_gray, (128,128)))
            H = face_rgb.shape[0]; W = face_rgb.shape[1]
            neck = face_rgb[int(H*0.75):H, int(W*0.25):int(W*0.75)] if H>10 else face_rgb
            cs = color_consistency_score(face_rgb, neck if neck.size else face_rgb)
            model_score = float(per_face_probs[face_idx]) if face_idx < len(per_face_probs) else None
            analyses.append({
                "frame_idx": int(face_frame_refs[face_idx]) if face_idx < len(face_frame_refs) else -1,
                "boundary_score": float(bscore),
                "hf_energy": float(hfe),
                "color_consistency": float(cs),
                "model_score": model_score
            })
        except Exception:
            continue
    return analyses

# -----------------------
# Main prediction function (annotates & saves frames)
# -----------------------
def predict_single_video_and_frames(video_path, save_dir=None, top_k=TOP_K_FRAMES, apply_compression=False):
    try:
        logger.info(f"[Predict] Starting prediction for {os.path.basename(video_path)}")

        # --- Optionally save ALL frames to disk (every frame) ---
        if save_dir and SAVE_ALL_FRAMES:
            try:
                all_res = video_reader.read_frames(video_path, num_frames=None)
                if all_res is not None:
                    all_frames, all_idxs = all_res
                    all_dir = os.path.join(save_dir, ALL_FRAMES_SUBDIR_NAME)
                    os.makedirs(all_dir, exist_ok=True)
                    for pos, fidx in enumerate(all_idxs):
                        frame_rgb = all_frames[pos]
                        out_path = os.path.join(all_dir, f"frame_{fidx:06d}.jpg")
                        cv2.imwrite(out_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    logger.info(f"[Predict] Saved all frames to {all_dir} ({len(all_idxs)} files)")
                else:
                    logger.info("[Predict] Could not read all frames to save.")
            except Exception:
                logger.exception("Error while saving all frames")

        # Run face extractor on sampled frames (existing behavior)
        faces_info = face_extractor.process_video(video_path)
        if len(faces_info) == 0:
            logger.info("[Predict] No faces found in sampled frames.")
            # Ensure consistent return signature with empty reasons summary
            reasons_empty = {"reasons": [{"type":"no_faces","score":0.0, "reason":"No faces found"}], "mean_model_score": None, "temporal_metrics": {}}
            return [], [], "NO_FACES", 0.0, [], reasons_empty

        processed_faces = []
        face_frame_refs = []
        face_boxes = []

        for frame_data in faces_info:
            fidx = frame_data["frame_idx"]
            boxes = frame_data.get("boxes", [])
            for i, face in enumerate(frame_data.get("faces", [])):
                resized = isotropically_resize_image(face, INPUT_SIZE)
                centered = put_to_center(resized, INPUT_SIZE)
                if apply_compression:
                    centered = image_compression(centered, quality=90, image_type=".jpg")
                processed_faces.append(centered)
                face_frame_refs.append(fidx)
                face_boxes.append(boxes[i] if i < len(boxes) else None)

        n_faces = len(processed_faces)
        if n_faces == 0:
            logger.info("[Predict] After processing, no faces to run inference on.")
            reasons_empty = {"reasons": [{"type":"no_faces","score":0.0, "reason":"No faces after processing"}], "mean_model_score": None, "temporal_metrics": {}}
            return [], [], "NO_FACES", 0.0, [], reasons_empty

        # Convert to tensor and normalize
        x_np = np.stack(processed_faces, axis=0)  # (N,H,W,3)
        x_t = torch.tensor(x_np, device=DEVICE).float().permute(0, 3, 1, 2)
        for i in range(x_t.shape[0]):
            x_t[i] = normalize_transform(x_t[i] / 255.0)
        if DEVICE == "cuda":
            x_t = x_t.half()

        # Inference
        num_models = len(models)
        per_model_outputs = [[] for _ in range(num_models)]
        with torch.no_grad():
            for start in range(0, n_faces, INFERENCE_BATCH):
                end = min(start + INFERENCE_BATCH, n_faces)
                batch = x_t[start:end]
                logger.info(f"[Predict] Inference batch faces {start}:{end} ...")
                for mi, m in enumerate(models):
                    y = m(batch)
                    y_sig = torch.sigmoid(y.squeeze())
                    per_model_outputs[mi].append(y_sig.cpu().numpy())

        per_model_face_probs = np.array([np.concatenate(per_model_outputs[mi], axis=0) for mi in range(num_models)])
        per_face_probs = per_model_face_probs.mean(axis=0)  # shape (n_faces,)

        # Map per-face probs to frame-level
        frame_to_probs = {}
        for idx_face, frame_idx in enumerate(face_frame_refs):
            frame_to_probs.setdefault(frame_idx, []).append(float(per_face_probs[idx_face]))
        frame_level = {fidx: max(probs) for fidx, probs in frame_to_probs.items()}

        # video level aggregation
        video_prob = confident_strategy(per_face_probs)
        label, accuracy = classify_label(video_prob)
        logger.info(f"[Predict] Video-level label={label} confidence={accuracy}")

        # top-k frames (based on frame_level)
        sorted_frames = sorted(frame_level.items(), key=lambda x: x[1], reverse=True)
        top_frames = sorted_frames[:top_k]

        # Build fallback analyses from processed faces (used if we can't re-read full frames)
        fallback_frame_analyses = _build_fallback_frame_analyses(processed_faces, face_frame_refs, face_boxes, per_face_probs)

        # Read sampled full frames (same sampling used by FaceExtractor)
        full_frames_res = video_reader.read_frames(video_path, num_frames=compute_frames_to_sample(video_path))
        thumbnail_b64s = []
        top_scores = []
        reasons_summary = {}

        # If we have the sampled frames, create analyses and thumbnails
        if full_frames_res is None:
            top_scores = [round(float(s), 4) for _, s in top_frames]
            logger.info("[Predict] Could not re-read full sampled frames to create thumbnails.")
            temporal_metrics = {"flow_variance": 0.0, "flow_mean": 0.0, "flow_samples": []}
            reasons_summary = generate_forgery_reasons_from_frame_analyses(fallback_frame_analyses, temporal_metrics, top_k_frames_map=top_frames)
        else:
            full_frames, idxs_read = full_frames_res
            # Prepare directories
            if save_dir and SAVE_FRAMES_TEMP:
                annotated_dir = os.path.join(save_dir, FRAMES_SUBDIR_NAME)
                os.makedirs(annotated_dir, exist_ok=True)
            if save_dir and SAVE_SAMPLE_FRAMES:
                sample_dir = os.path.join(save_dir, SAMPLE_FRAMES_SUBDIR_NAME)
                os.makedirs(sample_dir, exist_ok=True)

            # Map frame_idx -> list of face indices
            frame_face_map = {}
            for face_idx, frame_idx in enumerate(face_frame_refs):
                frame_face_map.setdefault(frame_idx, []).append(face_idx)

            # Annotate / save sampled frames
            for pos, fidx in enumerate(idxs_read):
                frame_rgb = full_frames[pos]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()

                if fidx in frame_face_map:
                    for face_idx in frame_face_map[fidx]:
                        prob = float(per_face_probs[face_idx])
                        lab, _ = classify_label(prob)
                        col = (30, 30, 200) if lab == "FAKE" else (50, 200, 50)
                        box = face_boxes[face_idx]
                        if box is not None:
                            draw_box_and_label(frame_bgr, box, lab, prob, label_col=col)

                # Save annotated frame (existing behavior)
                if save_dir and SAVE_FRAMES_TEMP:
                    out_path = os.path.join(save_dir, FRAMES_SUBDIR_NAME, f"frame_{fidx:06d}.jpg")
                    cv2.imwrite(out_path, frame_bgr)
                    logger.info(f"[Predict] Saved annotated frame {fidx} -> {out_path}")

                # Save sampled frame (raw RGB -> BGR jpg) if requested
                if save_dir and SAVE_SAMPLE_FRAMES:
                    sample_out = os.path.join(save_dir, SAMPLE_FRAMES_SUBDIR_NAME, f"frame_{fidx:06d}.jpg")
                    cv2.imwrite(sample_out, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if save_dir and SAVE_SAMPLE_FRAMES:
                try:
                    logger.info(f"[Predict] Saved sampled frames to {os.path.join(save_dir, SAMPLE_FRAMES_SUBDIR_NAME)}")
                except Exception:
                    pass

            # prepare thumbnails for top frames
            for fidx, score in top_frames:
                if fidx in idxs_read:
                    pos = idxs_read.index(fidx)
                    frame_rgb = full_frames[pos]
                    _, buff = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    b64 = base64.b64encode(buff).decode("utf-8")
                    thumbnail_b64s.append(b64)
                    top_scores.append(float(score))
                    logger.info(f"[Predict] Top frame {fidx} score={score:.4f} (thumbnail prepared)")

            # ---------------------
            # Forgery heuristics: per-frame analyses + temporal metrics
            # ---------------------
            idx_to_pos = {fidx: pos for pos, fidx in enumerate(idxs_read)}
            frame_face_boxes_map = {}
            for face_idx, fidx in enumerate(face_frame_refs):
                box = face_boxes[face_idx]
                if box is not None:
                    frame_face_boxes_map.setdefault(fidx, []).append(box)

            frame_analyses = []
            for face_idx, fidx in enumerate(face_frame_refs):
                pos = idx_to_pos.get(fidx, None)
                if pos is None:
                    continue
                frame_rgb = full_frames[pos]
                box = face_boxes[face_idx]
                if box is None:
                    continue
                x,y,w,h = [int(v) for v in box]
                x2 = min(x+w, frame_rgb.shape[1]-1)
                y2 = min(y+h, frame_rgb.shape[0]-1)
                face_crop = frame_rgb[y:y2, x:x2]
                ny = min(frame_rgb.shape[0], y2 + max(1, h//6))
                neck_crop = frame_rgb[y2:ny, x:x2] if ny > y2 else frame_rgb[y:y2, x:x2]
                try:
                    face_rgb = face_crop.copy()
                    face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
                except Exception:
                    continue
                hfe = high_freq_energy(cv2.resize(face_gray, (128,128)))
                bscore = boundary_artifact_score(cv2.resize(face_gray, (128,128)))
                cs = color_consistency_score(face_rgb, neck_crop if neck_crop.size else face_rgb)
                model_score = float(per_face_probs[face_idx]) if face_idx < len(per_face_probs) else None
                frame_analyses.append({
                    "frame_idx": int(fidx),
                    "boundary_score": float(bscore),
                    "hf_energy": float(hfe),
                    "color_consistency": float(cs),
                    "model_score": model_score
                })

            temporal_metrics = analyze_video_temporal_for_flow(full_frames, frame_face_boxes_map, idxs_read)
            reasons_summary = generate_forgery_reasons_from_frame_analyses(frame_analyses, temporal_metrics, top_k_frames_map=top_frames)

        logger.info(f"[Predict] Finished prediction for {os.path.basename(video_path)}")
        return per_face_probs.tolist(), list(frame_level.items()), label, accuracy, list(zip(thumbnail_b64s, top_scores)), reasons_summary

    except Exception as e:
        traceback.print_exc()
        logger.exception("Prediction error: %s", e)
        reasons_err = {"reasons": [{"type":"error","score":0.0,"reason":str(e)}], "mean_model_score": None, "temporal_metrics": {}}
        return None, None, "ERROR", 0.0, [], reasons_err

# -----------------------
# Uploads cleanup utilities
# -----------------------
def _is_upload_dir(path):
    """Return True if path looks like an upload directory directly under UPLOADS_BASE_DIR."""
    if not os.path.isdir(path):
        return False
    parent = os.path.dirname(path)
    return os.path.abspath(parent) == os.path.abspath(UPLOADS_BASE_DIR)

def cleanup_uploads(base_dir=UPLOADS_BASE_DIR, max_age_seconds=MAX_UPLOAD_AGE_SECONDS, dry_run=False):
    now = time.time()
    removed = []
    try:
        if not os.path.exists(base_dir):
            logger.debug(f"[Cleanup] uploads directory '{base_dir}' does not exist; nothing to do.")
            return removed
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if not _is_upload_dir(full):
                continue
            try:
                mtime = os.path.getmtime(full)
                age = now - mtime
                if age > max_age_seconds:
                    if dry_run:
                        logger.info(f"[Cleanup] (dry) would remove: {full} (age {age:.0f}s)")
                        removed.append(full)
                    else:
                        logger.info(f"[Cleanup] Removing old upload dir: {full} (age {age:.0f}s)")
                        shutil.rmtree(full, ignore_errors=True)
                        removed.append(full)
            except Exception as e:
                logger.exception(f"[Cleanup] Could not check/remove {full}: {e}")
    except Exception:
        logger.exception("[Cleanup] Unexpected error during cleanup")
    return removed

def _cleanup_worker(interval_seconds=CLEANUP_INTERVAL_SECONDS, max_age_seconds=MAX_UPLOAD_AGE_SECONDS, base_dir=UPLOADS_BASE_DIR):
    logger.info(f"[CleanupWorker] Starting: interval={interval_seconds}s, max_age={max_age_seconds}s")
    while True:
        try:
            cleanup_uploads(base_dir=base_dir, max_age_seconds=max_age_seconds, dry_run=False)
        except Exception:
            logger.exception("[CleanupWorker] Exception in worker loop")
        time.sleep(interval_seconds)

def start_cleanup_thread(interval_seconds=CLEANUP_INTERVAL_SECONDS, max_age_seconds=MAX_UPLOAD_AGE_SECONDS):
    t = threading.Thread(target=_cleanup_worker, args=(interval_seconds, max_age_seconds), daemon=True)
    t.start()
    logger.info("[Cleanup] Background cleanup thread started")
    return t

# -----------------------
# Flask route
# -----------------------
@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400

    file = request.files["file"]
    filename = file.filename or f"video_{uuid.uuid4().hex[:6]}.mp4"
    uid = uuid.uuid4().hex[:8]
    save_dir = os.path.join(UPLOADS_BASE_DIR, uid)
    os.makedirs(save_dir, exist_ok=True)
    # create video subdir and move uploaded file there
    video_dir = os.path.join(save_dir, VIDEO_SUBDIR_NAME)
    os.makedirs(video_dir, exist_ok=True)
    temp_save_path = os.path.join(save_dir, filename)
    file.save(temp_save_path)
    final_video_path = os.path.join(video_dir, filename)
    try:
        shutil.move(temp_save_path, final_video_path)
    except Exception:
        # fallback: if move fails, just keep at temp_save_path and use that path
        final_video_path = temp_save_path
    logger.info(f"[Route] Received upload -> {final_video_path}")

    try:
        per_face_probs, frame_level_items, label, confidence, top_frames_info, reasons_summary = predict_single_video_and_frames(
            final_video_path, save_dir=save_dir, top_k=TOP_K_FRAMES, apply_compression=False
        )

        if label == "NO_FACES":
            return jsonify({
                "error": "no_faces",
                "message": "No people detected in the uploaded video. Please upload a video containing a clear face.",
                "forgery_reasons": reasons_summary,
                "paths": {
                    "video_path": final_video_path,
                    "all_frames_path": os.path.join(save_dir, ALL_FRAMES_SUBDIR_NAME) if os.path.exists(os.path.join(save_dir, ALL_FRAMES_SUBDIR_NAME)) else None,
                    "sample_frames_path": os.path.join(save_dir, SAMPLE_FRAMES_SUBDIR_NAME) if os.path.exists(os.path.join(save_dir, SAMPLE_FRAMES_SUBDIR_NAME)) else None,
                    "annotated_frames_path": os.path.join(save_dir, FRAMES_SUBDIR_NAME) if os.path.exists(os.path.join(save_dir, FRAMES_SUBDIR_NAME)) else None,
                }
            }), 200

        frames_b64 = [t[0] for t in top_frames_info] if top_frames_info else []
        frames_scores = [round(float(t[1]), 4) for t in top_frames_info] if top_frames_info else []

        result = {
            "label": label,
            "confidence": confidence,
            "frames": frames_b64,
            "frame_scores": frames_scores,
            "forgery_reasons": reasons_summary,
            "paths": {
                "video_path": final_video_path,
                "all_frames_path": os.path.join(save_dir, ALL_FRAMES_SUBDIR_NAME) if (SAVE_ALL_FRAMES and os.path.exists(os.path.join(save_dir, ALL_FRAMES_SUBDIR_NAME))) else None,
                "sample_frames_path": os.path.join(save_dir, SAMPLE_FRAMES_SUBDIR_NAME) if (SAVE_SAMPLE_FRAMES and os.path.exists(os.path.join(save_dir, SAMPLE_FRAMES_SUBDIR_NAME))) else None,
                "annotated_frames_path": os.path.join(save_dir, FRAMES_SUBDIR_NAME) if (SAVE_FRAMES_TEMP and os.path.exists(os.path.join(save_dir, FRAMES_SUBDIR_NAME))) else None
            }
        }
        return jsonify(result)
    finally:
        # existing cleanup of per-request files if KEEP_FRAMES_AFTER_PRED is False
        try:
            if not KEEP_FRAMES_AFTER_PRED:
                try:
                    if os.path.exists(final_video_path):
                        os.remove(final_video_path)
                except Exception:
                    pass
                try:
                    frames_dir = os.path.join(save_dir, FRAMES_SUBDIR_NAME)
                    if os.path.exists(frames_dir):
                        shutil.rmtree(frames_dir, ignore_errors=True)
                except Exception:
                    pass
                try:
                    sample_dir = os.path.join(save_dir, SAMPLE_FRAMES_SUBDIR_NAME)
                    if os.path.exists(sample_dir):
                        shutil.rmtree(sample_dir, ignore_errors=True)
                except Exception:
                    pass
                try:
                    all_dir = os.path.join(save_dir, ALL_FRAMES_SUBDIR_NAME)
                    if os.path.exists(all_dir):
                        shutil.rmtree(all_dir, ignore_errors=True)
                except Exception:
                    pass
                try:
                    if os.path.isdir(os.path.join(save_dir, VIDEO_SUBDIR_NAME)) and not os.listdir(os.path.join(save_dir, VIDEO_SUBDIR_NAME)):
                        os.rmdir(os.path.join(save_dir, VIDEO_SUBDIR_NAME))
                except Exception:
                    pass
                try:
                    if os.path.isdir(save_dir) and not os.listdir(save_dir):
                        os.rmdir(save_dir)
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Cleanup error: %s", e)

        # run an on-demand cleanup pass (quick, defensive) - non-blocking on main flow
        try:
            # run synchronously (should be fast); set dry_run=False for production deletion
            cleanup_uploads(base_dir=UPLOADS_BASE_DIR, max_age_seconds=MAX_UPLOAD_AGE_SECONDS, dry_run=False)
        except Exception:
            logger.exception("On-demand cleanup failed")

# Debug/test endpoint
@app.route("/upload-test", methods=["POST"])
def upload_test():
    try:
        logger.info("=== /upload-test called ===")
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "no file provided"}), 400
        f = request.files["file"]
        fname = f.filename or f"upload_{uuid.uuid4().hex[:6]}.bin"
        dst_dir = os.path.join(UPLOADS_BASE_DIR, "test")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, fname)
        f.save(dst)
        logger.info("Saved debug upload to: %s", dst)
        return jsonify({"ok": True, "path": dst, "filename": fname}), 200
    except Exception as e:
        traceback.print_exc()
        logger.exception("Upload-test error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # start cleanup thread (daemon)
    try:
        start_cleanup_thread()
    except Exception:
        logger.exception("Failed to start cleanup thread")
    # disable use_reloader to avoid duplicate process logs during development
    app.run(debug=True, use_reloader=False)

