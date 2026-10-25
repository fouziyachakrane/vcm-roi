import os, time
import xml.etree.ElementTree as ET
import numpy as np
from ultralytics import YOLO

N_FRAMES = 200
VEHICLE_CLASSES = [2, 5, 7]   # COCO : car, bus, truck
IMGSZ = 640
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolo11n.pt")
    return _model

def box_from(b):
    x, y, w, h = [float(b.attrib[k]) for k in ("left", "top", "width", "height")]
    return [x, y, x + w, y + h]

def load_gt(seq):
    root = ET.parse(f"data/{seq}/{seq}.xml").getroot()
    ignored = [box_from(b) for r in root.iter("ignored_region") for b in r.iter("box")]
    gt = {}
    for frame in root.iter("frame"):
        num = int(frame.attrib["num"])
        if num <= N_FRAMES:
            gt[num] = [box_from(t.find("box")) for t in frame.iter("target")]
    return gt, ignored

def in_ignored(box, ignored):
    """Vrai si plus de la moitié de la boîte est dans une zone ignorée."""
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1e-9)
    inter = 0.0
    for a, b, c, d in ignored:
        w = min(x2, c) - max(x1, a)
        h = min(y2, d) - max(y1, b)
        if w > 0 and h > 0:
            inter += w * h
    return inter / area > 0.5

def iou_one_to_many(b, boxes):
    boxes = np.array(boxes, dtype=float)
    w = np.minimum(b[2], boxes[:, 2]) - np.maximum(b[0], boxes[:, 0])
    h = np.minimum(b[3], boxes[:, 3]) - np.maximum(b[1], boxes[:, 1])
    inter = np.clip(w, 0, None) * np.clip(h, 0, None)
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_g = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_b + area_g - inter + 1e-9)

def average_precision(preds, gt, thr):
    npos = sum(len(v) for v in gt.values())
    preds = sorted(preds, key=lambda p: -p[1])
    used = {f: np.zeros(len(v), bool) for f, v in gt.items()}
    tp = np.zeros(len(preds))
    for i, (f, conf, b) in enumerate(preds):
        g = gt.get(f, [])
        if len(g) == 0:
            continue
        ious = iou_one_to_many(b, g)
        j = int(ious.argmax())
        if ious[j] >= thr and not used[f][j]:
            used[f][j] = True
            tp[i] = 1
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    rec = tpc / npos
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, x, side="left")
    return float(mpre[idx].mean())

def evaluate(seq, image_dir, name):
    """Évalue YOLO sur les images de image_dir, avec les annotations de seq."""
    gt, ignored = load_gt(seq)
    files = sorted(os.listdir(image_dir))
    ext = os.path.splitext(files[0])[1]
    model = get_model()
    model(f"{image_dir}/{files[0]}", device="cpu", verbose=False)  # échauffement

    preds, times = [], []
    for num in range(1, N_FRAMES + 1):
        path = f"{image_dir}/img{num:05d}{ext}"
        t0 = time.perf_counter()
        r = model(path, device="cpu", conf=0.001, classes=VEHICLE_CLASSES,
                  imgsz=IMGSZ, verbose=False)[0]
        times.append(time.perf_counter() - t0)
        for box, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            if not in_ignored(box, ignored):
                preds.append((num, conf, box))

    ap50 = average_precision(preds, gt, 0.5)
    ap = float(np.mean([average_precision(preds, gt, t) for t in np.arange(0.5, 0.96, 0.05)]))
    ms = 1000 * float(np.mean(times))
    n_gt = sum(len(v) for v in gt.values())
    print(f"[{seq} {name}] vehicules annotes={n_gt}  AP50={ap50:.3f}  AP50-95={ap:.3f}  {ms:.0f} ms/image")
    return ap50, ap, ms