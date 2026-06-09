"""
grader_objectDetect.py  —  auto-grader for the Object Detection worksheet.

Usage (last cell of the notebook):
    from grader_objectDetect import grade
    grade(globals())
"""
import numpy as np


def grade(g):
    score = 0
    total = 100
    results = []

    def check(label, pts, condition):
        nonlocal score
        try:
            if condition():
                score += pts
                results.append(f"[OK  +{pts:2d}] {label}")
            else:
                results.append(f"[FAIL   0] {label}")
        except Exception as exc:
            results.append(f"[ERR    0] {label} — {exc}")

    import tensorflow as tf

    # ── Part 1.1  images (10 pts) ──────────────────────────────────────────
    check("images exists",              2,
          lambda: g.get("images") is not None)
    check("images shape (10,640,640,3)", 5,
          lambda: tuple(g["images"].shape) == (10, 640, 640, 3))
    check("images dtype float32",       3,
          lambda: g["images"].dtype == tf.float32)

    # ── Part 1.3  model + detections (15 pts) ─────────────────────────────
    check("model is YOLOV8Detector",    3,
          lambda: type(g.get("model")).__name__ == "YOLOV8Detector")
    check("detections has required keys", 4,
          lambda: all(k in g.get("detections", {})
                      for k in ("boxes", "confidence", "classes")))
    check("detections boxes shape[0]==10", 4,
          lambda: g["detections"]["boxes"].shape[0] == 10)
    check("detections boxes shape[2]==4",  4,
          lambda: g["detections"]["boxes"].shape[2] == 4)

    # ── Part 1.5/1.6  det_few / det_many (10 pts) ─────────────────────────
    def _n_dets(d, thr=0.1):
        return int(np.sum(np.array(d["confidence"]) > thr))

    check("det_few has correct keys",    2,
          lambda: all(k in g.get("det_few", {})
                      for k in ("boxes", "confidence", "classes")))
    check("det_many has correct keys",   2,
          lambda: all(k in g.get("det_many", {})
                      for k in ("boxes", "confidence", "classes")))
    check("det_few fewer dets than det_many", 6,
          lambda: _n_dets(g["det_few"]) < _n_dets(g["det_many"]))

    # ── Part 2  IoU conceptual (10 pts) ────────────────────────────────────
    check("is_tp_A is True  (iou=0.84 >= 0.50)", 5,
          lambda: g.get("is_tp_A") is True)
    check("is_tp_B is False (iou=0.00 < 0.50)",  5,
          lambda: g.get("is_tp_B") is False)

    # ── Part 3.1  annotation parsing (10 pts) ─────────────────────────────
    check("fnames_ann length == 10",     3,
          lambda: len(g.get("fnames_ann", [])) == 10)
    check("gt_boxes length == 10",       3,
          lambda: len(g.get("gt_boxes", [])) == 10)
    check("gt_classes entries are ints", 4,
          lambda: all(isinstance(c, int) for c in g["gt_classes"][0]))

    # ── Part 3.2  class_counts (10 pts) ───────────────────────────────────
    check("class_counts is a dict",       3,
          lambda: isinstance(g.get("class_counts"), dict))
    check('class_counts["person"] == 16', 7,
          lambda: g["class_counts"].get("person") == 16)

    # ── Part 4.1  pipeline (10 pts) ───────────────────────────────────────
    check("I_ev shape (10,640,640,3)",    5,
          lambda: tuple(g["I_ev"].shape) == (10, 640, 640, 3))
    check("b_ev boxes is RaggedTensor",   5,
          lambda: hasattr(g.get("b_ev", {}).get("boxes"), "flat_values"))

    # ── Part 4.3  COCO metrics (15 pts) ───────────────────────────────────
    check("eval_results is a dict",       3,
          lambda: isinstance(g.get("eval_results"), dict))
    check("mAP50 is a positive number",   7,
          lambda: isinstance(g.get("mAP50"), float) and g["mAP50"] > 0)
    check("mAP50 in valid range [0, 1]",  5,
          lambda: 0 <= g.get("mAP50", -1) <= 1)

    # ── Part 4.4  ap_person (10 pts) ──────────────────────────────────────
    check("n_gt_person == 16",                  3,
          lambda: g.get("n_gt_person") == 16)
    check("ap_person in valid range [0, 1]",    3,
          lambda: isinstance(g.get("ap_person"), float)
                  and 0 <= g["ap_person"] <= 1)
    check("ap_person > 0 (model detects person)", 4,
          lambda: g.get("ap_person", 0) > 0)

    print("\n".join(results))
    print(f"\nTotal: {score}/{total}")
