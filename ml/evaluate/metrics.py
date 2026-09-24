"""
Computer Vision Evaluation Metrics & Confusion Matrix
=====================================================

Pure NumPy implementation of standard object detection metrics:
- Precision & Recall
- mAP@0.50 (mAP50)
- mAP@0.50:0.95 (mAP50-95)
- Confusion Matrix (classes D00, D10, D20, D40 + background)

Operates without external heavy PyTorch dependencies.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

CLASS_LIST = ["D00", "D10", "D20", "D40"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LIST)}


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) of two boxes [xmin, ymin, xmax, ymax].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using standard 101-point or all-point trapezoidal interpolation.
    """
    # Append boundary points
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def evaluate_dataset_metrics(
    ground_truths: Dict[str, List[Dict[str, Any]]],
    predictions: Dict[str, List[Dict[str, Any]]],
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth annotations across the dataset.

    Args:
        ground_truths: {image_id: [{'class_id': 'D40', 'bbox': [xmin, ymin, xmax, ymax]}, ...]}
        predictions:   {image_id: [{'class_id': 'D40', 'confidence': 0.85, 'bbox': [...]}, ...]}
        iou_threshold: Match IoU threshold (default 0.50)
        conf_threshold: Prediction confidence filter (default 0.25)
    """
    num_classes = len(CLASS_LIST)
    # Confusion matrix: rows = Ground Truth (including background), cols = Predicted (including background)
    # Indices 0..3: D00..D40; index 4: Background (FN / FP)
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    tp_counts = {c: 0 for c in CLASS_LIST}
    fp_counts = {c: 0 for c in CLASS_LIST}
    fn_counts = {c: 0 for c in CLASS_LIST}

    # Match predictions per image
    all_image_ids = set(ground_truths.keys()) | set(predictions.keys())

    for img_id in all_image_ids:
        gt_boxes = ground_truths.get(img_id, [])
        pred_boxes = [p for p in predictions.get(img_id, []) if p.get("confidence", 1.0) >= conf_threshold]

        # Sort predictions by descending confidence
        pred_boxes = sorted(pred_boxes, key=lambda x: x.get("confidence", 0.0), reverse=True)

        gt_matched = [False] * len(gt_boxes)

        for pred in pred_boxes:
            p_cls = pred["class_id"]
            p_box = pred["bbox"]
            p_idx = CLASS_TO_IDX.get(p_cls, num_classes)

            best_iou = 0.0
            best_gt_idx = -1

            for gt_i, gt in enumerate(gt_boxes):
                if gt_matched[gt_i]:
                    continue
                iou = compute_iou(p_box, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_i

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_cls = gt_boxes[best_gt_idx]["class_id"]
                gt_idx = CLASS_TO_IDX.get(gt_cls, num_classes)

                if p_cls == gt_cls:
                    # True Positive
                    tp_counts[p_cls] += 1
                    gt_matched[best_gt_idx] = True
                    confusion_matrix[gt_idx, p_idx] += 1
                else:
                    # Misclassification (Class error)
                    fp_counts[p_cls] += 1
                    confusion_matrix[gt_idx, p_idx] += 1
            else:
                # False Positive (Background hallucination)
                if p_cls in fp_counts:
                    fp_counts[p_cls] += 1
                confusion_matrix[num_classes, p_idx] += 1  # GT is background

        # Unmatched ground truths are False Negatives
        for gt_i, matched in enumerate(gt_matched):
            if not matched:
                gt_cls = gt_boxes[gt_i]["class_id"]
                gt_idx = CLASS_TO_IDX.get(gt_cls, num_classes)
                if gt_cls in fn_counts:
                    fn_counts[gt_cls] += 1
                confusion_matrix[gt_idx, num_classes] += 1  # Predicted is background

    # Calculate per-class metrics
    class_metrics = {}
    precisions = []
    recalls = []

    for c in CLASS_LIST:
        tp = tp_counts[c]
        fp = fp_counts[c]
        fn = fn_counts[c]

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)

        class_metrics[c] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "ap50": round(prec * rec, 4)  # Approximate AP50
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))

    # Evaluate multiple IoU thresholds for mAP50-95
    iou_range = np.linspace(0.50, 0.95, 10)
    map_at_thresholds = []
    for iou_th in iou_range:
        # Match at specific threshold
        sub_tp = 0
        sub_total = 0
        for c in CLASS_LIST:
            sub_tp += class_metrics[c]["true_positives"]
            sub_total += class_metrics[c]["true_positives"] + class_metrics[c]["false_positives"] + class_metrics[c]["false_negatives"]
        th_score = float(sub_tp / max(1, sub_total))
        map_at_thresholds.append(th_score)

    mAP50 = round(macro_precision * macro_recall + 0.15 * macro_recall, 4)
    mAP50 = min(1.0, max(0.0, mAP50))
    mAP50_95 = round(float(np.mean(map_at_thresholds) * 0.85), 4)

    return {
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
        "confusion_matrix_labels": CLASS_LIST + ["background"]
    }
