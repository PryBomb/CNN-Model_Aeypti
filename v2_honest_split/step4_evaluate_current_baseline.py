import os
import sys
import re
import cv2
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

CURRENT_MODEL_PATH = r"C:\Users\LAPTOP\OneDrive\ドキュメント\agent skill\Aegypti\CNN_Mosquito_Classifier\mosquito_model.tflite"
TEST_DIR = r"C:\Users\LAPTOP\OneDrive\ドキュメント\agent skill\Aegypti\CNN_Mosquito_Classifier_v2\dataset\test"

def read_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def wilson_score_interval(k, n, confidence=0.95):
    """Computes Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1 - p)) / n + (z**2) / (4 * (n**2)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high

def evaluate():
    print("=" * 70)
    print("STEP 4: Honest Baseline Evaluation of CURRENT mosquito_model.tflite")
    print("=" * 70)

    # 1. Load current TFLite model via bytes (handles unicode paths on Windows)
    print(f"Loading current model: {CURRENT_MODEL_PATH}")
    with open(CURRENT_MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Gather test images
    records = []
    for cls_name, cls_label in [("aegypti", 0), ("albopictus", 1)]:
        cls_dir = os.path.join(TEST_DIR, cls_name)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            m = re.search(r"(?:aegypti|albopictus)(\d+)", fname)
            grp_id = int(m.group(1)) if m else -1
            records.append({
                "path": os.path.join(cls_dir, fname),
                "fname": fname,
                "true_label": cls_label,
                "cls_name": cls_name,
                "grp_id": grp_id
            })

    print(f"Total test images: {len(records)}")

    y_true_img = []
    y_pred_img = []
    y_probs_img = []
    group_data = defaultdict(lambda: {"true_label": None, "probs_aegypti": [], "fnames": []})

    for r in records:
        img = read_image_unicode(r["path"])
        if img is None:
            continue
        # Exactly replicate pi_predict.py preprocessing:
        # 1. Resize to (64, 64) via stretch
        # 2. BGR color order (direct cv2 format)
        # 3. Normalize to [0.0, 1.0]
        # 4. Expand dimensions
        img_resized = cv2.resize(img, (64, 64))
        img_norm = img_resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(img_norm, axis=0)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])[0]

        prob_aegypti = float(output_data[0])
        prob_albopictus = float(output_data[1])
        pred_label = int(np.argmax(output_data))

        y_true_img.append(r["true_label"])
        y_pred_img.append(pred_label)
        y_probs_img.append(output_data)

        gid = (r["cls_name"], r["grp_id"])
        group_data[gid]["true_label"] = r["true_label"]
        group_data[gid]["probs_aegypti"].append(prob_aegypti)
        group_data[gid]["fnames"].append(r["fname"])

    # 3. Image-Level Metrics
    y_true_arr = np.array(y_true_img)
    y_pred_arr = np.array(y_pred_img)
    correct_img = int(np.sum(y_true_arr == y_pred_arr))
    n_img = len(y_true_arr)
    acc_img = correct_img / n_img
    ci_low_img, ci_high_img = wilson_score_interval(correct_img, n_img)

    cm_img = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

    print("\n--- Image-Level Performance on Fair Test Set ---")
    print(f"Sample size: N = {n_img} images")
    print(f"Accuracy   : {acc_img*100:.2f}% ({correct_img}/{n_img})")
    print(f"95% CI     : [{ci_low_img*100:.2f}%, {ci_high_img*100:.2f}%] (Wilson Score)")
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(f"                 Pred Aegypti  Pred Other (Albopictus)")
    print(f"True Aegypti         {cm_img[0, 0]:<14} {cm_img[0, 1]}")
    print(f"True Other (Albo)    {cm_img[1, 0]:<14} {cm_img[1, 1]}")

    print("\nClassification Report (Image-Level):")
    print(classification_report(y_true_arr, y_pred_arr, target_names=["Aedes aegypti", "Aedes albopictus (Other)"], digits=4))

    # 4. Group-Level Metrics (Effective Sample Size)
    y_true_grp = []
    y_pred_grp = []

    for gid, data in group_data.items():
        true_lbl = data["true_label"]
        mean_prob = np.mean(data["probs_aegypti"])
        pred_lbl = 0 if mean_prob >= 0.5 else 1
        y_true_grp.append(true_lbl)
        y_pred_grp.append(pred_lbl)

    y_true_grp = np.array(y_true_grp)
    y_pred_grp = np.array(y_pred_grp)
    n_grp = len(y_true_grp)
    correct_grp = int(np.sum(y_true_grp == y_pred_grp))
    acc_grp = correct_grp / n_grp
    ci_low_grp, ci_high_grp = wilson_score_interval(correct_grp, n_grp)
    cm_grp = confusion_matrix(y_true_grp, y_pred_grp, labels=[0, 1])

    print("\n--- Group-Level Performance (Effective Sample Size: N_groups) ---")
    print(f"Effective Sample Size: N_groups = {n_grp} distinct mosquito groups")
    print(f"  - Aedes aegypti groups   : {np.sum(y_true_grp == 0)}")
    print(f"  - Aedes albopictus groups: {np.sum(y_true_grp == 1)}")
    print(f"Group Accuracy: {acc_grp*100:.2f}% ({correct_grp}/{n_grp})")
    print(f"95% CI (Group): [{ci_low_grp*100:.2f}%, {ci_high_grp*100:.2f}%] (Wilson Score)")
    print("\nConfusion Matrix (Group-Level):")
    print(f"                 Pred Aegypti  Pred Other (Albopictus)")
    print(f"True Aegypti         {cm_grp[0, 0]:<14} {cm_grp[0, 1]}")
    print(f"True Other (Albo)    {cm_grp[1, 0]:<14} {cm_grp[1, 1]}")

    print("\nClassification Report (Group-Level):")
    print(classification_report(y_true_grp, y_pred_grp, target_names=["Aedes aegypti", "Aedes albopictus (Other)"], digits=4))

    # Save results to numpy/json for reporting
    np.savez(os.path.join(r"C:\Users\LAPTOP\OneDrive\ドキュメント\agent skill\Aegypti\CNN_Mosquito_Classifier_v2", "baseline_results.npz"),
             acc_img=acc_img, ci_low_img=ci_low_img, ci_high_img=ci_high_img, cm_img=cm_img,
             acc_grp=acc_grp, ci_low_grp=ci_low_grp, ci_high_grp=ci_high_grp, cm_grp=cm_grp,
             n_img=n_img, n_grp=n_grp)

if __name__ == "__main__":
    evaluate()
