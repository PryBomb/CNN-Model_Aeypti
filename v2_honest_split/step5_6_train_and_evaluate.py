import os
import sys
import re
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

V2_DIR = r"C:\Users\LAPTOP\OneDrive\ドキュメント\agent skill\Aegypti\CNN_Mosquito_Classifier_v2"
DATASET_DIR = os.path.join(V2_DIR, "dataset")

def read_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def letterbox_image(img, target_size=(64, 64)):
    h, w = img.shape[:2]
    max_dim = max(h, w)
    canvas = np.zeros((max_dim, max_dim, 3), dtype=img.dtype)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return cv2.resize(canvas, target_size)

def wilson_score_interval(k, n, confidence=0.95):
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1 - p)) / n + (z**2) / (4 * (n**2)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high

def load_or_create_cache(variant_name, letterbox=False):
    cache_path = os.path.join(V2_DIR, f"dataset_cache_v2_{variant_name}.npz")
    if os.path.exists(cache_path):
        print(f"Loading {variant_name} data from cache: {cache_path}")
        data = np.load(cache_path)
        return (data["X_train"], data["Y_train"],
                data["X_val"], data["Y_val"],
                data["X_test"], data["Y_test"],
                data["test_meta"])

    print(f"Creating cache for {variant_name} (letterbox={letterbox})...")
    splits = {}
    test_meta = []

    for split in ["train", "val", "test"]:
        X_list, Y_list = [], []
        for cls_name, cls_label in [("aegypti", 0), ("albopictus", 1)]:
            cls_dir = os.path.join(DATASET_DIR, split, cls_name)
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fpath = os.path.join(cls_dir, fname)
                img = read_image_unicode(fpath)
                if img is None:
                    continue
                if letterbox:
                    proc = letterbox_image(img, (64, 64))
                else:
                    proc = cv2.resize(img, (64, 64))

                # Normalize to [0.0, 1.0], BGR (matches original model)
                proc_norm = proc.astype(np.float32) / 255.0
                X_list.append(proc_norm)
                Y_list.append(cls_label)

                if split == "test":
                    m = re.search(r"(?:aegypti|albopictus)(\d+)", fname)
                    grp_id = int(m.group(1)) if m else -1
                    test_meta.append(f"{cls_name}:{grp_id}:{fname}")

        splits[f"X_{split}"] = np.array(X_list, dtype=np.float32)
        splits[f"Y_{split}"] = np.array(Y_list, dtype=np.int32)
        print(f"  {split}: {len(X_list)} images loaded.")

    test_meta_arr = np.array(test_meta)
    np.savez_compressed(cache_path,
                        X_train=splits["X_train"], Y_train=splits["Y_train"],
                        X_val=splits["X_val"], Y_val=splits["Y_val"],
                        X_test=splits["X_test"], Y_test=splits["Y_test"],
                        test_meta=test_meta_arr)
    print(f"Cache saved to {cache_path}")
    return (splits["X_train"], splits["Y_train"],
            splits["X_val"], splits["Y_val"],
            splits["X_test"], splits["Y_test"],
            test_meta_arr)

def numpy_augment(X, Y, seed=42):
    """
    Pure-NumPy augmentation applied to training data before fitting.
    Avoids any Keras augmentation layer that might destabilise training.
    Operations (each applied randomly with p=0.5 per image):
      - horizontal flip
      - brightness jitter +-10% (additive, clipped)
    Returns augmented X, Y (doubles the dataset).
    """
    rng = np.random.default_rng(seed)
    X_aug_list = [X]
    Y_aug_list = [Y]

    # Horizontal flips
    flip_mask = rng.random(len(X)) < 0.5
    X_flip = X.copy()
    X_flip[flip_mask] = X[flip_mask, :, ::-1, :]
    X_aug_list.append(X_flip)
    Y_aug_list.append(Y)

    # Brightness jitter
    delta = rng.uniform(-0.10, 0.10, size=(len(X), 1, 1, 1)).astype(np.float32)
    X_bright = np.clip(X + delta, 0.0, 1.0)
    X_aug_list.append(X_bright)
    Y_aug_list.append(Y)

    X_out = np.concatenate(X_aug_list, axis=0)
    Y_out = np.concatenate(Y_aug_list, axis=0)

    # Shuffle
    idx = rng.permutation(len(X_out))
    return X_out[idx], Y_out[idx]

def build_model():
    """Build the CNN classifier WITHOUT Keras augmentation layers."""
    model = tf.keras.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(2, activation="softmax")
    ], name="mosquito_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def evaluate_predictions(y_true, y_pred_probs, test_meta):
    y_pred = np.argmax(y_pred_probs, axis=1)
    n_img = len(y_true)
    correct_img = int(np.sum(y_true == y_pred))
    acc_img = correct_img / n_img
    ci_low_img, ci_high_img = wilson_score_interval(correct_img, n_img)
    cm_img = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Group-level aggregation
    group_data = defaultdict(lambda: {"true_label": None, "probs_aegypti": []})
    for i, meta in enumerate(test_meta):
        parts = meta.split(":")
        cls_name = parts[0]
        grp_id = int(parts[1])
        gid = (cls_name, grp_id)
        group_data[gid]["true_label"] = y_true[i]
        group_data[gid]["probs_aegypti"].append(y_pred_probs[i, 0])

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

    return {
        "acc_img": acc_img,
        "ci_img": (ci_low_img, ci_high_img),
        "cm_img": cm_img,
        "y_true_img": y_true,
        "y_pred_img": y_pred,
        "acc_grp": acc_grp,
        "ci_grp": (ci_low_grp, ci_high_grp),
        "cm_grp": cm_grp,
        "n_img": n_img,
        "n_grp": n_grp
    }

def main():
    print("=" * 70)
    print("STEP 5: Retraining Variant A (Stretch) and Variant B (Letterbox)")
    print("=" * 70)

    # 1. Prepare data for Variant A
    print("\n--- Preparing Variant A (Stretch 64x64) ---")
    X_tr_a, Y_tr_a, X_val_a, Y_val_a, X_te_a, Y_te_a, meta_a = load_or_create_cache("stretch", letterbox=False)

    # Apply NumPy augmentation (triples the training set)
    print("Applying NumPy augmentation to training data...")
    X_tr_a_aug, Y_tr_a_aug = numpy_augment(X_tr_a, Y_tr_a, seed=42)
    print(f"  Augmented train size: {len(X_tr_a_aug)} (from {len(X_tr_a)})")

    # 2. Train Variant A
    print("\nTraining Variant A (Drop-in Replacement: Stretch + Dropout + Aug)...")
    tf.random.set_seed(42)
    np.random.seed(42)
    model_a = build_model()
    model_a.summary()

    es_a = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    lr_reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    hist_a = model_a.fit(
        X_tr_a_aug, Y_tr_a_aug,
        validation_data=(X_val_a, Y_val_a),
        epochs=40,
        batch_size=64,
        callbacks=[es_a, lr_reduce],
        verbose=2
    )

    model_a_path = os.path.join(V2_DIR, "model_v2_variant_a_stretch.keras")
    model_a.save(model_a_path)
    print(f"Variant A saved to {model_a_path}")

    # 3. Prepare data for Variant B
    print("\n--- Preparing Variant B (Letterbox Square Pad -> 64x64) ---")
    X_tr_b, Y_tr_b, X_val_b, Y_val_b, X_te_b, Y_te_b, meta_b = load_or_create_cache("letterbox", letterbox=True)

    print("Applying NumPy augmentation to training data...")
    X_tr_b_aug, Y_tr_b_aug = numpy_augment(X_tr_b, Y_tr_b, seed=42)
    print(f"  Augmented train size: {len(X_tr_b_aug)} (from {len(X_tr_b)})")

    # 4. Train Variant B
    print("\nTraining Variant B (Letterbox + Dropout + Aug)...")
    tf.random.set_seed(42)
    np.random.seed(42)
    model_b = build_model()

    es_b = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    lr_reduce_b = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    hist_b = model_b.fit(
        X_tr_b_aug, Y_tr_b_aug,
        validation_data=(X_val_b, Y_val_b),
        epochs=40,
        batch_size=64,
        callbacks=[es_b, lr_reduce_b],
        verbose=2
    )

    model_b_path = os.path.join(V2_DIR, "model_v2_variant_b_letterbox.keras")
    model_b.save(model_b_path)
    print(f"Variant B saved to {model_b_path}")

    # 5. Plot Training Curves
    print("\nGenerating training curves plot...")
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(hist_a.history["loss"], "b--", label="Variant A Train Loss")
    plt.plot(hist_a.history["val_loss"], "b-", label="Variant A Val Loss")
    plt.plot(hist_b.history["loss"], "g--", label="Variant B Train Loss")
    plt.plot(hist_b.history["val_loss"], "g-", label="Variant B Val Loss")
    plt.title("Cross-Entropy Loss (Early Stopping on Val)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(hist_a.history["accuracy"], "b--", label="Variant A Train Acc")
    plt.plot(hist_a.history["val_accuracy"], "b-", label="Variant A Val Acc")
    plt.plot(hist_b.history["accuracy"], "g--", label="Variant B Train Acc")
    plt.plot(hist_b.history["val_accuracy"], "g-", label="Variant B Val Acc")
    plt.title("Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(V2_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Training curves saved to: {curves_path}")

    # 6. Evaluate A and B on the Fair Test Set
    print("\n" + "=" * 70)
    print("STEP 6: Evaluating Variant A and Variant B on Fair Test Set")
    print("=" * 70)

    probs_a = model_a.predict(X_te_a, batch_size=64)
    eval_a = evaluate_predictions(Y_te_a, probs_a, meta_a)

    probs_b = model_b.predict(X_te_b, batch_size=64)
    eval_b = evaluate_predictions(Y_te_b, probs_b, meta_b)

    print("\n>>> VARIANT A (STRETCH) RESULTS <<<")
    print(f"Image-Level Accuracy: {eval_a['acc_img']*100:.2f}% (95% CI: [{eval_a['ci_img'][0]*100:.2f}%, {eval_a['ci_img'][1]*100:.2f}%])")
    print("Confusion Matrix (Image-Level):\n", eval_a['cm_img'])
    print("\nClassification Report (Image-Level):")
    print(classification_report(eval_a['y_true_img'], eval_a['y_pred_img'], target_names=["Aedes aegypti", "Aedes albopictus"], digits=4))
    print(f"Group-Level Accuracy: {eval_a['acc_grp']*100:.2f}% (n={eval_a['n_grp']} groups, 95% CI: [{eval_a['ci_grp'][0]*100:.2f}%, {eval_a['ci_grp'][1]*100:.2f}%])")
    print("Confusion Matrix (Group-Level):\n", eval_a['cm_grp'])

    print("\n>>> VARIANT B (LETTERBOX) RESULTS <<<")
    print(f"Image-Level Accuracy: {eval_b['acc_img']*100:.2f}% (95% CI: [{eval_b['ci_img'][0]*100:.2f}%, {eval_b['ci_img'][1]*100:.2f}%])")
    print("Confusion Matrix (Image-Level):\n", eval_b['cm_img'])
    print("\nClassification Report (Image-Level):")
    print(classification_report(eval_b['y_true_img'], eval_b['y_pred_img'], target_names=["Aedes aegypti", "Aedes albopictus"], digits=4))
    print(f"Group-Level Accuracy: {eval_b['acc_grp']*100:.2f}% (n={eval_b['n_grp']} groups, 95% CI: [{eval_b['ci_grp'][0]*100:.2f}%, {eval_b['ci_grp'][1]*100:.2f}%])")
    print("Confusion Matrix (Group-Level):\n", eval_b['cm_grp'])

    # 7. Convert the Best Model to TFLite as mosquito_model_v2.tflite
    best_model_name = "Variant B (Letterbox)" if eval_b["acc_img"] >= eval_a["acc_img"] else "Variant A (Stretch)"
    best_model = model_b if eval_b["acc_img"] >= eval_a["acc_img"] else model_a
    print(f"\nBest Model is: {best_model_name}")

    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(V2_DIR, "mosquito_model_v2.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Exported TFLite model to: {tflite_path} ({len(tflite_model)} bytes)")

    # 8. Save comparison metrics for report generator
    np.savez(os.path.join(V2_DIR, "retraining_eval_results.npz"),
             best_model_name=np.array(best_model_name),
             eval_a_acc_img=eval_a["acc_img"],
             eval_a_ci_img=np.array(eval_a["ci_img"]),
             eval_a_cm_img=eval_a["cm_img"],
             eval_a_acc_grp=eval_a["acc_grp"],
             eval_a_ci_grp=np.array(eval_a["ci_grp"]),
             eval_a_cm_grp=eval_a["cm_grp"],
             eval_b_acc_img=eval_b["acc_img"],
             eval_b_ci_img=np.array(eval_b["ci_img"]),
             eval_b_cm_img=eval_b["cm_img"],
             eval_b_acc_grp=eval_b["acc_grp"],
             eval_b_ci_grp=np.array(eval_b["ci_grp"]),
             eval_b_cm_grp=eval_b["cm_grp"])
    print("\nAll metrics saved. Done!")

if __name__ == "__main__":
    main()
