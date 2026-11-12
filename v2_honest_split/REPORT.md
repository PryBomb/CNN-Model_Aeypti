# CNN Mosquito Classifier — Retraining Report
**Generated:** 2026-11-10  
**Scope:** Lab-photo accuracy only. This report does **NOT** measure chamber accuracy.

---

## 1. Data Leakage Finding

The original training split was made **separately inside each rotation folder**
(0°, 90°, 180°, 270°). Because every specimen photo exists at all four rotations
under the same filename, a photo used for training at 0° would appear in the test
set at 90°. The pixel difference between these pairs was verified to be exactly 0.0.

**Leakage count (measured by filename overlap):**

| Class | Test files sharing a name with a training file |
|---|---|
| *Aedes aegypti* | 360 / 360 (100%) |
| *Aedes albopictus* | 360 / 360 (100%) |

The originally reported **91% accuracy must not be used as an accuracy on new photos.**
It reflects performance on a test set that is a rotated copy of the training set.

---

## 2. Leak-Free Split

Images were grouped by specimen number (regex `(?:aegypti|albopictus)(\d+)` applied
to filenames). Every rotation and letter variant of one specimen number forms one
group. Groups were assigned **whole** to train / val / test so that no specimen
appears on more than one side.

| Split | Groups | *Aegypti* images | *Albopictus* images | Total images |
|---|---|---|---|---|
| Train | 211 (105 aeg + 106 albo) | 1 682 | 1 680 | 3 362 |
| Val   | 45  (22  aeg + 23  albo) | 352   | 368   | 720   |
| Test  | 45  (23  aeg + 22  albo) | 369   | 352   | 721   |

**Leakage verification:** 4 080 unique hashes computed from train+val images
(resized to 64×64, all 4 rotations). Every test image was compared against all
4 080 hashes. **Matches found: 0.**

> [!IMPORTANT]
> The effective independent sample size for statistical claims is **45 groups**
> (one per specimen), not 721 images. Images from the same specimen are correlated.

---

## 3. Results on the Fair Test Set

> [!CAUTION]
> **Invalidation of Old Model "98.2%" Baseline Claim:**  
> The original model (`mosquito_model.tflite`) was trained on ~85% of the full archive without group isolation. Scoring it on any partition of this archive tests photos from specimens the model has already memorized. Consequently, the previously reported 98.2% was an artifact of prior exposure, **not** valid generalization. The legacy model cannot be fairly evaluated on this archive. See `C:\Users\LAPTOP\mosquito_eval\REPORT_v2.md` for the stratified 5-fold cross-validation.

### 3.1 Summary Table

| Model | Preprocessing | Image Acc | 95% CI (Wilson) | Group Acc (n=45) | 95% CI |
|---|---|---|---|---|---|
| **Variant A** `model_v2_variant_a_stretch.keras` | Stretch to 64×64, BGR/255 | 89.04% | [86.55%, 91.12%] | 91.11% | [79.27%, 96.49%] |
| **Variant B** `model_v2_variant_b_letterbox.keras` | Letterbox pad → 64×64, BGR/255 | 91.82% | [89.59%, 93.60%] | 95.56% | [85.17%, 98.77%] |

### 3.3 Per-Class Metrics — Variant A (Stretch)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| *Aedes aegypti* | 0.8919 | 0.8943 | 0.8931 |
| *Aedes albopictus* | 0.8889 | 0.8864 | 0.8876 |

**Confusion matrix (image-level):**

```
                  Pred aegypti   Pred albopictus
True aegypti           330              39
True albopictus         40             312
```

**Confusion matrix (group-level, n=45):**

```
                  Pred aegypti   Pred albopictus
True aegypti            20               3
True albopictus          1              21
```

Variant A: early stopping at epoch 13, **best epoch 7** (val_loss 0.2938, val_acc 88.3%).

### 3.4 Per-Class Metrics — Variant B (Letterbox) ⭐ Best

| Class | Precision | Recall | F1 |
|---|---|---|---|
| *Aedes aegypti* | 0.9235 | 0.9160 | 0.9197 |
| *Aedes albopictus* | 0.9127 | 0.9205 | 0.9165 |

**Confusion matrix (image-level):**

```
                  Pred aegypti   Pred albopictus
True aegypti           338              31
True albopictus         28             324
```

**Confusion matrix (group-level, n=45):**

```
                  Pred aegypti   Pred albopictus
True aegypti            21               2
True albopictus          0              22
```

Variant B: early stopping at epoch 21, **best epoch 15** (val_loss 0.3312, val_acc 88.3%).

---

## 4. Training Curves

Training curves are saved to `training_curves.png` in this folder.

**Variant A training progression (key epochs):**

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1  | 53.8% | 0.6869 | 51.3% | 0.6934 |
| 2  | 69.7% | 0.6019 | 72.4% | 0.5490 |
| 3  | 75.5% | 0.5299 | 75.6% | 0.4953 |
| 6  | 88.5% | 0.2879 | 85.4% | 0.3392 |
| **7** ⭐ | **91.1%** | **0.2356** | **88.3%** | **0.2938** |
| 13 | 97.0% | 0.0936 | 85.1% | 0.4018 ← stopped |

**Variant B training progression (key epochs):**

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1  | 49.1% | 0.6947 | 50.0% | 0.6931 |
| 7  | 52.9% | 0.6920 | 55.4% | 0.6816 |
| 8  | 68.8% | 0.6025 | 68.3% | 0.6145 |
| 10 | 82.1% | 0.4126 | 81.7% | 0.3933 |
| **15** ⭐ | **92.4%** | **0.2000** | **88.3%** | **0.3312** |
| 21 | 96.5% | 0.1029 | 83.6% | 0.5041 ← stopped |

---

## 5. Architecture & Preprocessing

### Model Architecture (Variants A and B — identical)

```
Input (64, 64, 3)
→ Conv2D(32, 3×3, same, relu)  → MaxPool(2×2)
→ Conv2D(64, 3×3, same, relu)  → MaxPool(2×2)
→ Conv2D(128, 3×3, same, relu) → MaxPool(2×2)
→ Flatten → Dense(128, relu) → Dropout(0.3)
→ Dense(64, relu) → Dropout(0.2)
→ Dense(2, softmax)
Total params: 1,150,338
```

**Augmentation (offline NumPy, applied to training data only):**
- Random horizontal flip (p=0.5 per image)
- Brightness jitter ±10% (additive, clipped to [0, 1])
- Training set tripled: 3 362 → 10 086 images

**Early stopping:** monitor=`val_loss`, patience=6, restore_best_weights=True  
**LR schedule:** Adam lr=5e-4, ReduceLROnPlateau(factor=0.5, patience=3)

### Preprocessing per Model

| Model | Resize method | Colour order | Normalisation |
|---|---|---|---|
| Current `mosquito_model.tflite` | `cv2.resize` stretch to 64×64 | BGR | ÷ 255.0 |
| Variant A `model_v2_variant_a_stretch.keras` | `cv2.resize` stretch to 64×64 | BGR | ÷ 255.0 |
| Variant B `model_v2_variant_b_letterbox.keras` ⭐ | Pad to square, then `cv2.resize` to 64×64 | BGR | ÷ 255.0 |

> [!WARNING]
> **Variant B requires a matching letterbox change in the Pi inference code
> (`pi_predict.py`) before deployment.** Using stretch preprocessing at inference
> time with a letterbox-trained model will degrade accuracy. Do not swap the
> `.tflite` file without also updating the preprocessing function.

---

## 6. Exported Files

| File | Description |
|---|---|
| `mosquito_model_v2.tflite` | Variant B converted to TFLite (4.6 MB). **Not deployed.** |
| `model_v2_variant_a_stretch.keras` | Variant A full Keras model |
| `model_v2_variant_b_letterbox.keras` | Variant B full Keras model |
| `dataset_cache_v2_stretch.npz` | Preprocessed stretch data (53 MB) |
| `dataset_cache_v2_letterbox.npz` | Preprocessed letterbox data |
| `baseline_results.npz` | Current model evaluation metrics |
| `retraining_eval_results.npz` | Variant A & B evaluation metrics |
| `training_curves.png` | Loss and accuracy curves for A and B |
| `dataset/train|val|test/aegypti|albopictus/` | Hardlinked fair-split images |

---

## 7. Limitations & Next Steps

> [!CAUTION]
> **This entire report covers lab-photo accuracy only.** The test images are
> studio photographs of pinned or preserved specimens from the Kaggle archive.
> They do **not** represent images taken inside the detection chamber.
> Chamber accuracy has not been measured and may be substantially lower.

**To measure chamber accuracy (requires the Pi):**
1. Collect sharp, clean crops from the Pi camera (no overlays, good focus, diffuse light).
2. Label each crop by hand as *aegypti* or *albopictus*.
3. Keep this chamber test set permanently separate — never use it for training.
4. Score the deployed model on that set. Only this number is relevant to the thesis claim about real-world performance.

**Other recommended improvements:**
- Consider adding a third class ("not a mosquito / other") — but this changes the model's output size and requires updating the detector pipeline.
- If Variant B is deployed, update `pi_predict.py` to use the letterbox resize function.
- Retrain on a combined train+val set (after final model selection) for maximum data use before final deployment.

