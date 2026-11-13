# CNN Mosquito Classifier — Honest Re-Evaluation Report (v2)

**Generated:** 2026-11-12  
**Scope:** Lab-photo accuracy only. This report does **NOT** measure chamber accuracy.  
**Working folder:** `C:\Users\LAPTOP\mosquito_eval` (all scripts and caches) +  
`C:\Users\LAPTOP\OneDrive\ドキュメント\agent skill\Aegypti\CNN_Mosquito_Classifier_v2` (this report)

---

## 0. Original Model Fingerprint Verification

```
SHA-256(mosquito_model.tflite) =
BFB1F4662F6DAA56773F3D8CD09915A81ABB178FF61F31F5B6AE2246CC69A03E
```

First 12 chars: **BFB1F4662F6D** ✅  
The original model file has not been modified.

---

## 1. Data Leakage — Invalidation of Previous Metrics

### 1.1 Original 91% Accuracy Claim

The archive contains every specimen at 4 programmatic rotations (0°, 90°, 180°, 270°). The original project split was made independently inside each rotation folder, so rotated copies of test photos were in training (pixel diff = 0.0 for all 720 test files — 100% contamination). The 91% figure cannot be cited as generalisation performance.

### 1.2 Old Model "98.2%" Baseline Retracted

`mosquito_model.tflite` was trained on ~85% of specimens with no group isolation. Any split of this archive contains specimens the model has already memorised. **The 98.2% figure is an artefact of memorisation. It has been retracted.**

---

## 2. Specimen Grouping & Near-Duplicate Check

Grouping rule: all images sharing the same integer specimen number (and all 4 rotations) form one group and are always assigned together to train, val, or test.

**Letter variants (a/b/c/d within a group)** are distinct photographs of the same physical specimen:  
mean pixel diff ≈ 6.3–6.6 on [0, 255], 92.9% of pixels differ, max diff = 210.

**Inter-group near-duplicate analysis — ZNCC method (CORRECTED)**  
*(ZNCC is the correlation of mean-subtracted, normalised pixel values in [-1.0, +1.0], where **higher = more similar** and 1.0 = identical. The earlier tables and contact sheet reported 1 - ZNCC (a distance) under the label ZNCC. Corrected values: within-group mean 0.576, between-group mean 0.662.)*

| Sample set | n pairs | Mean ZNCC | ± Std | Min ZNCC | Max ZNCC |
|:---|---:|---:|---:|---:|---:|
| Within-group (different shots/bursts of same specimen) | 1,809 | **0.576** | ±0.063 | 0.390 | 1.000 |
| Between-group (closest photo pair across all letters & rots) | 22,500 | **0.662** | ±0.044 | 0.470 | **0.835** |

> [!IMPORTANT]
> The between-group figure is a best match over thousands of candidates and 4 rotations, so it is not comparable to a typical within-group pair; pixel similarity cannot establish specimen identity; the 5 highest-ZNCC between-group pairs were inspected visually by the author; the specimens were damaged (detached legs, scattered parts), so whether they are the same specimen could not be determined; the remaining pairs were not inspected; no groups were merged. Cross-fold specimen leakage therefore cannot be ruled out, and the reported accuracy may be optimistic by an unknown amount.

**Top 20 closest between-group pairs (highest ZNCC correlation, same-class):**

| Rank | Class | Group A | Group B | ZNCC ↑ | SSIM Dist ↓ | dHash Dist ↓ | Rot B |
|---:|:---|:---|:---|---:|---:|---:|---:|
| 1 | albopictus | 125a | 97d | **0.8350** | 0.1464 | 0.2812 | 0° |
| 2 | albopictus | 110a | 48d | **0.8021** | 0.1756 | 0.1875 | 0° |
| 3 | albopictus | 125b | 147a | **0.8006** | 0.1754 | 0.2969 | 0° |
| 4 | aegypti | 128d | 140d | **0.8001** | 0.1758 | 0.2812 | 0° |
| 5 | albopictus | 146a | 36a | **0.7989** | 0.1803 | 0.2344 | 0° |
| 6 | albopictus | 117b | 55a | **0.7968** | 0.1804 | 0.2188 | 0° |
| 7 | albopictus | 151b | 45c | **0.7962** | 0.1863 | 0.2656 | 0° |
| 8 | albopictus | 138b | 147d | **0.7951** | 0.1813 | 0.2500 | 0° |
| 9 | albopictus | 124a | 133c | **0.7935** | 0.1832 | 0.2344 | 0° |
| 10 | albopictus | 34d | 52b | **0.7912** | 0.1916 | 0.1562 | 0° |
| 11 | albopictus | 53b | 97d | **0.7897** | 0.1859 | 0.2969 | 0° |
| 12 | albopictus | 13c | 77c | **0.7891** | 0.1882 | 0.2656 | 0° |
| 13 | albopictus | 107c | 69b | **0.7882** | 0.1894 | 0.3281 | 0° |
| 14 | albopictus | 151a | 9b | **0.7879** | 0.2008 | 0.2500 | 0° |
| 15 | albopictus | 115c | 66a | **0.7873** | 0.1900 | 0.1875 | 0° |
| 16 | aegypti | 118b | 45c | **0.7871** | 0.1880 | 0.2500 | 0° |
| 17 | albopictus | 130d | 99d | **0.7870** | 0.1882 | 0.1875 | 0° |
| 18 | albopictus | 17c | 12d | **0.7859** | 0.1918 | 0.1250 | 0° |
| 19 | aegypti | 128d | 68a | **0.7856** | 0.1892 | 0.1250 | 0° |
| 20 | albopictus | 150a | 9c | **0.7849** | 0.1919 | 0.2812 | 0° |

**Contact Sheets Generated (5 pairs per image, readable size):**
- Between-group top 20 by ZNCC: `closest_between_group_pairs_FIXED_1_to_5.png`, `6_to_10.png`, `11_to_15.png`, `16_to_20.png`
- Between-group top 10 by SSIM distance: `closest_between_group_pairs_FIXED_ssim_1_to_5.png`, `6_to_10.png`
- Between-group top 10 by dHash distance: `closest_between_group_pairs_FIXED_dhash_1_to_5.png`, `6_to_10.png`
- Within-group reference (10 pairs): `within_group_reference_pairs_1_to_5.png`, `6_to_10.png`

---

## 3. Stratified Group 5-Fold Cross-Validation

**Setup:** 301 independent specimen groups (150 *Aedes aegypti*, 151 *Aedes albopictus*; 4,803 total images including all 4 rotations). Stratified Group 5-Fold, seed = 42. No group appears in more than one of train / val / test per fold. Validation carved from training groups (7:1 inner split) for early stopping only; test fold never touched during training.

### 3.0 Verification of Fold Isolation & Zero-Leakage Audit

An automated audit (`fold_overlap_check.py`) verified complete data isolation across all 5 cross-validation folds:
- **Photo-group overlaps (train ∩ test):** Exactly **0** in all 5 folds. All letters (a/b/c/d) and all 4 rotations of every specimen number were kept together.
- **64×64 Image-hash overlaps (train ∩ test):** Exactly **0** in all 5 folds (verified via SHA-256 over all 4,803 images at 64×64 across all 4 rotations; no test hash appeared in training).
- **Validation set source:** Cut strictly from the training partition only (7:1 inner split of training groups); **0** validation groups overlap with the test fold.
- **Deployed model baseline invalidation:** The deployed `mosquito_model.tflite` was trained on the old leaked split (~85% of the archive with no group isolation), so its earlier 98.2% score is **NOT** a valid baseline.

**Architecture (Variants A and B):**  
Conv(32,3×3,ReLU)/MaxPool → Conv(64,3×3,ReLU)/MaxPool → Conv(64,3×3,ReLU) → Flatten → Dropout(0.3) → Dense(64,ReLU) → Dropout(0.2) → Dense(2,softmax). Adam (lr=5e-4), ReduceLROnPlateau, early stopping patience=6, offline augmentation (horizontal flip, ±10% brightness jitter).

- **Variant A (Stretch):** direct bilinear resize to 64×64.
- **Variant B (Letterbox):** center in black square then resize to 64×64. Same training settings as A.
- **Control:** exact `cnn_model_local.py` recipe — same 5-layer architecture, Adam, 20 epochs fixed, **no dropout, no augmentation, no early stopping**.

### 3.1 Per-Fold Group Accuracy — All Three Recipes

| Fold | Test Groups | Control | Variant A (Stretch) | Variant B (Letterbox) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 61 | 93.44% | 86.89% | 86.89% |
| 2 | 60 | **50.00%** ⚠ | 95.00% | 96.67% |
| 3 | 60 | 78.33% | 91.67% | 90.00% |
| 4 | 60 | **50.00%** ⚠ | 91.67% | 91.67% |
| 5 | 60 | 91.67% | 91.67% | 90.00% |

⚠ Folds 2 and 4: Control collapsed — predicted all *aegypti* (Albopictus recall = 0.0%). Intrinsic instability of original recipe without early stopping.

### 3.2 Per-Fold Image Accuracy — All Three Recipes

| Fold | Control | Variant A (Stretch) | Variant B (Letterbox) |
|:---:|:---:|:---:|:---:|
| 1 | 84.38% | 83.76% | 86.43% |
| 2 | 50.10% | 90.96% | 90.54% |
| 3 | 76.88% | 89.27% | 87.50% |
| 4 | 50.63% | 87.45% | 89.24% |
| 5 | 86.25% | 86.67% | 87.50% |

### 3.3 Summary — Control vs Variant A vs Variant B

Numbers are for **unthresholded** predictions. "Recall" here = **strict recall** (see §4 for definition).

| Metric | Control | Variant A (Stretch) | Variant B (Letterbox) |
|:---|:---:|:---:|:---:|
| **Mean Image Accuracy** | 69.65% | 87.62% | 88.24% |
| **Std Image Accuracy** | ±17.95% | ±2.72% | ±1.63% |
| **CV (Image Acc)** | **25.8%** | **3.1%** | **1.9%** |
| **Mean Group Accuracy** | 72.69% | **91.38%** | **91.04%** |
| **Std Group Accuracy** | ±21.52% | ±2.90% | ±3.59% |
| **CV (Group Acc)** | **29.6%** | **3.2%** | **3.9%** |
| Mean *Aegypti* Precision | 0.7709 | 0.8822 | 0.8790 |
| Mean *Aegypti* **Recall (strict)** | 0.8122 | 0.8784 | 0.8905 |

*CV = Std / Mean × 100%. Lower = more reliable across folds.*

**Paired statistical comparison (group accuracy, 5 folds):**

| Comparison | Mean diff | Paired t p | Wilcoxon p |
|:---|:---:|:---:|:---:|
| A vs Control | +18.69% | 0.1519 | 0.2500 |
| B vs Control | +18.36% | 0.1699 | 0.3125 |
| B vs A | −0.33% | — | **1.0000** |

> [!NOTE]
> A vs Control and B vs Control do not reach α = 0.05 because n = 5 folds is underpowered and the two collapsed Control folds inflate variance. The key finding is **reliability**: Control CV = 29.6% vs Variant A CV = 3.2%. B vs A: p = 1.0 — statistically indistinguishable.

---

## 4. Deployment Threshold Analysis (Confidence ≥ 0.70) — Corrected Definitions

The detector only labels a detection when classifier confidence ≥ 0.70; below this it outputs "species unconfirmed".

| Term | Formula | Meaning |
|:---|:---|:---|
| **Strict Recall** | TP / (TP + FN_all) | Of all true *Aegypti*, what fraction get a confident *Aegypti* label? FN includes unconfirmed true positives. **Operationally honest.** |
| *Conditional Recall* | TP / (TP + FN_confirmed) | Of confirmed *Aegypti* predictions, what fraction are correct? Denominator excludes unconfirmed — always higher than strict recall; listed for completeness. |

> [!CAUTION]
> The previous version of this report used conditional recall and labelled it "recall". That was incorrect. Strict recall is the honest figure.

### 4.1 Threshold Metrics @ ≥ 0.70 — All Three Recipes

| Metric | Control | Variant A | Variant B |
|:---|:---:|:---:|:---:|
| Mean Unconfirmed Share | 45.43% ± 49.81% | 12.66% ± 3.92% | 10.61% ± 1.30% |
| Mean *Aegypti* Precision | 0.5787 ± 0.5286 | **0.9208 ± 0.0456** | **0.9161 ± 0.0297** |
| **Mean *Aegypti* Recall (strict)** | 0.3618 ± 0.3397 | **0.8018 ± 0.1033** | **0.8306 ± 0.0717** |
| *Mean Aegypti Conditional Recall* | *0.4246 ± 0.3983* | *0.9172 ± 0.0466* | *0.9205 ± 0.0413* |

**Operational trade-off at ≥ 0.70 (Variant A):**

| | No threshold | @ ≥ 0.70 |
|:---|:---:|:---:|
| *Aegypti* Precision | 88.2% | 92.1% |
| *Aegypti* **Strict Recall** | 87.8% | **80.2%** |
| Unconfirmed fraction | 0% | 12.7% |

Gating at 0.70 gains +3.9 pp precision at the cost of −7.6 pp strict recall; ~1 in 8 frames becomes "species unconfirmed".

---

## 5. TFLite Export & Parity Check

Fold 1 Variant A → `mosquito_model_v2_check.tflite` (4.61 MB, in `C:\Users\LAPTOP\mosquito_eval`).  
Variant B → `CNN_Mosquito_Classifier_v2\mosquito_model_v2.tflite` (4.6 MB, **not deployed**).

| Check | Value |
|:---|:---:|
| Parity sample | 200 test images |
| Top-1 agreement | **200 / 200 (100.00%)** |
| Max absolute prob diff | **2.384 × 10⁻⁷** |

### Preprocessing Specification (Document Only — Do Not Edit Detector)

```
Target Input Shape   : [1, 64, 64, 3]
Data Type            : Float32
Colour Order         : BGR  (standard cv2.imread — no channel swap)
Normalisation        : img / 255.0  → [0.0, 1.0]
Resize (Variant A)   : cv2.resize(img, (64, 64))  — direct stretch, no padding
Resize (Variant B)   : pad shorter side to square with black, then cv2.resize to 64×64
Output               : [1, 2]  softmax probabilities
Class 0              : Aedes aegypti
Class 1              : Aedes albopictus
Confidence gate      : max(prob) >= 0.70 → accept argmax label
                       max(prob) <  0.70 → "species unconfirmed"
```

> [!WARNING]
> Variant B requires a matching letterbox change in `pi_predict.py` before deployment. Do not swap the `.tflite` without updating preprocessing.

---

## 6. Limitations

> [!CAUTION]
> **All results in this report are lab-photo accuracy only.** Chamber accuracy has not been measured and may be substantially lower.

1. **Specimen identity cannot be verified from the dataset.** Different specimen numbers may show the same physical mosquito photographed twice, or close relatives. The between-group figure is a best match over thousands of candidates and 4 rotations, so it is not comparable to a typical within-group pair; pixel similarity cannot establish specimen identity; the 5 highest-ZNCC between-group pairs were inspected visually by the author; the specimens were damaged (detached legs, scattered parts), so whether they are the same specimen could not be determined; the remaining pairs were not inspected; no groups were merged. Cross-fold specimen leakage therefore cannot be ruled out, and the reported accuracy may be optimistic by an unknown amount.

2. **Dataset source and photography protocol are unknown.** The Kaggle archive provides no metadata about the number of unique individuals, capture conditions, or whether specimens were pinned, preserved, or live. Dataset variability may not represent variability in chamber captures.

3. **All results are lab-photo accuracy, not chamber accuracy.** Test images are studio photographs. The detection chamber produces motion-blurred, variable-distance, and occluded frames under different lighting. Chamber accuracy must be measured independently (collect ≥ 50–100 labelled chamber crops and score the deployed model against them).

4. **No "not a mosquito" class.** The classifier always outputs one of two species. Any non-mosquito object presented to the classifier may receive a confident species label. A rejection class would require retraining and pipeline changes.

5. **5-fold CV with n = 301 groups is underpowered for formal hypothesis testing.** Paired tests (§3.3) do not reach α = 0.05. The correct interpretation is that the new recipe is more reliable (CV 3.2% vs 29.6%), not that mean accuracy is certifiably higher.

---

## 7. Files

**In `C:\Users\LAPTOP\mosquito_eval\` (evaluation workspace):**

| File | Size | Description |
|:---|---:|:---|
| `step1_grouping_check.py` | 6.3 KB | Original ahash grouping check (superseded) |
| `step1_duplicate_check_rigorous.py` | 11.2 KB | ZNCC/SSIM/dHash inter-group analysis |
| `step2_4_kfold_and_checks.py` | 24.6 KB | 5-fold CV, Variants A and B, TFLite export |
| `step2_missing_control_and_metrics.py` | 17.3 KB | Control recipe + corrected threshold metrics |
| `eval_cache_stretch.npz` | 52.9 MB | 64×64 stretch image cache |
| `eval_cache_letterbox.npz` | 42.6 MB | 64×64 letterbox image cache |
| `fold{1-5}_variant_A_stretch.keras` | ~13.2 MB × 5 | Fold models, Variant A |
| `fold{1-5}_variant_B_letterbox.keras` | ~13.2 MB × 5 | Fold models, Variant B |
| `fold{1-5}_control.keras` | ~7.4 MB × 5 | Fold models, Control |
| `fold_metrics.npz` | 1.1 KB | Numerical fold results |
| `fold_results_summary.json` | 1.7 KB | A vs B summary and paired tests |
| `control_and_corrected_metrics.json` | 2.4 KB | Control results + corrected threshold |
| `mosquito_model_v2_check.tflite` | 4.61 MB | TFLite export of Fold 1 Variant A |
| `closest_between_group_pairs.png` | 2.18 MB | Contact sheet, 10 closest between-group pairs |
| `REPORT_v2.md` | — | Copy of this document (also in CNN_Mosquito_Classifier_v2) |

**In `CNN_Mosquito_Classifier_v2\` (this folder):**

| File | Description |
|:---|:---|
| `mosquito_model_v2.tflite` | Variant B TFLite export (4.6 MB). **Not deployed.** |
| `model_v2_variant_a_stretch.keras` | Variant A full Keras model (single-split, not 5-fold) |
| `model_v2_variant_b_letterbox.keras` | Variant B full Keras model (single-split, not 5-fold) |
| `dataset_cache_v2_stretch.npz` | Preprocessed stretch data (53 MB) |
| `dataset_cache_v2_letterbox.npz` | Preprocessed letterbox data (43 MB) |
| `REPORT_v2.md` | This document |

**Files NOT modified:** `Downloads\archive`, `mosquito_model.tflite` (SHA-256 `BFB1F4662F6D...`), `cnn_model_local.py`, `pi_predict.py`, `config.py`, any detector/tracker/status-server file, any `.db` or captures file.

