"""
step1_duplicate_check_rigorous.py
Rigorous Near-Duplicate and Specimen Group Analysis across Groups.

Corrected direction:
- ZNCC = np.dot(x_norm, y_norm) in [-1.0, 1.0]. HIGHER = MORE SIMILAR (1.0 = identical).
- SSIM distance = 1.0 - SSIM. LOWER = MORE SIMILAR (0.0 = identical).
- dHash distance = normalized Hamming distance. LOWER = MORE SIMILAR (0.0 = identical).
"""

import os, sys, re, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8')

ARCHIVE  = r"C:\Users\LAPTOP\Downloads\archive"
EVAL_DIR = r"C:\Users\LAPTOP\mosquito_eval"
V2_DIR   = r"C:\Users\LAPTOP\OneDrive\ドキュメント\agent skill\Aegypti\CNN_Mosquito_Classifier_v2"

def check_disk(tag=""):
    import shutil
    free = shutil.disk_usage("C:\\").free / 1e9
    print(f"[disk] {tag}  C: free = {free:.2f} GB", flush=True)
    if free < 8.0:
        sys.exit("STOPPING: C: free space dropped below 8 GB.")
    return free

def center_square(raw, sz=64):
    h, w = raw.shape[:2]
    if w >= h:
        sq = raw[:, (w-h)//2 : (w-h)//2 + h]
    else:
        sq = raw[(h-w)//2 : (h-w)//2 + w, :]
    return cv2.resize(sq, (sz, sz))

def ssim_metric(x, y):
    # returns SSIM distance = 1.0 - SSIM
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (0.01)**2
    c2 = (0.03)**2
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(1.0 - (num / den))

def dhash_bits(im_gray, sz=(9, 8)):
    small = cv2.resize(im_gray, sz)
    return (small[:, 1:] > small[:, :-1]).flatten()

print("=" * 70, flush=True)
print("RIGOROUS NEAR-DUPLICATE ANALYSIS ACROSS SPECIMEN GROUPS (CORRECTED)", flush=True)
print("=" * 70, flush=True)

check_disk("start")

# 1. Collect all canonical 0-degree photos
records = []
for cls in ["aegypti", "albopictus"]:
    cls_root = os.path.join(ARCHIVE, cls, cls)
    for folder in os.listdir(cls_root):
        if re.search(r'_0(?:_testing)?$', folder):
            fpath_dir = os.path.join(cls_root, folder)
            for fname in os.listdir(fpath_dir):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                m = re.search(r'(?:aegypti|albopictus)(\d+)([a-zA-Z]*)', fname)
                if not m: continue
                gnum = int(m.group(1))
                letter = m.group(2).lower()
                records.append({
                    "cls": cls,
                    "group_num": gnum,
                    "letter": letter,
                    "path": os.path.join(fpath_dir, fname),
                    "fname": fname
                })

print(f"Total canonical (rot 0) photos collected: {len(records)}", flush=True)

# 2. Preprocess images: center square 64x64 grayscale
img_vectors = []    # normalized vectors for ZNCC
img_grays   = []    # raw 64x64 gray for SSIM
img_dhashes = []    # 64-bit boolean array for dHash
meta_list   = []    # index -> (cls, group_num, letter, rotation, path)

for idx, r in enumerate(records):
    raw_bgr = cv2.imdecode(np.fromfile(r["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw_bgr is None: continue
    raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    sq_gray = center_square(raw_gray, sz=64)

    for rot_k in [0, 1, 2, 3]:
        rot_gray = np.rot90(sq_gray, rot_k)
        vec = rot_gray.astype(np.float32).flatten()
        vec_norm = vec - vec.mean()
        std = np.linalg.norm(vec_norm)
        if std > 0: vec_norm /= std

        img_vectors.append(vec_norm)
        img_grays.append(rot_gray.astype(np.float32) / 255.0)
        img_dhashes.append(dhash_bits(rot_gray))
        meta_list.append({
            "rec_idx": idx,
            "cls": r["cls"],
            "group_num": r["group_num"],
            "letter": r["letter"],
            "rot_deg": rot_k * 90,
            "path": r["path"],
            "fname": r["fname"]
        })

    if (idx + 1) % 200 == 0 or (idx + 1) == len(records):
        print(f"  Processed {idx + 1}/{len(records)} photos into rotated vectors ...", flush=True)

img_vectors = np.array(img_vectors, dtype=np.float32)
print(f"Total vector pool: {img_vectors.shape[0]} vectors of dim {img_vectors.shape[1]}", flush=True)

# 3. Compute all-pairs ZNCC per class
# For between-group: track max ZNCC per pair of groups
# For within-group: track max ZNCC per pair of distinct photos of same group
group_pair_max = {}   # (cls, g1, g2) -> (max_zncc, pi_global, ti_global)
within_photo_max = {} # (rec1, rec2) -> (max_zncc, pi_global, ti_global)

for cls in ["aegypti", "albopictus"]:
    print(f"\nProcessing class: {cls} ...", flush=True)
    probe_indices = [i for i, m in enumerate(meta_list) if m["cls"] == cls and m["rot_deg"] == 0]
    target_indices = [i for i, m in enumerate(meta_list) if m["cls"] == cls]

    X_probe = img_vectors[probe_indices]
    X_target = img_vectors[target_indices]

    print(f"  Multiplying ({len(probe_indices)}, 4096) x (4096, {len(target_indices)}) ...", flush=True)
    Corr = np.dot(X_probe, X_target.T) # shape: (len(probe), len(target)), values in [-1, 1]

    for pi_local, pi_global in enumerate(probe_indices):
        m_probe = meta_list[pi_global]
        g_probe = m_probe["group_num"]
        rec_probe = m_probe["rec_idx"]

        for ti_local, ti_global in enumerate(target_indices):
            m_target = meta_list[ti_global]
            g_target = m_target["group_num"]
            rec_target = m_target["rec_idx"]

            # Skip comparing a photo to itself
            if rec_probe == rec_target:
                continue

            zncc_val = float(Corr[pi_local, ti_local])

            # Within-group (same group, different letters)
            if g_probe == g_target:
                pair_rec = (min(rec_probe, rec_target), max(rec_probe, rec_target))
                if pair_rec not in within_photo_max or zncc_val > within_photo_max[pair_rec][0]:
                    within_photo_max[pair_rec] = (zncc_val, pi_global, ti_global)
            else:
                # Between-group (same class, different groups)
                pair_grp = (cls, min(g_probe, g_target), max(g_probe, g_target))
                if pair_grp not in group_pair_max or zncc_val > group_pair_max[pair_grp][0]:
                    group_pair_max[pair_grp] = (zncc_val, pi_global, ti_global)

within_vals = np.array([v[0] for v in within_photo_max.values()])
between_vals = np.array([v[0] for v in group_pair_max.values()])

print("\n" + "=" * 70, flush=True)
print("DISTRIBUTION COMPARISON: WITHIN-GROUP VS BETWEEN-GROUP (CORRECTED ZNCC)", flush=True)
print("=" * 70, flush=True)
print(f"Within-group photo pairs (different letters/bursts of same specimen): n = {len(within_vals)}", flush=True)
print(f"  Mean   : {within_vals.mean():.4f} ± {within_vals.std():.4f}", flush=True)
print(f"  Median : {np.median(within_vals):.4f}", flush=True)
print(f"  Min    : {within_vals.min():.4f}", flush=True)
print(f"  Max    : {within_vals.max():.4f}", flush=True)

print(f"\nBetween-group specimen pairs (closest photo pair across all letters & rotations): n = {len(between_vals)}", flush=True)
print(f"  Mean   : {between_vals.mean():.4f} ± {between_vals.std():.4f}", flush=True)
print(f"  Median : {np.median(between_vals):.4f}", flush=True)
print(f"  Min    : {between_vals.min():.4f}", flush=True)
print(f"  Max    : {between_vals.max():.4f}", flush=True)

# 4. Sort between-group pairs
all_between_items = []
for (cls_name, g1, g2), (zncc_v, pi, ti) in group_pair_max.items():
    g1_gray = img_grays[pi]
    g2_gray = img_grays[ti]
    d_ssim = ssim_metric(g1_gray, g2_gray)
    d_dhash = float(np.mean(img_dhashes[pi] != img_dhashes[ti]))
    all_between_items.append({
        "zncc": zncc_v,
        "ssim_dist": d_ssim,
        "dhash_dist": d_dhash,
        "m1": meta_list[pi],
        "m2": meta_list[ti]
    })

# 4a. Top 20 by HIGHEST ZNCC
top20_zncc = sorted(all_between_items, key=lambda x: x["zncc"], reverse=True)[:20]

print("\n" + "=" * 70, flush=True)
print("TOP 20 BETWEEN-GROUP PAIRS WITH HIGHEST ZNCC (MOST SIMILAR)", flush=True)
print("=" * 70, flush=True)
print(f"{'Rank':<5} {'Class':<12} {'Group A':<10} {'Group B':<10} {'ZNCC ↑':<10} {'SSIM Dist ↓':<12} {'dHash Dist ↓':<14} {'Rot B':<8} {'Filenames'}", flush=True)
print("-" * 105, flush=True)
for i, item in enumerate(top20_zncc):
    m1, m2 = item["m1"], item["m2"]
    ga = f"{m1['group_num']}{m1['letter']}"
    gb = f"{m2['group_num']}{m2['letter']}"
    rot = f"{m2['rot_deg']}°"
    fnames = f"{m1['fname']} vs {m2['fname']}"
    print(f"{i+1:<5} {m1['cls']:<12} {ga:<10} {gb:<10} {item['zncc']:<10.4f} {item['ssim_dist']:<12.4f} {item['dhash_dist']:<14.4f} {rot:<8} {fnames}", flush=True)

# 4b. Top 10 by LOWEST SSIM Distance
top10_ssim = sorted(all_between_items, key=lambda x: x["ssim_dist"])[:10]

# 4c. Top 10 by LOWEST dHash Distance
top10_dhash = sorted(all_between_items, key=lambda x: x["dhash_dist"])[:10]

# 5. Within-group pairs for reference
all_within_items = []
for (rec1, rec2), (zncc_v, pi, ti) in within_photo_max.items():
    g1_gray = img_grays[pi]
    g2_gray = img_grays[ti]
    d_ssim = ssim_metric(g1_gray, g2_gray)
    d_dhash = float(np.mean(img_dhashes[pi] != img_dhashes[ti]))
    all_within_items.append({
        "zncc": zncc_v,
        "ssim_dist": d_ssim,
        "dhash_dist": d_dhash,
        "m1": meta_list[pi],
        "m2": meta_list[ti]
    })

# Select 10 diverse within-group pairs (including user-mentioned pairs)
specific_pairs = []
for item in all_within_items:
    m1, m2 = item["m1"], item["m2"]
    names = {f"{m1['group_num']}{m1['letter']}", f"{m2['group_num']}{m2['letter']}"}
    if names == {"125a", "125b"} or names == {"128a", "128d"}:
        specific_pairs.append(item)

other_within = [item for item in sorted(all_within_items, key=lambda x: x["zncc"], reverse=True) if item not in specific_pairs]
ref10_within = specific_pairs + other_within[:(10 - len(specific_pairs))]

print("\n" + "=" * 70, flush=True)
print("10 WITHIN-GROUP REFERENCE PAIRS (SAME SPECIMEN, DIFFERENT BURST/SHOT)", flush=True)
print("=" * 70, flush=True)
print(f"{'Rank':<5} {'Class':<12} {'Group A':<10} {'Group B':<10} {'ZNCC ↑':<10} {'SSIM Dist ↓':<12} {'dHash Dist ↓':<14} {'Rot B':<8} {'Filenames'}", flush=True)
print("-" * 105, flush=True)
for i, item in enumerate(ref10_within):
    m1, m2 = item["m1"], item["m2"]
    ga = f"{m1['group_num']}{m1['letter']}"
    gb = f"{m2['group_num']}{m2['letter']}"
    rot = f"{m2['rot_deg']}°"
    fnames = f"{m1['fname']} vs {m2['fname']}"
    print(f"{i+1:<5} {m1['cls']:<12} {ga:<10} {gb:<10} {item['zncc']:<10.4f} {item['ssim_dist']:<12.4f} {item['dhash_dist']:<14.4f} {rot:<8} {fnames}", flush=True)

# 6. Helper to generate contact sheet PNGs (5 pairs per image, readable size)
def save_contact_sheet(pairs, filename, title_prefix="Pair"):
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 4.2 * n_pairs), dpi=150)
    if n_pairs == 1: axes = np.array([axes])

    for i, item in enumerate(pairs):
        m1, m2 = item["m1"], item["m2"]
        raw1 = cv2.imdecode(np.fromfile(m1["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        raw2 = cv2.imdecode(np.fromfile(m2["path"], dtype=np.uint8), cv2.IMREAD_COLOR)

        c1 = cv2.cvtColor(center_square(raw1, sz=256), cv2.COLOR_BGR2RGB)
        c2 = cv2.cvtColor(center_square(raw2, sz=256), cv2.COLOR_BGR2RGB)
        c2_rot = np.rot90(c2, m2["rot_deg"] // 90)

        axes[i, 0].imshow(c1)
        axes[i, 0].set_title(f"{title_prefix} {i+1} [Left: Group {m1['group_num']}{m1['letter']}]\n{m1['fname']} (rot 0°)", fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(c2_rot)
        axes[i, 1].set_title(f"[Right: Group {m2['group_num']}{m2['letter']} (rot {m2['rot_deg']}°)]\nZNCC={item['zncc']:.4f}  |  SSIM Dist={item['ssim_dist']:.4f}  |  dHash Dist={item['dhash_dist']:.3f}", fontsize=11, fontweight='bold')
        axes[i, 1].axis('off')

    plt.tight_layout()
    out_eval = os.path.join(EVAL_DIR, filename)
    plt.savefig(out_eval, bbox_inches='tight')
    out_v2 = os.path.join(V2_DIR, filename)
    plt.savefig(out_v2, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename} ({os.path.getsize(out_eval)/1e3:.1f} KB)", flush=True)

print("\n" + "=" * 70, flush=True)
print("GENERATING READABLE CONTACT SHEETS (5 PAIRS PER IMAGE)", flush=True)
print("=" * 70, flush=True)

# 20 between-group pairs with highest ZNCC (4 files of 5 pairs)
save_contact_sheet(top20_zncc[0:5],   "closest_between_group_pairs_FIXED_1_to_5.png",   title_prefix="ZNCC Rank")
save_contact_sheet(top20_zncc[5:10],  "closest_between_group_pairs_FIXED_6_to_10.png",  title_prefix="ZNCC Rank")
save_contact_sheet(top20_zncc[10:15], "closest_between_group_pairs_FIXED_11_to_15.png", title_prefix="ZNCC Rank")
save_contact_sheet(top20_zncc[15:20], "closest_between_group_pairs_FIXED_16_to_20.png", title_prefix="ZNCC Rank")

# 10 between-group pairs with lowest SSIM distance (2 files of 5 pairs)
save_contact_sheet(top10_ssim[0:5],   "closest_between_group_pairs_FIXED_ssim_1_to_5.png",  title_prefix="SSIM Rank")
save_contact_sheet(top10_ssim[5:10],  "closest_between_group_pairs_FIXED_ssim_6_to_10.png", title_prefix="SSIM Rank")

# 10 between-group pairs with lowest dHash distance (2 files of 5 pairs)
save_contact_sheet(top10_dhash[0:5],  "closest_between_group_pairs_FIXED_dhash_1_to_5.png", title_prefix="dHash Rank")
save_contact_sheet(top10_dhash[5:10], "closest_between_group_pairs_FIXED_dhash_6_to_10.png",title_prefix="dHash Rank")

# 10 within-group reference pairs (2 files of 5 pairs)
save_contact_sheet(ref10_within[0:5],  "within_group_reference_pairs_1_to_5.png",  title_prefix="Within Ref")
save_contact_sheet(ref10_within[5:10], "within_group_reference_pairs_6_to_10.png", title_prefix="Within Ref")

# Copy this script into CNN_Mosquito_Classifier_v2 as requested in step 5
src_script = os.path.join(EVAL_DIR, "step1_duplicate_check_rigorous.py")
dst_script = os.path.join(V2_DIR, "step1_duplicate_check_rigorous.py")
import shutil
shutil.copy2(src_script, dst_script)
print(f"Copied script to: {dst_script}", flush=True)

check_disk("end")
print("\nANALYSIS & GENERATION COMPLETE.", flush=True)
