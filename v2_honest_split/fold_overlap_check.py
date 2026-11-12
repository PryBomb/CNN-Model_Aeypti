"""
fold_overlap_check.py
Fast, exact verification of group-number and 64x64 image-hash isolation
across all 5 cross-validation folds.
"""
import os, sys, re, hashlib, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8')

ARCHIVE  = r"C:\Users\LAPTOP\Downloads\archive"
CACHE_PATH = r"C:\Users\LAPTOP\mosquito_eval\eval_cache_stretch.npz"
N_FOLDS  = 5
SEED     = 42

def collect_archive():
    records = []
    for cls, lbl in [("aegypti", 0), ("albopictus", 1)]:
        cls_root = os.path.join(ARCHIVE, cls, cls)
        for folder in sorted(os.listdir(cls_root)):
            fp = os.path.join(cls_root, folder)
            if not os.path.isdir(fp): continue
            rot_m = re.search(r'_(\d+)(?:_testing)?$', folder)
            if not rot_m: continue
            rotation = int(rot_m.group(1))
            for fname in os.listdir(fp):
                if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue
                m = re.search(r'(?:aegypti|albopictus)(\d+)([a-zA-Z]*)', fname)
                if not m: continue
                records.append(dict(
                    cls=cls, label=lbl,
                    group_num=int(m.group(1)),
                    letter=m.group(2).lower(),
                    rotation=rotation,
                    path=os.path.join(fp, fname)
                ))
    return records

def main():
    print("=== VERIFYING GROUPED 5-FOLD CROSS-VALIDATION ===", flush=True)
    records = collect_archive()
    n_records = len(records)
    print(f"Total archive image records: {n_records}", flush=True)

    # 1. Rebuild group mapping exactly as in step2_4_kfold_and_checks.py
    group_keys = sorted(set((r["cls"], r["group_num"]) for r in records))
    group_to_idx = {gk: i for i, gk in enumerate(group_keys)}
    group_labels = np.array([0 if gk[0]=="aegypti" else 1 for gk in group_keys])
    print(f"Total unique photo groups: {len(group_keys)} (aegypti={sum(group_labels==0)}, albopictus={sum(group_labels==1)})", flush=True)

    group_to_records = defaultdict(list)
    for i, r in enumerate(records):
        gk = (r["cls"], r["group_num"])
        group_to_records[group_to_idx[gk]].append(i)

    # 2. Load 64x64 cache to hash every image
    print(f"Loading 64x64 images from {CACHE_PATH} ...", flush=True)
    cache = np.load(CACHE_PATH)
    X = cache["X"]  # (4803, 64, 64, 3)
    assert len(X) == n_records, f"Cache size {len(X)} != records {n_records}"

    print("Computing SHA-256 hashes for all 4,803 images at 64x64 (all rotations) ...", flush=True)
    img_hashes = []
    for i in range(len(X)):
        img_bytes = (X[i] * 255.0).astype(np.uint8).tobytes()
        img_hashes.append(hashlib.sha256(img_bytes).hexdigest())

    # 3. Replicate 5-fold CV split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_assignments = list(skf.split(group_keys, group_labels))

    def grp_indices_to_img_indices(grp_idx_arr):
        idxs = []
        for gi in grp_idx_arr:
            idxs.extend(group_to_records[gi])
        return np.array(idxs, dtype=int)

    all_pass = True
    print("\n--- PER-FOLD ISOLATION AUDIT ---", flush=True)

    for fold_idx, (train_grp_idx, test_grp_idx) in enumerate(fold_assignments):
        fold_num = fold_idx + 1

        # Inner validation split (cut from train_grp_idx ONLY)
        inner_skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=SEED + fold_idx)
        inner_train_local, inner_val_local = next(
            inner_skf.split(train_grp_idx, group_labels[train_grp_idx])
        )
        inner_train_grp = train_grp_idx[inner_train_local]
        inner_val_grp   = train_grp_idx[inner_val_local]

        # Check A: Group number overlap between train and test
        train_grp_set = set(train_grp_idx.tolist())
        test_grp_set  = set(test_grp_idx.tolist())
        val_grp_set   = set(inner_val_grp.tolist())

        grp_overlap = train_grp_set.intersection(test_grp_set)
        val_test_overlap = val_grp_set.intersection(test_grp_set)

        # Check B: Image indices and 64x64 image hashes
        tr_img_idx  = grp_indices_to_img_indices(inner_train_grp)
        val_img_idx = grp_indices_to_img_indices(inner_val_grp)
        te_img_idx  = grp_indices_to_img_indices(test_grp_idx)

        # Full training set images = train + val
        full_train_img_idx = np.concatenate([tr_img_idx, val_img_idx])
        img_idx_overlap = set(full_train_img_idx).intersection(set(te_img_idx))

        train_hashes = set(img_hashes[i] for i in full_train_img_idx)
        test_hashes  = set(img_hashes[i] for i in te_img_idx)
        hash_overlap = train_hashes.intersection(test_hashes)

        pass_fold = (len(grp_overlap) == 0 and
                     len(val_test_overlap) == 0 and
                     len(img_idx_overlap) == 0 and
                     len(hash_overlap) == 0)

        if not pass_fold:
            all_pass = False

        status_str = "PASS" if pass_fold else "FAIL"
        print(f"Fold {fold_num}: [{status_str}]", flush=True)
        print(f"  Train groups: {len(inner_train_grp)} | Val groups: {len(inner_val_grp)} | Test groups: {len(test_grp_idx)}", flush=True)
        print(f"  Train images: {len(tr_img_idx)} | Val images: {len(val_img_idx)} | Test images: {len(te_img_idx)}", flush=True)
        print(f"  Photo-group overlap (train AND test): {len(grp_overlap)}", flush=True)
        print(f"  Validation-set overlap with test    : {len(val_test_overlap)}", flush=True)
        print(f"  Image-index overlap (train AND test): {len(img_idx_overlap)}", flush=True)
        print(f"  64x64 Image-hash overlap (train AND test): {len(hash_overlap)}", flush=True)
        print("", flush=True)

    if all_pass:
        print("RESULT: ALL 5 FOLDS PERFECTLY ISOLATED. ZERO OVERLAPS.", flush=True)
    else:
        print("RESULT: FAILURE DETECTED.", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
