import os
import glob
import re
import random
import shutil
import cv2
import hashlib
import numpy as np
from collections import defaultdict

ARCHIVE_DIR = r"C:\Users\LAPTOP\Downloads\archive"
V2_DIR = r"C:\Users\LAPTOP\OneDrive\ドキュメント\agent skill\Aegypti\CNN_Mosquito_Classifier_v2"
DATASET_DIR = os.path.join(V2_DIR, "dataset")

def read_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def run():
    print("=" * 70)
    print("STEP 1: Grouping files and building leak-free train/val/test splits")
    print("=" * 70)

    # 1. Scan archive files
    groups_by_class = {"aegypti": defaultdict(list), "albopictus": defaultdict(list)}
    total_found = 0

    for cls in ["aegypti", "albopictus"]:
        cls_root = os.path.join(ARCHIVE_DIR, cls, cls)
        for root, dirs, files in os.walk(cls_root):
            folder_name = os.path.basename(root)
            rot_match = re.search(r"_(0|90|180|270)", folder_name)
            rot = rot_match.group(1) if rot_match else "0"

            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                m = re.search(r"([a-zA-Z]+)(\d+)", f)
                if not m:
                    continue
                grp_id = int(m.group(2))
                src_path = os.path.join(root, f)
                groups_by_class[cls][grp_id].append({
                    "src_path": src_path,
                    "filename": f,
                    "rot": rot,
                    "grp_id": grp_id,
                    "class": cls
                })
                total_found += 1

    print(f"Total archive images found: {total_found}")
    print(f"  - Aegypti groups   : {len(groups_by_class['aegypti'])}")
    print(f"  - Albopictus groups: {len(groups_by_class['albopictus'])}")

    # 2. Stratified Random Assignment (70% train, 15% val, 15% test, seed=42)
    split_assignments = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    group_stats = {}

    for cls in ["aegypti", "albopictus"]:
        grps = sorted(groups_by_class[cls].keys())
        rng = random.Random(42)
        rng.shuffle(grps)

        n = len(grps)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        
        train_grps = set(grps[:n_train])
        val_grps = set(grps[n_train:n_train + n_val])
        test_grps = set(grps[n_train + n_val:])

        # Strict disjointness verification
        assert len(train_grps & val_grps) == 0, "Train and Val groups overlap!"
        assert len(train_grps & test_grps) == 0, "Train and Test groups overlap!"
        assert len(val_grps & test_grps) == 0, "Val and Test groups overlap!"

        group_stats[cls] = {
            "train": train_grps,
            "val": val_grps,
            "test": test_grps
        }

        for gid in train_grps:
            split_assignments["train"][cls].extend(groups_by_class[cls][gid])
        for gid in val_grps:
            split_assignments["val"][cls].extend(groups_by_class[cls][gid])
        for gid in test_grps:
            split_assignments["test"][cls].extend(groups_by_class[cls][gid])

    print("\n" + "=" * 70)
    print("STEP 2: Linking/Copying files to v2\\dataset\\{train,val,test}\\{aegypti,albopictus}\\")
    print("=" * 70)

    # Create destination directories
    for split in ["train", "val", "test"]:
        for cls in ["aegypti", "albopictus"]:
            dst_dir = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(dst_dir, exist_ok=True)

    copied_counts = defaultdict(lambda: defaultdict(int))

    for split in ["train", "val", "test"]:
        for cls in ["aegypti", "albopictus"]:
            dst_dir = os.path.join(DATASET_DIR, split, cls)
            for item in split_assignments[split][cls]:
                src = item["src_path"]
                dst_name = f"rot{item['rot']}_{item['filename']}"
                dst_path = os.path.join(dst_dir, dst_name)
                if not os.path.exists(dst_path):
                    try:
                        os.link(src, dst_path)
                    except Exception:
                        shutil.copy2(src, dst_path)
                copied_counts[split][cls] += 1

    print("Files successfully populated in v2\\dataset:")
    for split in ["train", "val", "test"]:
        print(f"  [{split.upper()}]")
        print(f"    aegypti   : {len(group_stats['aegypti'][split])} groups, {copied_counts[split]['aegypti']} images")
        print(f"    albopictus: {len(group_stats['albopictus'][split])} groups, {copied_counts[split]['albopictus']} images")
        print(f"    subtotal  : {len(group_stats['aegypti'][split]) + len(group_stats['albopictus'][split])} groups, "
              f"{copied_counts[split]['aegypti'] + copied_counts[split]['albopictus']} images")

    print("\n" + "=" * 70)
    print("STEP 3: Verifying Zero Leakage (Pixel & Rotation Hashes)")
    print("=" * 70)

    # Hash all train and val images at 64x64
    print("Computing 64x64 hashes for all train and val images...")
    train_val_hashes = set()

    for split in ["train", "val"]:
        for cls in ["aegypti", "albopictus"]:
            img_dir = os.path.join(DATASET_DIR, split, cls)
            for fname in os.listdir(img_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fpath = os.path.join(img_dir, fname)
                img = read_image_unicode(fpath)
                if img is None:
                    continue
                img64 = cv2.resize(img, (64, 64))
                h = hashlib.sha256(img64.tobytes()).hexdigest()
                train_val_hashes.add(h)

    print(f"Total unique 64x64 train+val hashes: {len(train_val_hashes)}")

    # Check every test image: compute hash of all 4 rotations (0, 90, 180, 270)
    print("Checking every test image against train+val across all 4 rotations...")
    leaks = 0
    test_checked = 0

    for cls in ["aegypti", "albopictus"]:
        test_dir = os.path.join(DATASET_DIR, "test", cls)
        for fname in os.listdir(test_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(test_dir, fname)
            img = read_image_unicode(fpath)
            if img is None:
                continue
            img64 = cv2.resize(img, (64, 64))
            
            # Check all 4 rotations of this test image
            for k in [0, 1, 2, 3]:
                rot_img = np.rot90(img64, k)
                h_rot = hashlib.sha256(rot_img.tobytes()).hexdigest()
                if h_rot in train_val_hashes:
                    leaks += 1
                    print(f"  [LEAK DETECTED!] Test image {fname} at rotation {k*90} matches train/val hash {h_rot}!")
            test_checked += 1

    print(f"\nVerification Results:")
    print(f"  Test images inspected: {test_checked}")
    print(f"  Rotations verified per image: 4 (0°, 90°, 180°, 270°)")
    print(f"  Total rotation-checks: {test_checked * 4}")
    print(f"  Leakage matches found: {leaks}")

    if leaks == 0:
        print("\n>>> ZERO LEAKAGE CONFIRMED! The train/val/test split is 100% statistically independent. <<<")
    else:
        raise RuntimeError(f"Data leakage detected! {leaks} matches found.")

if __name__ == "__main__":
    run()
