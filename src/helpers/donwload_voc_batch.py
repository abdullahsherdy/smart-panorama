# download_voc_minimal.py
# Downloads only the ~600 VOC segmentation images + their annotations + masks
# Total size: ~120MB instead of 2GB

import os, urllib.request, tarfile, shutil
from pathlib import Path

BASE = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012"

# Step 1 — download devkit only (annotations + image lists, no JPEGs) ~2MB
def download_devkit(dest="data/voc"):
    os.makedirs(dest, exist_ok=True)
    url = f"{BASE}/VOCdevkit_18-May-2011.tar"
    print("Downloading devkit (~2MB)...")
    urllib.request.urlretrieve(url, f"{dest}/devkit.tar")
    with tarfile.open(f"{dest}/devkit.tar") as t:
        t.extractall(dest)
    os.remove(f"{dest}/devkit.tar")
    print("Devkit extracted.")

# Step 2 — download only the trainval segmentation image list
def get_seg_image_ids(voc_root="data/voc/VOCdevkit/VOC2012"):
    seg_list = Path(voc_root) / "ImageSets/Segmentation/trainval.txt"
    return [line.strip() for line in seg_list.read_text().splitlines() if line.strip()]

# Step 3 — download only those specific JPEGs from the devkit tar
# (VOC2012 trainval.tar contains all images; we extract selectively)
def download_seg_images_only(dest="data/voc"):
    url = f"{BASE}/VOCtrainval_11-May-2012.tar"
    voc_root = f"{dest}/VOCdevkit/VOC2012"
    ids = get_seg_image_ids(voc_root)
    needed = {f"VOCdevkit/VOC2012/JPEGImages/{i}.jpg" for i in ids}

    print(f"Streaming tar to extract {len(needed)} segmentation images only...")
    print("(This downloads ~120MB of a 2GB file — will take a few minutes)")

    with urllib.request.urlopen(url) as response:
        with tarfile.open(fileobj=response, mode="r|") as tar:
            for member in tar:
                if member.name in needed:
                    tar.extract(member, dest)
                    needed.discard(member.name)
                    if len(needed) % 100 == 0:
                        print(f"  {len(needed)} remaining...")
                if not needed:
                    break

    print(f"Done. Images saved to {voc_root}/JPEGImages/")

if __name__ == "__main__":
    download_devkit()
    download_seg_images_only()
    print("\nFinal structure:")
    print("  data/voc/VOCdevkit/VOC2012/JPEGImages/     (~600 files, ~120MB)")
    print("  data/voc/VOCdevkit/VOC2012/Annotations/    (XML bboxes, already in devkit)")
    print("  data/voc/VOCdevkit/VOC2012/SegmentationClass/  (PNG masks, already in devkit)")