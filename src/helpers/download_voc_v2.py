# download_voc_minimal.py
# Downloads VOC2012 segmentation subset (JPEG + mask + annotation) for trainval IDs.

import os, urllib.request, tarfile
from pathlib import Path

BASE = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012"
VOC_TAR_URL = f"{BASE}/VOCtrainval_11-May-2012.tar"
SEG_LIST_MEMBER = "VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt"


def _extract_seg_list_from_trainval_tar(dest="data/voc"):
    """Extract only segmentation trainval list from VOC2012 trainval tar."""
    print("Extracting segmentation ID list from VOC2012 trainval tar...")
    with urllib.request.urlopen(VOC_TAR_URL) as response:
        with tarfile.open(fileobj=response, mode="r|") as tar:
            for member in tar:
                if member.name == SEG_LIST_MEMBER:
                    tar.extract(member, dest)
                    print("Segmentation trainval list extracted.")
                    return
    raise RuntimeError("Could not find trainval list inside VOC trainval tar.")


def get_seg_image_ids(voc_root="data/voc/VOCdevkit/VOC2012"):
    seg_list = Path(voc_root) / "ImageSets/Segmentation/trainval.txt"
    if not seg_list.exists():
        _extract_seg_list_from_trainval_tar(dest=str(Path(voc_root).parents[1]))
    return [line.strip() for line in seg_list.read_text().splitlines() if line.strip()]


def download_seg_subset(dest="data/voc", only_masks=False):
    """
    Download only segmentation split assets from VOC2012 trainval tar:
    - JPEGImages/*.jpg
    - SegmentationClass/*.png
    - Annotations/*.xml
    """
    voc_root = f"{dest}/VOCdevkit/VOC2012"
    ids = get_seg_image_ids(voc_root)
    needed = set()
    for i in ids:
        if not only_masks:
            needed.add(f"VOCdevkit/VOC2012/JPEGImages/{i}.jpg")
            needed.add(f"VOCdevkit/VOC2012/Annotations/{i}.xml")
        needed.add(f"VOCdevkit/VOC2012/SegmentationClass/{i}.png")
    needed.add(SEG_LIST_MEMBER)

    print(f"Streaming VOC tar and extracting {len(ids)} segmentation samples...")
    print("(This can take a while because source tar is large.)")

    with urllib.request.urlopen(VOC_TAR_URL) as response:
        with tarfile.open(fileobj=response, mode="r|") as tar:
            for member in tar:
                if member.name in needed:
                    tar.extract(member, dest)
                    needed.discard(member.name)
                    if len(needed) % 300 == 0:
                        print(f"  {len(needed)} remaining...")
                if not needed:
                    break

    print("Done.")
    print(f"  JPEGs: {voc_root}/JPEGImages/")
    print(f"  Masks: {voc_root}/SegmentationClass/")
    print(f"  XMLs : {voc_root}/Annotations/")

if __name__ == "__main__":
    os.makedirs("data/voc", exist_ok=True)
    download_seg_subset(only_masks=True)
    print("\nFinal structure:")
    print("  data/voc/VOCdevkit/VOC2012/JPEGImages/")
    print("  data/voc/VOCdevkit/VOC2012/Annotations/")
    print("  data/voc/VOCdevkit/VOC2012/SegmentationClass/")