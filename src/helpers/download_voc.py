# download_voc_v2.py
# Minimal VOC2012 downloader (teammate flow): stream official trainval tar once per pass,
# extract only Segmentation trainval list + up to N image triples:
# JPEGImages, Annotations, SegmentationClass.
#
# Output layout (always under repo `data/voc/`, regardless of cwd):
#   data/voc/VOCdevkit/VOC2012/{JPEGImages,Annotations,SegmentationClass,ImageSets/Segmentation}
#
# Stages 5–6 (`segment.py`, `classify.py`) expect voc_root =
# `<repo>/data/voc/VOCdevkit/VOC2012`.

from __future__ import annotations

import argparse
import errno
import http.client
import json
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012"
VOC_TAR_URL = f"{BASE}/VOCtrainval_11-May-2012.tar"
SEG_LIST_MEMBER = "VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt"

# Project cap: VOC snippet size for classify + segment pipelines.
PROJECT_MAX_SNIPPET_IMAGES = 100
MAX_TAR_PASSES = 3

# urllib ``timeout`` is one number for the underlying socket on this platform/build.
# A modest value (e.g. 600) applies to blocking reads — tarfile can idle long while skipping
# a ~2 GB stream → spurious TimeoutError. Use a very large ceiling (days) for this one-shot
# Oxford download so slow links can finish; TCP still fails fast on true disconnects.
TAR_STREAM_SOCKET_TIMEOUT_SEC = 7 * 24 * 3600.0

STREAM_OP_MAX_RETRIES = 10
STREAM_OP_BACKOFF_SEC = 4.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _norm_member(name: str) -> str:
    return name.replace("\\", "/")


def _safe_extract(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: str) -> None:
    if sys.version_info >= (3, 12):
        tar.extract(member, dest, filter="data")  # type: ignore[call-arg]
    else:
        tar.extract(member, dest)


def _trainval_tar_request() -> urllib.request.Request:
    return urllib.request.Request(
        VOC_TAR_URL,
        headers={
            "User-Agent": "smart-panorama-voc-v2/2.1",
            "Accept-Encoding": "identity",
        },
    )


def _urlopen_trainval() -> object:
    req = _trainval_tar_request()
    return urllib.request.urlopen(req, timeout=TAR_STREAM_SOCKET_TIMEOUT_SEC)


def _is_transient_stream_error(e: BaseException) -> bool:
    if isinstance(e, (TimeoutError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return True
    if isinstance(e, tarfile.ReadError):
        return True
    if isinstance(e, ssl.SSLError):
        return True
    if isinstance(e, http.client.IncompleteRead):
        return True
    rd = getattr(http.client, "RemoteDisconnected", None)
    if rd is not None and isinstance(e, rd):
        return True
    if isinstance(e, urllib.error.URLError):
        return True
    if isinstance(e, urllib.error.HTTPError):
        return e.code in (408, 425, 429) or e.code >= 500
    if isinstance(e, OSError):
        w = getattr(e, "winerror", None)
        if w in (10054, 10053, 10060, 10061):
            return True
        en = getattr(e, "errno", None)
        return en in (errno.ECONNRESET, errno.ETIMEDOUT, errno.EPIPE, errno.ECONNABORTED)
    return False


def resolve_voc_bundle_root(dest: str | Path | None) -> Path:
    """Directory that contains (or will contain) ``VOCdevkit/`` — default ``<repo>/data/voc``."""
    if dest is None:
        return repo_root() / "data" / "voc"
    p = Path(dest)
    return p.resolve() if p.is_absolute() else (repo_root() / p)


def extract_trainval_list_if_missing(bundle: Path) -> None:
    target = bundle / SEG_LIST_MEMBER
    if target.is_file():
        return
    print("Extract Segmentation/trainval.txt (streaming VOC trainval tar)...")
    bundle.mkdir(parents=True, exist_ok=True)
    last_err: BaseException | None = None
    for attempt in range(1, STREAM_OP_MAX_RETRIES + 1):
        try:
            with _urlopen_trainval() as response:
                with tarfile.open(fileobj=response, mode="r|") as tar:
                    for member in tar:
                        if _norm_member(member.name) == SEG_LIST_MEMBER:
                            _safe_extract(tar, member, str(bundle))
                            print(f"  wrote {SEG_LIST_MEMBER}")
                            return
            raise RuntimeError("Reached end of tar without finding trainval.txt.")
        except Exception as e:
            last_err = e
            if isinstance(e, RuntimeError) and "Reached end of tar" in str(e):
                raise
            if not _is_transient_stream_error(e):
                raise
            wait = min(STREAM_OP_BACKOFF_SEC * (2 ** (attempt - 1)), 120.0)
            print(
                f"  trainval list stream failed ({e.__class__.__name__}); "
                f"retry {attempt}/{STREAM_OP_MAX_RETRIES} in {wait:.0f}s …"
            )
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def read_seg_trainval_ids(voc2012: Path, max_images: int) -> list[str]:
    seg_list = voc2012 / "ImageSets" / "Segmentation" / "trainval.txt"
    if not seg_list.is_file():
        raise FileNotFoundError(f"Missing {seg_list}")
    ids = [
        ln.strip()
        for ln in seg_list.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    effective = PROJECT_MAX_SNIPPET_IMAGES
    if max_images > 0:
        effective = min(max_images, PROJECT_MAX_SNIPPET_IMAGES)
    elif max_images < 0:
        raise ValueError("--max-images must be ≥ 0")
    return ids[:effective]


def download_seg_subset(bundle: Path, max_images: int) -> Path:
    
    """
    Download JPEG + VOC XML annotation + SegmentationClass mask per ID (subset of Segmentation/trainval).

    Parameters
    ----------
    bundle
        e.g. ``<repo>/data/voc`` — **not** the inner VOC2012 folder.
    max_images
        At least 1, at most :data:`PROJECT_MAX_SNIPPET_IMAGES` when positive.
        ``0`` is treated like "use full project cap" (same as ``100``).
    """
    
    extract_trainval_list_if_missing(bundle)
    voc2012 = bundle / "VOCdevkit" / "VOC2012"
    ids = read_seg_trainval_ids(voc2012, max_images=max_images)

    needed: set[str] = set()
    for i in ids:
        needed.add(f"VOCdevkit/VOC2012/JPEGImages/{i}.jpg")
        needed.add(f"VOCdevkit/VOC2012/Annotations/{i}.xml")
        needed.add(f"VOCdevkit/VOC2012/SegmentationClass/{i}.png")
    needed.add(SEG_LIST_MEMBER)

    bundle.mkdir(parents=True, exist_ok=True)
    (voc2012 / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (voc2012 / "Annotations").mkdir(parents=True, exist_ok=True)
    (voc2012 / "SegmentationClass").mkdir(parents=True, exist_ok=True)

    total_need = len(needed)
    print(
        f"Streaming VOC tar — extracting {len(ids)} image(s) × 3 + list "
        f"({total_need} tar member(s)); up to {MAX_TAR_PASSES} pass(es)..."
    )

    for pass_no in range(1, MAX_TAR_PASSES + 1):
        if not needed:
            break
        stream_ok = False
        last_err: BaseException | None = None
        for attempt in range(1, STREAM_OP_MAX_RETRIES + 1):
            try:
                with _urlopen_trainval() as response:
                    with tarfile.open(fileobj=response, mode="r|") as tar:
                        last_report = len(needed)
                        for member in tar:
                            name = _norm_member(member.name)
                            if name not in needed:
                                continue
                            try:
                                _safe_extract(tar, member, str(bundle))
                            except Exception:
                                rel = bundle / name
                                if rel.is_file():
                                    rel.unlink()
                                raise
                            needed.discard(name)
                            remain = len(needed)
                            if (
                                remain
                                and (remain != last_report)
                                and remain % 50 == 0
                            ):
                                print(f"  … {remain} tar member(s) left")
                                last_report = remain
                            if not needed:
                                break
                stream_ok = True
                break
            except Exception as e:
                last_err = e
                if not _is_transient_stream_error(e):
                    raise
                wait = min(STREAM_OP_BACKOFF_SEC * (2 ** (attempt - 1)), 120.0)
                print(
                    f"  pass {pass_no} stream retry {attempt}/{STREAM_OP_MAX_RETRIES} "
                    f"({e.__class__.__name__}), wait {wait:.0f}s …"
                )
                time.sleep(wait)
        if not stream_ok:
            assert last_err is not None
            raise last_err
        print(
            f"  pass {pass_no}/{MAX_TAR_PASSES}: {len(needed)} tar member(s) remaining"
        )

    if needed:
        ex = sorted(needed)[:8]
        raise RuntimeError(f"Still missing {len(needed)} path(s). Examples: {ex}")

    manifest = bundle / "_download_manifest_v2.json"
    manifest.write_text(
        json.dumps(
            {"n_ids": len(ids), "image_ids": ids, "voc2012_relpath": "VOCdevkit/VOC2012"},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Done. ")
    print(f"  JPEGs -> {voc2012 / 'JPEGImages'}")
    print(f"  Masks -> {voc2012 / 'SegmentationClass'}")
    print(f"  XMLs  -> {voc2012 / 'Annotations'}")
    print(f"  manifest -> {manifest}")
    return bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download ≤100 VOC2012 segmentation IDs (JPEG+XML+mask) into data/voc."
    )

    # dest 
    p.add_argument(
        "--dest",
        type=str,
        default=str((repo_root() / "data" / "voc").relative_to(repo_root())),
        help="Path relative to repo root; default data/voc (contains VOCdevkit/).",
    )

    # max-images number
    p.add_argument(
        "--max-images",
        type=int,
        default=PROJECT_MAX_SNIPPET_IMAGES,
        help=f"Subset size (cap {PROJECT_MAX_SNIPPET_IMAGES}). Use 0 for cap-only default.",
    )

    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    bundle = resolve_voc_bundle_root(a.dest)
    mi = a.max_images

    if mi <= 0:
        mi = PROJECT_MAX_SNIPPET_IMAGES
    else:
        mi = min(mi, PROJECT_MAX_SNIPPET_IMAGES)

    print(f"VOC bundle root (absolute): {bundle}")
    download_seg_subset(bundle, max_images=mi)