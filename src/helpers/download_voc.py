"""
download_voc.py
================

Faster, resumable VOC2012 downloader.

What it does
------------
1. Downloads the official VOC2012 trainval tar (~2 GB) from Oxford **once**, into
   ``data/voc/_cache/VOCtrainval_11-May-2012.tar``:

   - Uses HTTP ``Range`` requests to resume on disconnect (no 2 GB redo).
   - Optional multi-connection parallel download (``--connections``) for speed
     on slow / single-stream-throttled links.

2. Extracts the requested subset locally from the cached tar (random-access,
   near-instant) into a standard VOC layout that ``segment.py`` and
   ``classify.py`` consume:

       <dest>/
         JPEGImages/<id>.jpg
         Annotations/<id>.xml
         SegmentationClass/<id>.png   (only IDs in Segmentation list)
         ImageSets/Main/{train,val,trainval}.txt
         ImageSets/Segmentation/{train,val,trainval}.txt

Common usage
------------
Get everything (recommended for max classification accuracy)::

    python src/helpers/download_voc.py --max-images 0

Subset, say first 1000 IDs from the segmentation trainval list::

    python src/helpers/download_voc.py --max-images 1000 --ids-list seg

Use a tar you already have (skip download)::

    python src/helpers/download_voc.py --tar-path D:/path/to/VOCtrainval.tar
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set, Tuple

VOC_TAR_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
)
TAR_FILENAME = "VOCtrainval_11-May-2012.tar"
USER_AGENT = "smart-panorama-voc/3.0"

INNER_VOC_PREFIX = "VOCdevkit/VOC2012"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_under_repo(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (repo_root() / p).resolve()


def _request(url: str, headers: Optional[dict] = None) -> urllib.request.Request:
    h = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if headers:
        h.update(headers)
    return urllib.request.Request(url, headers=h)


def _head_content_length(url: str) -> Tuple[int, bool]:
    """Return (content_length, supports_ranges) via HEAD."""
    req = _request(url)
    req.method = "HEAD"  # type: ignore[attr-defined]
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers.get("Content-Length", "0") or 0)
        accept = (r.headers.get("Accept-Ranges", "") or "").lower()
    return size, accept == "bytes"


def _format_mb(n: int) -> str:
    return f"{n / (1024 * 1024):8.1f} MB"


# ---------- Downloader ----------------------------------------------------- #


class _ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.start = time.time()
        self.lock = threading.Lock()
        self._last_print = 0.0

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n
            now = time.time()
            if now - self._last_print >= 1.0 or self.done >= self.total:
                self._last_print = now
                self._print()

    def _print(self) -> None:
        elapsed = max(1e-3, time.time() - self.start)
        speed = self.done / elapsed
        remaining = max(0, self.total - self.done)
        eta = remaining / speed if speed > 0 else 0
        pct = 100.0 * self.done / max(1, self.total)
        sys.stdout.write(
            f"\r  [{pct:5.1f}%] {_format_mb(self.done)}/{_format_mb(self.total)}"
            f"  {speed/1024/1024:5.2f} MB/s  ETA {eta:6.0f}s "
        )
        sys.stdout.flush()


def _download_range(
    url: str,
    start: int,
    end_inclusive: int,
    dest: Path,
    progress: _ProgressTracker,
    max_retries: int = 8,
) -> None:
    """Download ``[start, end_inclusive]`` into ``dest`` at offset ``start``.

    Resumes from whatever bytes are already on disk for this range.
    """
    have = 0
    if dest.exists():
        size = dest.stat().st_size
        if start <= size <= end_inclusive + 1:
            have = size - start
        elif size > end_inclusive + 1:
            have = end_inclusive + 1 - start
    if have > 0:
        progress.add(have)
    if start + have > end_inclusive:
        return

    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            req = _request(
                url,
                headers={"Range": f"bytes={start + have}-{end_inclusive}"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp, open(
                dest, "r+b"
            ) as f:
                f.seek(start + have)
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    progress.add(len(chunk))
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt == max_retries:
                raise
            wait = min(backoff, 60.0)
            sys.stdout.write(
                f"\n  range [{start}-{end_inclusive}] retry {attempt}/{max_retries} "
                f"({e.__class__.__name__}); wait {wait:.0f}s\n"
            )
            time.sleep(wait)
            backoff *= 2


def _ensure_sparse_file(path: Path, size: int) -> None:
    """Create or extend ``path`` to ``size`` bytes (truncate if larger)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "wb") as f:
            f.truncate(size)
        return
    cur = path.stat().st_size
    if cur < size:
        with open(path, "r+b") as f:
            f.truncate(size)
    elif cur > size:
        with open(path, "r+b") as f:
            f.truncate(size)


def download_tar(
    cache_path: Path, connections: int = 8, url: str = VOC_TAR_URL
) -> Path:
    """Download (or resume) the VOC trainval tar to ``cache_path``."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Probing tar size: {url}")
    total, supports_ranges = _head_content_length(url)
    if total <= 0:
        raise RuntimeError("Server did not return a usable Content-Length.")
    print(f"Tar size: {_format_mb(total)}  (Range support: {supports_ranges})")

    if cache_path.exists() and cache_path.stat().st_size == total:
        print(f"Tar already cached: {cache_path}")
        return cache_path

    if not supports_ranges:
        connections = 1
        print("Server does not advertise Range support; falling back to 1 stream.")

    _ensure_sparse_file(cache_path, total)

    n = max(1, int(connections))
    chunk = total // n
    ranges: List[Tuple[int, int]] = []
    for i in range(n):
        s = i * chunk
        e = (s + chunk - 1) if i < n - 1 else (total - 1)
        ranges.append((s, e))

    progress = _ProgressTracker(total)
    print(f"Downloading with {n} connection(s)...")
    if n == 1:
        s, e = ranges[0]
        _download_range(url, s, e, cache_path, progress)
    else:
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = [
                ex.submit(_download_range, url, s, e, cache_path, progress)
                for s, e in ranges
            ]
            for fut in as_completed(futures):
                fut.result()
    print()  # newline after progress

    actual = cache_path.stat().st_size
    if actual != total:
        raise RuntimeError(
            f"Downloaded size mismatch: got {actual}, expected {total}. "
            f"Re-run to resume."
        )
    print(f"Tar download OK -> {cache_path}")
    return cache_path


# ---------- Extraction ----------------------------------------------------- #


def _read_id_list(tar: tarfile.TarFile, member_name: str) -> List[str]:
    try:
        m = tar.getmember(member_name)
    except KeyError:
        return []
    f = tar.extractfile(m)
    if f is None:
        return []
    text = f.read().decode("utf-8", errors="ignore")
    return [ln.split()[0].strip() for ln in text.splitlines() if ln.strip()]


def _safe_extract(
    tar: tarfile.TarFile, member: tarfile.TarInfo, dest_path: Path
) -> None:
    """Extract a single member's *content* to ``dest_path`` (renaming the path)."""
    src = tar.extractfile(member)
    if src is None:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 256)


def _filter_existing(members_to_paths: dict, force: bool) -> dict:
    if force:
        return members_to_paths
    return {m: p for m, p in members_to_paths.items() if not p.is_file() or p.stat().st_size == 0}


def extract_subset(
    tar_path: Path,
    dest: Path,
    max_images: int,
    ids_list: str = "seg",
    with_jpeg: bool = True,
    with_xml: bool = True,
    with_mask: bool = True,
    force: bool = False,
) -> dict:
    """
    Extract a subset of VOC2012 from a cached tar to ``<dest>/`` in flat layout.

    Parameters
    ----------
    ids_list
        ``"seg"`` (Segmentation/trainval.txt, ~2913 IDs with masks),
        ``"main"`` (Main/trainval.txt, ~17125 IDs, no masks needed),
        ``"all"``  (union of both).
    max_images
        ``0`` or negative → no cap (all IDs from the chosen list).
    """
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Opening cached tar: {tar_path}")
    with tarfile.open(tar_path, mode="r") as tar:
        seg_trainval = _read_id_list(
            tar, f"{INNER_VOC_PREFIX}/ImageSets/Segmentation/trainval.txt"
        )
        main_trainval = _read_id_list(
            tar, f"{INNER_VOC_PREFIX}/ImageSets/Main/trainval.txt"
        )

        if ids_list == "seg":
            ids = list(seg_trainval)
        elif ids_list == "main":
            ids = list(main_trainval)
        elif ids_list == "all":
            seen: Set[str] = set()
            ids = []
            for s in seg_trainval + main_trainval:
                if s not in seen:
                    seen.add(s)
                    ids.append(s)
        else:
            raise ValueError(f"Unknown ids_list: {ids_list}")

        if not ids:
            raise RuntimeError(
                f"ID list '{ids_list}' is empty. The tar may be incomplete."
            )

        if max_images and max_images > 0:
            ids = ids[:max_images]
        seg_id_set = set(seg_trainval)

        print(f"Selected {len(ids)} ID(s) from list='{ids_list}'.")
        print(
            f"  with_jpeg={with_jpeg} with_xml={with_xml} "
            f"with_mask={with_mask} force={force}"
        )

        # Always copy ImageSets/* into dest so downstream tools can read them.
        list_files = [
            "ImageSets/Main/train.txt",
            "ImageSets/Main/val.txt",
            "ImageSets/Main/trainval.txt",
            "ImageSets/Segmentation/train.txt",
            "ImageSets/Segmentation/val.txt",
            "ImageSets/Segmentation/trainval.txt",
        ]
        wanted: dict = {}
        for rel in list_files:
            wanted[f"{INNER_VOC_PREFIX}/{rel}"] = dest / rel

        for stem in ids:
            if with_jpeg:
                wanted[f"{INNER_VOC_PREFIX}/JPEGImages/{stem}.jpg"] = (
                    dest / "JPEGImages" / f"{stem}.jpg"
                )
            if with_xml:
                wanted[f"{INNER_VOC_PREFIX}/Annotations/{stem}.xml"] = (
                    dest / "Annotations" / f"{stem}.xml"
                )
            if with_mask and stem in seg_id_set:
                wanted[f"{INNER_VOC_PREFIX}/SegmentationClass/{stem}.png"] = (
                    dest / "SegmentationClass" / f"{stem}.png"
                )

        wanted = _filter_existing(wanted, force=force)
        total = len(wanted)
        if total == 0:
            print("Nothing to extract (all targets already exist).")
            return {"ids": ids, "extracted": 0}

        print(f"Extracting {total} file(s) ...")
        done = 0
        last_print = time.time()
        names = list(wanted.keys())
        for member_name in names:
            try:
                m = tar.getmember(member_name)
            except KeyError:
                continue
            _safe_extract(tar, m, wanted[member_name])
            done += 1
            now = time.time()
            if now - last_print >= 1.0 or done == total:
                last_print = now
                pct = 100.0 * done / total
                sys.stdout.write(f"\r  [{pct:5.1f}%] {done}/{total}")
                sys.stdout.flush()
        print()
        print(f"Extracted {done}/{total} files into {dest}")
        return {"ids": ids, "extracted": done}


# ---------- CLI ------------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cache VOC2012 trainval tar (resumable, parallel) and extract a subset "
            "into a flat VOC layout consumed by segment.py / classify.py."
        )
    )
    p.add_argument(
        "--dest",
        default="data/voc/VOC2012_subset_300",
        help="Output VOC root (flat layout). Default: data/voc/VOC2012_subset_300",
    )
    p.add_argument(
        "--cache-dir",
        default="data/voc/_cache",
        help="Where to keep the downloaded tar. Default: data/voc/_cache",
    )
    p.add_argument(
        "--tar-path",
        default=None,
        help="Use an existing tar on disk instead of downloading.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Cap on number of IDs (0 = use the entire chosen list).",
    )
    p.add_argument(
        "--ids-list",
        choices=("seg", "main", "all"),
        default="seg",
        help=(
            "Which list to draw IDs from. "
            "seg = Segmentation/trainval (~2913 IDs, has masks). "
            "main = Main/trainval (~17125 IDs, no masks). "
            "all = union."
        ),
    )
    p.add_argument("--connections", type=int, default=8, help="Parallel HTTP streams.")
    p.add_argument("--no-jpeg", action="store_true", help="Skip JPEGImages.")
    p.add_argument("--no-xml", action="store_true", help="Skip Annotations XML.")
    p.add_argument("--no-mask", action="store_true", help="Skip SegmentationClass.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = _resolve_under_repo(args.dest)

    if args.tar_path:
        tar_path = _resolve_under_repo(args.tar_path)
        if not tar_path.is_file():
            raise FileNotFoundError(f"--tar-path not found: {tar_path}")
        print(f"Using existing tar: {tar_path}")
    else:
        cache_dir = _resolve_under_repo(args.cache_dir)
        tar_path = cache_dir / TAR_FILENAME
        download_tar(tar_path, connections=max(1, int(args.connections)))

    print(f"Destination: {dest}")
    extract_subset(
        tar_path=tar_path,
        dest=dest,
        max_images=args.max_images,
        ids_list=args.ids_list,
        with_jpeg=not args.no_jpeg,
        with_xml=not args.no_xml,
        with_mask=not args.no_mask,
        force=args.force,
    )
    print("Done.")


if __name__ == "__main__":
    main()
