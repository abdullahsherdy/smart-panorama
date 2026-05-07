# download_voc_batch.py
# VOC2012 snippet for Stages 5–6: JPEG + SegmentationClass PNG + Annotations XML.
#
# VOC is distributed as ONE ~2 GB tar (no official per-file URLs). Default behaviour:
# **stream that archive** from Oxford (HTTPS) and extract only the tiny subset you asked for —
# the full tar is **not saved to disk**.
# Optional `--cache-tar`: save the tar under data/voc/_cache/ once (reuse for reruns).

from __future__ import annotations

import argparse
import errno
import http.client
import json
import os
import ssl
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://host.robots.ox.ac.uk/pascal/VOC/voc2012"
TRAINVAL_TAR_URL = f"{BASE}/VOCtrainval_11-May-2012.tar"
TRAINVAL_TAR_FILENAME = "VOCtrainval_11-May-2012.tar"
TRAINVAL_TXT_MEMBER = "VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt"
JPEG_PREFIX = "VOCdevkit/VOC2012/JPEGImages/"
ANN_PREFIX = "VOCdevkit/VOC2012/Annotations/"
SEG_CLASS_PREFIX = "VOCdevkit/VOC2012/SegmentationClass/"
DOWNLOAD_MANIFEST_NAME = "_download_manifest.json"

EXPECTED_MIN_TAR_BYTES = 1_500_000_000
CHUNK_BYTES = 8 * 1024 * 1024
MAX_TAR_PASSES_STREAM = 8
MAX_TAR_PASSES_LOCAL = 2
DEFAULT_SNIPPET_BATCH = 100

# Long HTTP(S) tar streams are often reset or truncated; retry with backoff.
STREAM_RETRY_ATTEMPTS = 12
STREAM_RETRY_BACKOFF_SEC = 4.0
DOWNLOAD_RETRY_ATTEMPTS = 12
DOWNLOAD_RETRY_BACKOFF_SEC = 5.0
# Per low-level socket read (not total transfer time); keep high for slow links.
URL_OPEN_TIMEOUT_SEC = 600.0


def _is_transient_stream_error(e: BaseException) -> bool:
    if isinstance(e, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError)):
        return True
    # Truncated HTTP body mid-member → "unexpected end of data"; treat as network drop.
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
        win = getattr(e, "winerror", None)
        if win in (10054, 10053, 10060, 10061):
            return True
        en = getattr(e, "errno", None)
        return en in (errno.ECONNRESET, errno.ETIMEDOUT, errno.EPIPE, errno.ECONNABORTED)
    return False


def _unlink_partial_member(dest: Path, member: tarfile.TarInfo) -> None:
    rel = member.name.replace("\\", "/")
    p = dest / rel
    if p.is_file():
        p.unlink()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_data_dir() -> Path:
    return repo_root() / "data"


def default_voc_dir() -> Path:
    return repo_data_dir() / "voc"


def resolve_dest(dest: str | os.PathLike[str] | None) -> Path:
    if dest is None:
        return default_voc_dir()
    p = Path(dest)
    return p.resolve() if p.is_absolute() else (repo_root() / p)


def voc2012_path(voc_root: Path) -> Path:
    return voc_root / "VOCdevkit" / "VOC2012"


def _cache_tar_path(voc_root: Path) -> Path:
    return voc_root / "_cache" / TRAINVAL_TAR_FILENAME


def _safe_extract(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: str) -> None:
    if sys.version_info >= (3, 12):
        tar.extract(member, dest, filter="data")  # type: ignore[call-arg]
    else:
        tar.extract(member, dest)


def _download_trainval_tar_once(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        "Downloading VOC trainval tar once (~2 GB) → "
        f"{out_path.relative_to(repo_root())} …"
    )
    req = urllib.request.Request(
        TRAINVAL_TAR_URL,
        headers={"User-Agent": "smart-panorama-voc-helper/1.0"},
    )
    last_err: BaseException | None = None
    for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
        tmp_fd, tmp_name = tempfile.mkstemp(
            suffix=".tar.part", dir=out_path.parent, text=False
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            with urllib.request.urlopen(req, timeout=URL_OPEN_TIMEOUT_SEC) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total and done % (64 * CHUNK_BYTES) < CHUNK_BYTES:
                            print(f"  … {done / 1e9:.2f} / {total / 1e9:.2f} GB")
            if done < EXPECTED_MIN_TAR_BYTES:
                raise RuntimeError(
                    f"Download too small ({done} bytes); remove partial file and retry."
                )
            os.replace(tmp_path, out_path)
            print(f"  saved ({done / 1e9:.2f} GB)")
            return
        except RuntimeError:
            if tmp_path.is_file():
                tmp_path.unlink(missing_ok=True)
            raise
        except Exception as e:
            last_err = e
            if tmp_path.is_file():
                tmp_path.unlink(missing_ok=True)
            if not _is_transient_stream_error(e):
                raise
            wait = min(
                DOWNLOAD_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)),
                180.0,
            )
            print(
                f"  download interrupted ({e.__class__.__name__}), "
                f"retry {attempt}/{DOWNLOAD_RETRY_ATTEMPTS} in {wait:.0f}s …"
            )
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def _ensure_cached_trainval_tar(voc_root: Path) -> Path:
    cache = _cache_tar_path(voc_root)
    if cache.is_file() and cache.stat().st_size >= EXPECTED_MIN_TAR_BYTES:
        sz = cache.stat().st_size
        print(f"[cache] Using trainval tar ({sz / 1e9:.2f} GB)")
        return cache
    if cache.is_file():
        cache.unlink()
    _download_trainval_tar_once(cache)
    return cache


def _read_trainval_ids(voc2012_root: Path) -> list[str]:
    seg = voc2012_root / "ImageSets" / "Segmentation" / "trainval.txt"
    if not seg.is_file():
        raise FileNotFoundError(f"Missing {seg}")
    return [ln.strip() for ln in seg.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _select_batch_ids(all_ids: list[str], max_images: int) -> list[str]:
    if max_images <= 0:
        return list(all_ids)
    return list(all_ids[:max_images])


def _extract_from_local_tar(dest: Path, wanted: set[str], tar_path: Path) -> set[str]:
    remaining = set(wanted)
    if not remaining:
        return remaining
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            name = member.name.replace("\\", "/")
            if name not in remaining:
                continue
            _safe_extract(tar, member, str(dest))
            remaining.remove(name)
            if not remaining:
                break
    return remaining


def _extract_from_url_stream(dest: Path, wanted: set[str]) -> set[str]:
    remaining = set(wanted)
    if not remaining:
        return remaining
    dest.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        TRAINVAL_TAR_URL,
        headers={"User-Agent": "smart-panorama-voc-helper/1.0"},
    )
    last_err: BaseException | None = None
    for attempt in range(1, STREAM_RETRY_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=URL_OPEN_TIMEOUT_SEC) as response:
                with tarfile.open(fileobj=response, mode="r|") as tar:
                    for member in tar:
                        name = member.name.replace("\\", "/")
                        if name not in remaining:
                            continue
                        try:
                            _safe_extract(tar, member, str(dest))
                        except Exception:
                            _unlink_partial_member(dest, member)
                            raise
                        remaining.remove(name)
                        if not remaining:
                            break
            return remaining
        except Exception as e:
            last_err = e
            if not _is_transient_stream_error(e):
                raise
            wait = min(STREAM_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)), 120.0)
            print(
                f"      stream interrupted ({e.__class__.__name__}); "
                f"retry stream {attempt}/{STREAM_RETRY_ATTEMPTS} in {wait:.0f}s …"
            )
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def _extract_until_done(
    dest: Path, wanted: set[str], label: str, *, local_tar: Path | None
) -> None:
    missing = set(wanted)
    n_pass = MAX_TAR_PASSES_LOCAL if local_tar else MAX_TAR_PASSES_STREAM
    for attempt in range(1, n_pass + 1):
        if not missing:
                    break
        print(f"    pass {attempt}/{n_pass} — {len(missing)} member(s) ({label}) …")
        if local_tar is not None:
            missing = _extract_from_local_tar(dest, missing, local_tar)
        else:
            missing = _extract_from_url_stream(dest, missing)
    if missing:
        raise RuntimeError(
            f"{label}: still missing {len(missing)} paths. Example: {next(iter(missing))}"
        )


def _phase_trainval_txt(dest: Path, local_tar: Path | None) -> None:
    tv = voc2012_path(dest) / "ImageSets" / "Segmentation" / "trainval.txt"
    if tv.is_file():
        print("Phase 1: trainval.txt OK — skip")
        return
    print("Phase 1: extract trainval.txt …")
    _extract_until_done(dest, {TRAINVAL_TXT_MEMBER}, "trainval", local_tar=local_tar)


def _wanted_for_ids(ids: list[str]) -> set[str]:
    out: set[str] = set()
    for i in ids:
        out.add(f"{JPEG_PREFIX}{i}.jpg")
        out.add(f"{ANN_PREFIX}{i}.xml")
        out.add(f"{SEG_CLASS_PREFIX}{i}.png")
    return out


def _write_manifest(path: Path, batch_ids: list[str]) -> None:
    payload = {
        "image_ids": batch_ids,
        "n_ids": len(batch_ids),
        "assets": ["JPEGImages", "Annotations", "SegmentationClass"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_voc_snippet(
    dest: str | os.PathLike[str] | None = None,
    *,
    max_images: int = DEFAULT_SNIPPET_BATCH,
    cache_tar: bool = False,
) -> Path:
    """
    Pull a small VOC trainval snippet (JPEG + XML + mask PNG per ID).

    Parameters
    ----------
    cache_tar
        If ``True``, download the trainval tarball to ``data/voc/_cache/`` and reuse it
        (saves bandwidth on reruns). If ``False`` (default), **only snippet files are written**
        locally — the full `.tar` is **not saved**; extraction uses HTTP streaming (multiple
        passes may still hit the Oxford server).

    Returns
    -------
    VOC root ``Path``.
    """
    root = resolve_dest(dest)
    print(f"VOC root: {root}")
    print(
        "Mode: snippet only (streaming HTTP; **no full .tar saved**)"
        if not cache_tar
        else "Mode: cache full tar under data/voc/_cache (--cache-tar)"
    )
    root.mkdir(parents=True, exist_ok=True)

    local: Path | None = _ensure_cached_trainval_tar(root) if cache_tar else None

    _phase_trainval_txt(root, local)
    voc2012 = voc2012_path(root)
    batch = _select_batch_ids(_read_trainval_ids(voc2012), max_images)

    for d in ("JPEGImages", "Annotations", "SegmentationClass"):
        (voc2012 / d).mkdir(parents=True, exist_ok=True)

    _write_manifest(voc2012 / DOWNLOAD_MANIFEST_NAME, batch)

    wanted = _wanted_for_ids(batch)
    print(f"Phase 2: {len(batch)} IDs × 3 files = {len(wanted)} …")
    _extract_until_done(root, wanted, "snippet", local_tar=local)

    rel = voc2012.relative_to(repo_root())
    print("Done.")
    print(f"  JPEGs: {rel / 'JPEGImages'}")
    print(f"  Masks: {rel / 'SegmentationClass'}")
    print(f"  XMLs : {rel / 'Annotations'}")
    return root


def download_voc_stages_5_and_6(
    dest: str | os.PathLike[str] | None = None,
    *,
    max_images: int = DEFAULT_SNIPPET_BATCH,
    want_jpeg: bool = True,
    want_xml: bool = True,
    want_masks: bool = True,
    cache_tar: bool = False,
) -> Path:
    if not (want_jpeg and want_xml and want_masks):
        raise ValueError("JPEG + XML + masks only (single snippet mode).")
    return download_voc_snippet(dest, max_images=max_images, cache_tar=cache_tar)

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VOC snippet: JPEG + mask PNG + annotation XML.")
    p.add_argument(
        "--dest",
        default=str(default_voc_dir().relative_to(repo_root())),
        help="Relative to repo root",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_SNIPPET_BATCH,
        help=f"First N trainval IDs (default {DEFAULT_SNIPPET_BATCH}); 0 = all ~2913",
    )
    p.add_argument(
        "--cache-tar",
        action="store_true",
        help=(
            "Also save the full VOC trainval tarball (~2 GB) under data/voc/_cache/. "
            "Default: no — only snippet files are stored."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_cli()
    download_voc_snippet(dest=a.dest, max_images=a.batch, cache_tar=a.cache_tar)
