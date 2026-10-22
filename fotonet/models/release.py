"""Resolve official fotonete checkpoints from their GitHub releases.

Weight generations
------------------
``fotonete``  -> generation 1 -> release ``v1``
``fotonete2`` -> generation 2 -> release ``v2``
``fotonet3``  -> ...          -> release ``v<generation>``

``fotonetn`` (the v1.1 family) lives in the ``v1.1`` release.

Every generation is one graph with different trained weights, and each lives in
its own ``v<generation>`` release. The package version on PyPI is unrelated
to the weight generation: a user on fotonet 8.8.5 asking for ``fotonete`` still
gets the weights published in the ``v1`` release.

Why there is no checksum gate
-----------------------------
An earlier version pinned a SHA-256 and refused to load on mismatch, which made
over-the-air weight updates impossible - publishing new weights broke every
installed copy until the pin was raised in a new release. That is the opposite
of what a floating "latest weights for this generation" pointer should do.

Freshness is instead decided by asking the release itself. A cached checkpoint
is re-downloaded whenever GitHub reports a different ETag for the asset, so a
re-published generation is picked up on the next resolve without touching the
package. The network check is best effort: if GitHub cannot be reached, the
cached checkpoint is used rather than failing the import.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fotonet.models.e.registry import model_generation, normalize_model_id


RELEASE_HOST = "https://github.com/iMGEai/fotonet/releases/download"
CHECKPOINT_FILENAME = "fotonete.pt"
USER_AGENT = "fotonet-release-resolver/2"
CACHE_META_SUFFIX = ".meta.json"


def release_tag(generation: int) -> str:
    """Weight generations publish as v<generation> (v1, v2, ...)."""
    return f"v{int(generation)}"


def release_url(generation: int) -> str:
    return f"{RELEASE_HOST}/{release_tag(generation)}/{CHECKPOINT_FILENAME}"


def checkpoint_filename(generation: int) -> str:
    """Local filename convention: generation 1 keeps the historical name."""
    return CHECKPOINT_FILENAME if int(generation) == 1 else f"fotonete{int(generation)}.pt"


def _cache_dir() -> Path:
    return Path(
        os.environ.get("FOTONETE_CACHE_DIR", Path.home() / ".cache" / "fotonet" / "models")
    ).expanduser()


def _remote_etag(url: str):
    """Return the asset ETag/Last-Modified, or None when it cannot be determined."""
    try:
        request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=20) as response:
            return response.headers.get("ETag") or response.headers.get("Last-Modified")
    except (HTTPError, URLError, OSError, ValueError):
        return None


def _read_meta(path: Path):
    try:
        with open(path.with_suffix(CACHE_META_SUFFIX), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _write_meta(path: Path, url: str, etag) -> None:
    try:
        with open(path.with_suffix(CACHE_META_SUFFIX), "w", encoding="utf-8") as handle:
            json.dump({"url": url, "etag": etag, "fetched": os.path.getmtime(path)}, handle)
    except OSError:
        pass  # A missing sidecar only costs one extra revalidation later.


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=destination.name, suffix=".download", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=120) as response, temporary:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    temporary.write(chunk)
        os.replace(temporary_path, destination)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the official fotonete checkpoint. "
            "Install it manually or set FOTONETE_MODEL_PATH. Download URL: " + url
        ) from exc


def resolve_release_checkpoint(model_ref: str = "fotonete") -> Path:
    """Return the checkpoint for a model id, downloading it when needed.

    ``FOTONETE_MODEL_PATH`` still wins outright, then a matching checkpoint in
    the working directory, then the on-disk cache (refreshed when the release
    asset has changed), and only then the network.
    """
    model_id = normalize_model_id(model_ref)
    if model_id is None:
        raise ValueError(
            f"No released checkpoint is registered for model '{model_ref}'. "
            "Use 'fotonete' or a generation such as 'fotonete2'."
        )
    generation = model_generation(model_id)
    url = release_url(generation)

    override = os.environ.get("FOTONETE_MODEL_PATH")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FOTONETE_MODEL_PATH does not point to a file: {path}")
        return path

    local_checkpoint = Path.cwd() / checkpoint_filename(generation)
    if local_checkpoint.is_file():
        return local_checkpoint

    cached = _cache_dir() / checkpoint_filename(generation)
    if cached.is_file():
        etag = _remote_etag(url)
        if etag is None:
            # Offline or GitHub is unhappy: a stale checkpoint beats no model.
            return cached
        if _read_meta(cached).get("etag") == etag:
            return cached
        try:
            _download(url, cached)
        except RuntimeError:
            return cached  # Keep working on the copy we already trust.
        _write_meta(cached, url, etag)
        return cached

    etag = _remote_etag(url)
    _download(url, cached)
    _write_meta(cached, url, etag)
    return cached

# ---------------------------------------------------------------- v1.1 family
# A v1.1 model's weights are the asset ``<model_id>.pt`` in the ``v1.1``
# release (``fotonetn.pt`` today). A model with no published asset resolves to
# None and the caller warns that it is untrained.
V11_RELEASE_TAG = "v1.1"
MISSING_TTL_SECONDS = 24 * 3600


def _offline():
    return os.environ.get("FOTONET_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def v11_release_url(model_id: str) -> str:
    return f"{RELEASE_HOST}/{V11_RELEASE_TAG}/{model_id}.pt"


def resolve_v11_release(model_id: str):
    """Path to published v1.1 weights for ``model_id``, or None when there are none.

    A cached file wins. A recent "not published" answer is remembered for a
    day so constructing models offline or before a release never stalls.
    ``FOTONET_OFFLINE=1`` skips the network entirely.
    """
    import time

    model_id = str(model_id).strip().lower()
    cached = _cache_dir() / f"{model_id}.pt"
    if cached.is_file():
        return cached
    url = v11_release_url(model_id)
    marker = cached.with_suffix(".pt.missing")
    if marker.is_file() and time.time() - marker.stat().st_mtime < MISSING_TTL_SECONDS:
        try:
            # A miss only counts for the URL it was recorded against.
            if marker.read_text(encoding="utf-8").strip() == url:
                return None
        except OSError:
            pass
    if _offline():
        return None
    try:
        request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=5) as response:
            etag = response.headers.get("ETag") or response.headers.get("Last-Modified")
    except (HTTPError, URLError, OSError, ValueError):
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(url, encoding="utf-8")
        except OSError:
            pass
        return None
    _download(url, cached)
    _write_meta(cached, url, etag)
    return cached
