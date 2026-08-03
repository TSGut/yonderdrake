"""Download and decode the BrainWeb phantom used by Caputo-Wismer demos."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

BRAINWEB_PAGE = "https://brainweb.bic.mni.mcgill.ca/anatomic_normal.html"
BRAINWEB_DOWNLOAD = "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1"
BRAINWEB_ALIAS = "phantom_1.0mm_normal_crisp"
BRAINWEB_SHAPE_ZYX = (181, 217, 181)
BRAINWEB_RAW_SHA256 = (
    "8693f9fde4f2b233237f2c5d0c4e4e2aac705bb08efd1e13ae70079ab7456c54"
)
BRAINWEB_SLICE_INDEX = 80
BRAINWEB_SLICE_Z_MM = -72 + BRAINWEB_SLICE_INDEX
MM_PER_MODEL_UNIT = 90.0

LABEL_NAMES = {
    0: "background",
    1: "CSF",
    2: "grey matter",
    3: "white matter",
    4: "fat",
    5: "muscle / skin",
    6: "skin",
    7: "skull",
    8: "glial matter",
    9: "connective tissue",
}


def _download_crisp_phantom(destination: Path) -> None:
    """Download the official unsigned-byte, gzip-compressed crisp phantom."""
    form = urlencode(
        {
            "do_download_alias": BRAINWEB_ALIAS,
            "format_value": "raw_byte",
            "zip_value": "gnuzip",
            "who_name": "",
            "who_institution": "",
            "who_email": "",
            "download_for_real": "[Start download!]",
        }
    ).encode()
    request = Request(
        BRAINWEB_DOWNLOAD,
        data=form,
        headers={"User-Agent": "Yonderdrake BrainWeb demo"},
        method="POST",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310
            temporary.write_bytes(response.read())
        temporary.replace(destination)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the BrainWeb crisp phantom. Download the "
            f"raw-byte/gzip volume from {BRAINWEB_PAGE} and save it as "
            f"{destination}."
        ) from error


def load_brainweb_slice(
    cache_directory: Path,
    *,
    slice_index: int = BRAINWEB_SLICE_INDEX,
) -> np.ndarray:
    """Return one axial label slice, downloading and validating the volume."""
    archive = cache_directory / f"{BRAINWEB_ALIAS}.rawb.gz"
    if not archive.exists():
        _download_crisp_phantom(archive)
    try:
        raw = gzip.decompress(archive.read_bytes())
    except (OSError, EOFError) as error:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"Invalid BrainWeb gzip archive: {archive}") from error
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BRAINWEB_RAW_SHA256:
        raise RuntimeError(
            "BrainWeb phantom checksum mismatch: "
            f"expected {BRAINWEB_RAW_SHA256}, received {digest}"
        )
    expected_size = int(np.prod(BRAINWEB_SHAPE_ZYX))
    if len(raw) != expected_size:
        raise RuntimeError(
            f"BrainWeb phantom has {len(raw)} bytes; expected {expected_size}"
        )
    if not 0 <= slice_index < BRAINWEB_SHAPE_ZYX[0]:
        raise ValueError(
            f"slice index must be in [0, {BRAINWEB_SHAPE_ZYX[0] - 1}]"
        )
    volume = np.frombuffer(raw, dtype=np.uint8).reshape(BRAINWEB_SHAPE_ZYX)
    return np.array(volume[slice_index], copy=True)


def slice_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Return the BrainWeb voxel-centre coordinates in model units."""
    x = (np.arange(BRAINWEB_SHAPE_ZYX[2]) - 90.0) / MM_PER_MODEL_UNIT
    y = (np.arange(BRAINWEB_SHAPE_ZYX[1]) - 126.0) / MM_PER_MODEL_UNIT
    return x, y


def sample_labels(labels: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample a BrainWeb slice at model-coordinate points."""
    x_index = np.rint(
        points[:, 0] * MM_PER_MODEL_UNIT + 90.0
    ).astype(np.int64)
    y_index = np.rint(
        points[:, 1] * MM_PER_MODEL_UNIT + 126.0
    ).astype(np.int64)
    result = np.zeros(points.shape[0], dtype=np.uint8)
    inside = (
        (x_index >= 0)
        & (x_index < labels.shape[1])
        & (y_index >= 0)
        & (y_index < labels.shape[0])
    )
    result[inside] = labels[y_index[inside], x_index[inside]]
    return result
