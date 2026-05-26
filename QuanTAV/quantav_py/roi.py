from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .config import QuantavParams
from .preprocess import (
    as_bool_mask,
    bbox_from_mask,
    ensure_same_shape,
    largest_component,
    remove_small_components,
)


@dataclass(slots=True)
class RoiResult:
    image: np.ndarray | None
    tumor_mask: np.ndarray
    vessel_mask: np.ndarray
    peritumoral_mask: np.ndarray
    crop_slices: tuple[slice, slice, slice]


def prepare_roi(
    image: np.ndarray | None,
    tumor_mask: np.ndarray,
    vessel_mask: np.ndarray,
    params: QuantavParams,
) -> RoiResult:
    """Clean masks and extract peritumoral vessels."""

    if image is not None and image.ndim != 3:
        raise ValueError(f"image must be 3D, got shape {image.shape}")
    tumor = as_bool_mask(tumor_mask, "tumor_mask")
    vessel = as_bool_mask(vessel_mask, "vessel_mask")
    ensure_same_shape(tumor, vessel, image)

    if params.clean_tumor:
        tumor = largest_component(tumor, connectivity=3)
    if params.fill_tumor_holes:
        tumor = ndi.binary_fill_holes(tumor)

    distance_to_tumor = ndi.distance_transform_edt(~tumor, sampling=params.normalized_spacing())
    peritumoral = distance_to_tumor <= float(params.peritumoral_radius_mm)
    vessel = vessel & peritumoral & ~tumor
    vessel = remove_small_components(vessel, params.min_vessel_component_size, connectivity=1)

    crop_slices = tuple(slice(0, s) for s in tumor.shape)
    if params.crop_to_roi:
        margin = tuple(
            int(np.ceil(float(params.crop_margin_mm) / spacing))
            for spacing in params.normalized_spacing()
        )
        crop_slices = bbox_from_mask(peritumoral | vessel | tumor, margin)
        tumor = tumor[crop_slices]
        vessel = vessel[crop_slices]
        peritumoral = peritumoral[crop_slices]
        if image is not None:
            image = image[crop_slices]

    return RoiResult(
        image=image,
        tumor_mask=tumor,
        vessel_mask=vessel,
        peritumoral_mask=peritumoral,
        crop_slices=crop_slices,
    )

