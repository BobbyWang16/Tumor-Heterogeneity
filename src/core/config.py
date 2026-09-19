"""
Configuration management using dataclasses.

Replaces ad-hoc dict configs with typed, validated configuration objects.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class DataConfig:
    """Dataset and path configuration."""
    dataset: str
    data_root: str = "./data"
    use_processed: bool = True  # Use image_process / mask_process if available
    output_root: str = "./results"
    image_suffix: str = ".nii.gz"
    mask_suffix: str = ".nii.gz"

    @property
    def processed_available(self) -> bool:
        base = os.path.join(self.data_root, self.dataset)
        return self.use_processed and all(
            os.path.isdir(os.path.join(base, name))
            for name in ("image_process", "mask_process")
        )

    @property
    def image_dir(self) -> str:
        base = os.path.join(self.data_root, self.dataset)
        if self.processed_available:
            return os.path.join(base, "image_process")
        return os.path.join(base, "image")

    @property
    def mask_dir(self) -> str:
        base = os.path.join(self.data_root, self.dataset)
        if self.processed_available:
            return os.path.join(base, "mask_process")
        return os.path.join(base, "mask")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.output_root, self.dataset)


@dataclass
class ModelConfig:
    """Base model configuration."""
    n_workers: int = 4
    verbose: bool = True
    save_intermediates: bool = False


@dataclass
class BIHConfig(ModelConfig):
    """BIH (Box-counting) configuration."""
    box_sizes: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    max_workers: int = 4


@dataclass
class DHIConfig(ModelConfig):
    """DHI (Density) configuration."""
    clip_percentiles: tuple = (1, 99)
    nbins: int = 64
    win_lcv: int = 3


@dataclass
class HABConfig(ModelConfig):
    """HAB (Habitat) configuration."""
    hu_min: int = -1000
    hu_max: int = 400
    k_habitat: int = 4
    shell_thickness_mm: float = 3.0
    radial_bins: int = 3


@dataclass
class PTHConfig(ModelConfig):
    """PTH (Peritumoral) configuration."""
    ring_mm: float = 5.0
    target_block_size_mm3: float = 5000
    compactness: float = 15.0


@dataclass
class SHIConfig(ModelConfig):
    """SHI (Shape) configuration."""
    target_spacing: float = 1.0
    scales: Optional[List[int]] = None

    def __post_init__(self):
        if self.scales is None:
            self.scales = [1, 2, 4, 8, 16]


@dataclass
class THIConfig(ModelConfig):
    """THI (Texture) configuration."""
    k: int = 3
    padding: int = 2
    clip: tuple = (1, 99)


@dataclass
class ITHFusionConfig(ModelConfig):
    """ITH Fusion Score configuration."""
    n_clusters: int = 6
    adaptive_k: bool = True
    scales: List[int] = field(default_factory=lambda: [1, 2, 3])
    use_hog: bool = True
    spatial_sigma: float = 5.0
    feature_sigma: float = 1.0
    n_neighbors: int = 20
    min_area: int = 200
