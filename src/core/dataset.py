"""
Unified dataset loader for ITH project.

Manages multi-modal data per dataset:
- image / image_process: CT volumes (raw / preprocessed)
- mask / mask_process: segmentation masks (raw / preprocessed)
- clinical.xlsx: clinical features (age, stage, survival, etc.)
- rad_features.xlsx: radiomics features (pyradiomics)
- gene/: gene expression data (optional)
- multimask/: multiple observer masks (optional, for ICC)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from .io import load_image, load_seg, match_image_mask_pairs
from .config import DataConfig


class DatasetLoader:
    """Load and manage all data for a single dataset."""

    def __init__(
        self,
        dataset_name: str,
        data_root: str = "./data",
        use_processed: bool = True,
        image_suffix: str = ".nii.gz",
        mask_suffix: str = ".nii.gz",
    ):
        self.dataset = dataset_name
        self.data_root = Path(data_root)
        self.use_processed = use_processed
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        # Resolve paths
        base = self.data_root / dataset_name
        self._base = base

        config = DataConfig(dataset_name, data_root=data_root, use_processed=use_processed)
        self.image_dir = Path(config.image_dir)
        self.mask_dir = Path(config.mask_dir)

        self.clinical_path = base / "clinical.xlsx"
        self.rad_features_path = base / "rad_features.xlsx"
        self.gene_dir = base / "gene"
        self.multimask_dir = base / "multimask"

        # Cache
        self._clinical: Optional[pd.DataFrame] = None
        self._rad_features: Optional[pd.DataFrame] = None
        self._pairs: Optional[List[Tuple[str, str, str]]] = None

    # ------------------------------------------------------------------ #
    #  Image / Mask pairs
    # ------------------------------------------------------------------ #

    @property
    def pairs(self) -> List[Tuple[str, str, str]]:
        """Return list of (case_id, image_path, mask_path)."""
        if self._pairs is None:
            self._pairs = match_image_mask_pairs(
                str(self.image_dir),
                str(self.mask_dir),
                self.image_suffix,
                self.mask_suffix,
            )
        return self._pairs

    def load_case(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CT image and mask for a specific case."""
        id2path = {cid: (img, msk) for cid, img, msk in self.pairs}
        if case_id not in id2path:
            raise ValueError(f"Case {case_id} not found in {self.dataset}")
        img_path, msk_path = id2path[case_id]
        vol, _ = load_image(img_path)
        msk, _ = load_seg(msk_path)
        return vol, msk

    # ------------------------------------------------------------------ #
    #  Clinical data
    # ------------------------------------------------------------------ #

    @property
    def clinical(self) -> Optional[pd.DataFrame]:
        """Load clinical features."""
        if self._clinical is None:
            if self.clinical_path.exists():
                self._clinical = pd.read_excel(self.clinical_path)
                # Standardize ID column name
                if "PatientID" in self._clinical.columns:
                    self._clinical = self._clinical.rename(columns={"PatientID": "case_id"})
            else:
                warnings.warn(f"Clinical data not found: {self.clinical_path}")
                self._clinical = pd.DataFrame()
        return self._clinical

    def get_survival_data(self) -> Optional[pd.DataFrame]:
        """Return survival data (time, event) if available."""
        df = self.clinical
        if df is None or df.empty:
            return None
        cols = [c for c in ["case_id", "time", "event"] if c in df.columns]
        if len(cols) < 3:
            return None
        return df[cols].copy()

    # ------------------------------------------------------------------ #
    #  Radiomics features
    # ------------------------------------------------------------------ #

    @property
    def radiomics(self) -> Optional[pd.DataFrame]:
        """Load radiomics features."""
        if self._rad_features is None:
            if self.rad_features_path.exists():
                self._rad_features = pd.read_excel(self.rad_features_path)
                # Standardize ID column
                first_col = self._rad_features.columns[0]
                if "Unnamed" in first_col or first_col in ["0", "index"]:
                    self._rad_features = self._rad_features.rename(
                        columns={first_col: "case_id"}
                    )
            else:
                warnings.warn(f"Radiomics features not found: {self.rad_features_path}")
                self._rad_features = pd.DataFrame()
        return self._rad_features

    def get_feature_names(self, category: Optional[str] = None) -> List[str]:
        """Get radiomics feature names, optionally filtered by category."""
        df = self.radiomics
        if df is None or df.empty:
            return []
        exclude = {"case_id", "PatientID", "index"}
        names = [c for c in df.columns if c not in exclude]
        if category:
            names = [n for n in names if category in n.lower()]
        return names

    # ------------------------------------------------------------------ #
    #  Gene data (optional)
    # ------------------------------------------------------------------ #

    @property
    def has_gene_data(self) -> bool:
        return self.gene_dir.exists()

    def load_gene_data(self) -> Optional[pd.DataFrame]:
        """Load gene expression data if available."""
        if not self.has_gene_data:
            return None
        # Look for common gene data formats
        for pattern in ["*.csv", "*.xlsx", "*.tsv", "*.txt"]:
            files = list(self.gene_dir.glob(pattern))
            if files:
                f = files[0]
                if f.suffix == ".csv":
                    return pd.read_csv(f, index_col=0)
                elif f.suffix in [".xlsx", ".xls"]:
                    return pd.read_excel(f, index_col=0)
                elif f.suffix in [".tsv", ".txt"]:
                    return pd.read_csv(f, sep="\t", index_col=0)
        warnings.warn(f"No recognized gene data file in {self.gene_dir}")
        return None

    # ------------------------------------------------------------------ #
    #  Multi-observer masks (for ICC, optional)
    # ------------------------------------------------------------------ #

    @property
    def has_multimask(self) -> bool:
        return bool(self.get_observer_mask_dirs())

    def get_observer_mask_dirs(self) -> Dict[str, Path]:
        """Return {observer_name: mask_directory} for ICC testing."""
        observers = {
            path.name[len("mask_"):]: path
            for path in sorted(self._base.glob("mask_*"))
            if path.is_dir() and path.name != "mask_process"
        }
        if self.multimask_dir.is_dir():
            for subdir in sorted(self.multimask_dir.iterdir()):
                if subdir.is_dir():
                    observers[subdir.name] = subdir
        return observers

    # ------------------------------------------------------------------ #
    #  Merged table builder
    # ------------------------------------------------------------------ #

    def build_merged_table(
        self,
        include_clinical: bool = True,
        include_radiomics: bool = True,
        include_ith_scores: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build a merged table with clinical + radiomics + ITH scores.

        Args:
            include_clinical: whether to include clinical features
            include_radiomics: whether to include radiomics features
            include_ith_scores: DataFrame with ITH model scores (case_id, *_score)

        Returns:
            Merged DataFrame with case_id as index
        """
        dfs = []

        if include_clinical:
            c = self.clinical
            if c is not None and not c.empty:
                dfs.append(c.set_index("case_id"))

        if include_radiomics:
            r = self.radiomics
            if r is not None and not r.empty:
                dfs.append(r.set_index("case_id"))

        if include_ith_scores is not None:
            dfs.append(include_ith_scores.set_index("case_id"))

        if not dfs:
            return pd.DataFrame()

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.join(df, how="outer", rsuffix="_dup")

        # Drop duplicate columns
        merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged.reset_index()

    # ------------------------------------------------------------------ #
    #  Info
    # ------------------------------------------------------------------ #

    def info(self) -> Dict:
        """Return summary info about the dataset."""
        n_pairs = len(self.pairs)
        n_clinical = len(self.clinical) if self.clinical is not None else 0
        n_rad = len(self.radiomics) if self.radiomics is not None else 0
        n_rad_feats = len(self.get_feature_names()) if self.radiomics is not None else 0

        info = {
            "dataset": self.dataset,
            "n_cases_image_mask": n_pairs,
            "n_clinical_records": n_clinical,
            "n_radiomics_records": n_rad,
            "n_radiomics_features": n_rad_feats,
            "image_dir": str(self.image_dir),
            "mask_dir": str(self.mask_dir),
            "has_gene": self.has_gene_data,
            "has_multimask": self.has_multimask,
            "n_observers": len(self.get_observer_mask_dirs()) if self.has_multimask else 0,
        }
        return info

    def __repr__(self) -> str:
        info = self.info()
        lines = [
            f"DatasetLoader({info['dataset']})",
            f"  Cases (image+mask): {info['n_cases_image_mask']}",
            f"  Clinical records:   {info['n_clinical_records']}",
            f"  Radiomics records:  {info['n_radiomics_records']} x {info['n_radiomics_features']} features",
            f"  Gene data:          {info['has_gene']}",
            f"  Multi-observer:     {info['has_multimask']} ({info['n_observers']} observers)",
        ]
        return "\n".join(lines)
