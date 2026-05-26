"""
ITH Fusion Score (ITH-FS) — Comprehensive Heterogeneity Assessment.

Multi-scale 3D voxel features + spatial spectral clustering + heterogeneity confidence map.
"""

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from src.core import BaseHeterogeneityModel, ITHFusionConfig


EPS = 1e-8


# ------------------------------------------------------------------ #
#  Multi-scale feature extraction
# ------------------------------------------------------------------ #

def _extract_voxel_features(vol, msk, scales=[1, 2, 3]):
    """Extract multi-scale voxel features."""
    coords = np.argwhere(msk > 0)
    all_features = []

    for r in scales:
        feats = []
        for z, y, x in coords:
            z0, z1 = max(0, z - r), min(vol.shape[0], z + r + 1)
            y0, y1 = max(0, y - r), min(vol.shape[1], y + r + 1)
            x0, x1 = max(0, x - r), min(vol.shape[2], x + r + 1)
            patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
            vals = patch[msk[z0:z1, y0:y1, x0:x1] > 0]
            if vals.size == 0:
                vals = np.array([vol[z, y, x]], dtype=np.float32)

            mean, std = float(vals.mean()), float(vals.std())
            vmin, vmax = float(vals.min()), float(vals.max())
            prange = vmax - vmin
            center = float(vol[z, y, x])
            diff = vals - center
            contrast = float((diff ** 2).mean()) if vals.size else 0.0
            grad = float(np.abs(diff).mean()) if vals.size else 0.0

            feats.append([center, mean, std, vmin, vmax, prange, contrast, grad])

        all_features.append(np.array(feats, dtype=np.float32))

    features = np.hstack(all_features)
    return StandardScaler().fit_transform(features), coords


# ------------------------------------------------------------------ #
#  Spatial spectral clustering
# ------------------------------------------------------------------ #

def _build_graph(features, coords, n_neighbors=20, sigma_feat=1.0, sigma_space=5.0):
    """Build spatial-feature joint similarity graph."""
    N = features.shape[0]
    knn_graph = kneighbors_graph(features, n_neighbors=n_neighbors, mode='distance')
    # Use COO for consistent row/col/data
    knn_coo = knn_graph.tocoo()
    rows, cols, data = knn_coo.row, knn_coo.col, knn_coo.data

    feat_weights = np.exp(-data ** 2 / (2 * sigma_feat ** 2))
    spatial_dists = np.linalg.norm(coords[rows] - coords[cols], axis=1)
    spatial_weights = np.exp(-spatial_dists ** 2 / (2 * sigma_space ** 2))

    W = csr_matrix((feat_weights * spatial_weights, (rows, cols)), shape=(N, N))
    return (W + W.T) / 2


def _spectral_cluster(features, mask, coords, n_clusters=6, n_neighbors=20,
                      sigma_feat=1.0, sigma_space=5.0):
    """Spatial spectral clustering using sparse normalized Laplacian."""
    N = features.shape[0]
    W = _build_graph(features, coords, n_neighbors, sigma_feat, sigma_space)
    D_diag = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = 1.0 / np.sqrt(D_diag + EPS)

    # Compute D^{-1/2} W D^{-1/2} sparsely
    W_coo = W.tocoo()
    rows, cols, data = W_coo.row, W_coo.col, W_coo.data
    data_norm = data * D_inv_sqrt[rows] * D_inv_sqrt[cols]
    W_norm = csr_matrix((data_norm, (rows, cols)), shape=(N, N))

    # L_sym = I - D^{-1/2} W D^{-1/2} as sparse matrix
    L_sym = csr_matrix((N, N))
    L_sym.setdiag(1.0)
    L_sym = L_sym - W_norm

    try:
        eigenvalues, eigen_vectors = eigsh(L_sym, k=n_clusters, which='SM', maxiter=min(N * 10, 100000))
    except Exception:
        # Fallback: use fewer clusters or shift-invert mode
        eigenvalues, eigen_vectors = eigsh(L_sym, k=min(n_clusters, N - 1), which='SM',
                                            sigma=0.0, maxiter=min(N * 20, 200000))

    row_norms = np.linalg.norm(eigen_vectors, axis=1, keepdims=True)
    eigen_vectors_norm = eigen_vectors / (row_norms + EPS)

    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(eigen_vectors_norm)

    label_map = np.zeros_like(mask, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords):
        label_map[z, y, x] = labels[i] + 1

    return label_map, eigen_vectors_norm


# ------------------------------------------------------------------ #
#  ITH score computation
# ------------------------------------------------------------------ #

def _compute_ith_score(label_map, mask, min_area=200):
    """Compute ITH score from cluster label map."""
    total_size = np.sum(mask > 0)
    if total_size == 0:
        return 0.0, {}

    unique_labels = np.unique(label_map[mask > 0])
    num_regions_list, max_area_list = [], []
    structure = ndimage.generate_binary_structure(3, 3)

    for label in unique_labels:
        if label == 0:
            continue
        labeled, num_regions = ndimage.label(label_map == label, structure=structure)
        valid_regions, max_area = 0, 0
        for region_id in range(1, num_regions + 1):
            area = np.sum(labeled == region_id)
            if area > (1 if total_size <= min_area else 1):
                valid_regions += 1
                max_area = max(max_area, area)
        if valid_regions > 0:
            num_regions_list.append(valid_regions)
            max_area_list.append(max_area)

    if not num_regions_list:
        return 0.0, {}

    ith_base = sum(float(max_area_list[k]) / num_regions_list[k] for k in range(len(num_regions_list))) / total_size
    ith_base = 1 - ith_base

    # Spatial coherence
    coherence = _spatial_coherence(label_map, mask)
    coherence_bonus = 1.0 + 0.3 * (coherence - 0.5)
    ith_score = np.clip(ith_base * coherence_bonus, 0.0, 1.0)

    return ith_score, {"ith_base": ith_base, "coherence": coherence, "n_clusters": len(num_regions_list)}


def _spatial_coherence(label_map, mask):
    """Compute spatial coherence score using vectorized operations."""
    if mask.sum() == 0:
        return 0.0

    total_neighbors, consistent_neighbors = 0, 0
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                shifted = np.roll(label_map, (dz, dy, dx), axis=(0, 1, 2))
                shifted_mask = np.roll(mask, (dz, dy, dx), axis=(0, 1, 2))
                valid = (mask > 0) & (shifted_mask > 0)
                total_neighbors += valid.sum()
                consistent_neighbors += ((label_map == shifted) & valid).sum()

    return consistent_neighbors / total_neighbors if total_neighbors > 0 else 0.0


# ------------------------------------------------------------------ #
#  ITH-FS Model
# ------------------------------------------------------------------ #

class ITHFusionModel(BaseHeterogeneityModel):
    """ITH Fusion Score model."""

    name = "ith_fusion"
    config_class = ITHFusionConfig

    def compute(self, image, mask):
        features, coords = _extract_voxel_features(image, mask, scales=self.cfg.scales)
        n_clusters = self.cfg.n_clusters

        label_map, eigen_vectors = _spectral_cluster(
            features, mask, coords, n_clusters=n_clusters,
            n_neighbors=self.cfg.n_neighbors,
            sigma_feat=self.cfg.feature_sigma,
            sigma_space=self.cfg.spatial_sigma
        )

        ith_score, details = _compute_ith_score(label_map, mask, min_area=self.cfg.min_area)

        # Confidence map
        confidence = self._compute_confidence(label_map, mask, eigen_vectors)

        return {
            "score": float(ith_score),
            "features": {"ith_score": float(ith_score), "coherence": details["coherence"]},
            "label_map": label_map,
            "confidence_map": confidence,
            "details": details,
        }

    @staticmethod
    def _compute_confidence(label_map, mask, eigen_vectors):
        """Compute heterogeneity confidence map."""
        coords = np.argwhere(mask > 0)
        N = len(coords)
        confidence = np.ones(N, dtype=np.float32)

        for label in np.unique(label_map[mask > 0]):
            if label == 0:
                continue
            label_mask = (label_map[mask > 0] == label)
            if not np.any(label_mask):
                continue
            cluster_center = np.mean(eigen_vectors[label_mask], axis=0)
            cluster_vecs = eigen_vectors[label_mask]
            dists = np.linalg.norm(cluster_vecs - cluster_center, axis=1)
            max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
            confidence[label_mask] = 1.0 - dists / max_dist

        confidence_map = np.zeros_like(mask, dtype=np.float32)
        for i, (z, y, x) in enumerate(coords):
            confidence_map[z, y, x] = confidence[i]

        return confidence_map
