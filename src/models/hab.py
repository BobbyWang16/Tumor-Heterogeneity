"""
HAB (Habitat Analysis Based) heterogeneity model.

Segments tumor into k habitat regions using GMM clustering with spatial priors.
"""

import numpy as np
from scipy import ndimage
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.mixture import GaussianMixture
from skimage.measure import label as connected_components

from src.core import BaseHeterogeneityModel, HABConfig, clip_hu


class HABModel(BaseHeterogeneityModel):
    """Habitat-based intratumoral heterogeneity model."""

    name = "hab"
    config_class = HABConfig

    def compute(self, image, mask):
        image = clip_hu(image, mask, self.cfg.hu_min, self.cfg.hu_max)
        priors = self._compute_spatial_priors(mask, self.cfg.shell_thickness_mm, self.cfg.radial_bins)
        habitat_map, habitat_info = self._compute_habitat_with_priors(
            image, mask, priors, self.cfg.k_habitat, self.cfg.hu_min, self.cfg.hu_max
        )
        if habitat_map is None:
            return {"score": 0.0, "features": {}, "label_map": np.zeros_like(mask, dtype=np.int16), "details": {}}

        features = self._compute_hab_features(image, mask, habitat_map, priors, habitat_info, self.cfg.k_habitat)
        self.feature_names = list(features.keys())
        return {"score": features.get("H_HAB", 0.0), "features": features, "label_map": habitat_map, "details": habitat_info}

    @staticmethod
    def _compute_spatial_priors(mask_array, shell_thickness_mm, radial_bins):
        distance_to_boundary = ndimage.distance_transform_edt(mask_array)
        max_dist = distance_to_boundary.max()
        normalized_radial = distance_to_boundary / max_dist if max_dist > 0 else np.zeros_like(mask_array, dtype=float)
        priors = {"radial_distance": normalized_radial}

        radial_labels = np.zeros_like(mask_array, dtype=int)
        tumor_voxels = mask_array > 0
        if max_dist > 0:
            bins = np.linspace(0, 1, radial_bins + 1)
            radial_labels[tumor_voxels] = np.digitize(normalized_radial[tumor_voxels], bins, right=False)
        priors["radial_layers"] = radial_labels

        shell_mask = (mask_array > 0) & (distance_to_boundary <= shell_thickness_mm)
        priors["boundary_shell"] = shell_mask.astype(int)
        return priors

    @staticmethod
    def _compute_habitat_with_priors(image_array, mask_array, priors, k_habitat, hu_min, hu_max):
        tumor_mask = mask_array > 0
        n_voxels = tumor_mask.sum()
        if n_voxels < 50 or n_voxels < k_habitat:
            return None, {}

        features_list = []
        hu_norm = (image_array - hu_min) / (hu_max - hu_min)
        features_list.append(hu_norm[tumor_mask])

        for sigma in [1, 2, 3]:
            log_response = ndimage.gaussian_laplace(image_array.astype(float), sigma=sigma)
            log_vals = log_response[tumor_mask]
            log_norm = (log_vals - log_vals.min()) / (log_vals.max() - log_vals.min() + 1e-6)
            features_list.append(log_norm)

        features_list.append(priors["radial_distance"][tumor_mask] * 2.0)
        feature_vectors = np.stack(features_list, axis=1)

        gmm = GaussianMixture(n_components=k_habitat, covariance_type="full", random_state=42, n_init=10)
        labels = gmm.fit_predict(feature_vectors)

        mean_hu_per_habitat = [image_array[tumor_mask][labels == k].mean() if (labels == k).sum() > 0 else hu_min for k in range(k_habitat)]
        sorted_indices = np.argsort(mean_hu_per_habitat)
        label_mapping = {old: new for new, old in enumerate(sorted_indices)}
        labels_sorted = np.array([label_mapping[l] for l in labels])

        habitat_map = np.zeros_like(mask_array, dtype=int)
        habitat_map[tumor_mask] = labels_sorted + 1
        habitat_map = ndimage.median_filter(habitat_map, size=3)
        habitat_map[mask_array == 0] = 0

        habitat_names = []
        for k in range(k_habitat):
            mean_hu = mean_hu_per_habitat[sorted_indices[k]]
            if mean_hu < -50:
                habitat_names.append(f"Habitat{k+1}_Necrotic-like")
            elif mean_hu < -300:
                habitat_names.append(f"Habitat{k+1}_GGO-like")
            elif mean_hu < -200:
                habitat_names.append(f"Habitat{k+1}_Intermediate")
            else:
                habitat_names.append(f"Habitat{k+1}_Solid-like")

        return habitat_map, {"names": habitat_names, "mean_hu": [mean_hu_per_habitat[sorted_indices[k]] for k in range(k_habitat)]}

    @staticmethod
    def _compute_hab_features(image_array, mask_array, habitat_map, priors, habitat_info, k_habitat):
        features = {}
        tumor_mask = mask_array > 0
        labels_raw = habitat_map[tumor_mask] - 1
        radial_layers_raw = priors["radial_layers"][tumor_mask]
        radial_distance_raw = priors["radial_distance"][tumor_mask]
        shell_mask_raw = priors["boundary_shell"][tumor_mask].astype(bool)
        image_values_raw = image_array[tumor_mask]

        valid_mask = (labels_raw >= 0) & (labels_raw < k_habitat)
        labels = labels_raw[valid_mask]
        radial_layers = radial_layers_raw[valid_mask]
        radial_distance = radial_distance_raw[valid_mask]
        shell_mask = shell_mask_raw[valid_mask]
        image_values = image_values_raw[valid_mask]

        if len(labels) == 0:
            return features

        volumes = [(labels == k).sum() for k in range(k_habitat)]
        probs = np.array(volumes) / sum(volumes)
        features["HAB_comp_entropy"] = float(entropy(probs[probs > 0]))
        for k, name in enumerate(habitat_info["names"]):
            features[f"{name}_fraction"] = float(volumes[k] / sum(volumes))

        radial_habs = []
        radial_bins = priors["radial_layers"].max()
        for layer_id in range(1, radial_bins + 1):
            layer_mask = radial_layers == layer_id
            if layer_mask.sum() > 0:
                layer_probs = np.bincount(labels[layer_mask], minlength=k_habitat) / len(labels[layer_mask])
                radial_habs.append(layer_probs)

        if len(radial_habs) > 1:
            js_distances = [jensenshannon(radial_habs[i], radial_habs[i + 1]) for i in range(len(radial_habs) - 1)]
            features["HAB_radial_stratification"] = float(np.mean(js_distances))
            features["HAB_core_shell_divergence"] = float(jensenshannon(radial_habs[0], radial_habs[-1]))
        else:
            features["HAB_radial_stratification"] = 0.0
            features["HAB_core_shell_divergence"] = 0.0

        if shell_mask.sum() > 10:
            shell_labels = labels[shell_mask]
            shell_probs = np.bincount(shell_labels, minlength=k_habitat) / len(shell_labels)
            features["HAB_shell_entropy"] = float(entropy(shell_probs + 1e-10))

            shell_habitat_map = habitat_map.copy()
            shell_habitat_map[priors["boundary_shell"] == 0] = 0
            interface_voxels, total_neighbors = 0, 0
            for shift in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                shifted = np.roll(shell_habitat_map, shift, axis=(0, 1, 2))
                valid = (shell_habitat_map > 0) & (shifted > 0)
                interface_voxels += ((shell_habitat_map != shifted) & valid).sum()
                total_neighbors += valid.sum()
            features["HAB_shell_mixing"] = float(interface_voxels / (total_neighbors + 1e-6))
        else:
            features["HAB_shell_entropy"] = 0.0
            features["HAB_shell_mixing"] = 0.0

        total_fragments = 0
        for k in range(k_habitat):
            habitat_k = (habitat_map == k + 1).astype(int)
            if habitat_k.sum() > 0:
                total_fragments += connected_components(habitat_k, connectivity=3).max()
        features["HAB_fragmentation"] = float(np.log1p(total_fragments))

        interface_voxels, total_boundary = 0, 0
        for shift in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            shifted = np.roll(habitat_map, shift, axis=(0, 1, 2))
            boundary = (habitat_map > 0) & (shifted > 0) & (habitat_map != shifted)
            interface_voxels += boundary.sum()
            total_boundary += ((habitat_map > 0) & (shifted > 0)).sum()
        features["HAB_global_mixing"] = float(interface_voxels / (total_boundary + 1e-6))

        ggo_mask = image_values < -300
        solid_mask = image_values >= -100
        features["HAB_ggo_fraction"] = float(ggo_mask.sum() / len(labels))
        features["HAB_solid_fraction"] = float(solid_mask.sum() / len(labels))
        features["HAB_subsolid_coexistence"] = float(4 * features["HAB_ggo_fraction"] * features["HAB_solid_fraction"])

        necrotic_mask = image_values < -30
        features["HAB_necrotic_fraction"] = float(necrotic_mask.sum() / len(labels))
        features["HAB_necrotic_centrality"] = float(radial_distance[necrotic_mask].mean() if necrotic_mask.sum() > 0 else 0.0)

        features["H_HAB"] = float(np.clip(
            0.15 * np.clip(features["HAB_comp_entropy"] / 1.5, 0, 1) +
            0.25 * features["HAB_radial_stratification"] +
            0.20 * features["HAB_core_shell_divergence"] +
            0.15 * features["HAB_shell_entropy"] / 1.5 +
            0.10 * features["HAB_shell_mixing"] +
            0.10 * features["HAB_global_mixing"] +
            0.05 * np.clip(features["HAB_fragmentation"] / 3, 0, 1),
            0, 1
        ))
        return features
