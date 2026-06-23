"""
Advanced clustering methods for dental instance segmentation.
Supports DBSCAN, HDBSCAN, MeanShift, and spatial-aware clustering.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any


def dbscan_cluster(
    embeddings: np.ndarray,
    eps: float = 0.7,
    min_samples: int = 300,
    spatial_coords: Optional[np.ndarray] = None,
    spatial_weight: float = 0.3
) -> np.ndarray:
    """
    DBSCAN clustering with optional spatial weighting.

    Args:
        embeddings: (N, D) normalized embeddings
        eps: Neighborhood radius
        min_samples: Minimum samples for core point
        spatial_coords: (N, 3) optional spatial coordinates to include
        spatial_weight: Weight for spatial coordinates

    Returns:
        labels: (N,) cluster labels (-1 for noise)
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("sklearn is required for DBSCAN clustering")

    # Standardize embeddings
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    # Optionally include spatial coordinates
    if spatial_coords is not None and spatial_weight > 0:
        # Standardize spatial coordinates
        spatial_scaled = scaler.fit_transform(spatial_coords)
        features = np.concatenate([
            emb_scaled,
            spatial_scaled * spatial_weight
        ], axis=-1)
    else:
        features = emb_scaled

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
    return labels.astype(np.int32)


def hdbscan_cluster(
    embeddings: np.ndarray,
    min_cluster_size: int = 200,
    min_samples: int = 50,
    spatial_coords: Optional[np.ndarray] = None,
    spatial_weight: float = 0.3
) -> np.ndarray:
    """
    HDBSCAN clustering - better for varying density clusters.

    Args:
        embeddings: (N, D) normalized embeddings
        min_cluster_size: Minimum size for a cluster
        min_samples: Minimum samples for core point
        spatial_coords: (N, 3) optional spatial coordinates
        spatial_weight: Weight for spatial coordinates

    Returns:
        labels: (N,) cluster labels (-1 for noise)
    """
    try:
        import hdbscan
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[WARN] hdbscan not installed, falling back to DBSCAN")
        return dbscan_cluster(embeddings, spatial_coords=spatial_coords, spatial_weight=spatial_weight)

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    if spatial_coords is not None and spatial_weight > 0:
        spatial_scaled = scaler.fit_transform(spatial_coords)
        features = np.concatenate([
            emb_scaled,
            spatial_scaled * spatial_weight
        ], axis=-1)
    else:
        features = emb_scaled

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'  # Excess of Mass
    )
    labels = clusterer.fit_predict(features)
    return labels.astype(np.int32)


def meanshift_cluster(
    embeddings: np.ndarray,
    bandwidth: Optional[float] = None,
    spatial_coords: Optional[np.ndarray] = None,
    spatial_weight: float = 0.3
) -> np.ndarray:
    """
    MeanShift clustering - finds cluster centers via kernel density estimation.

    Args:
        embeddings: (N, D) normalized embeddings
        bandwidth: Kernel bandwidth (None = auto-estimate)
        spatial_coords: (N, 3) optional spatial coordinates
        spatial_weight: Weight for spatial coordinates

    Returns:
        labels: (N,) cluster labels
    """
    try:
        from sklearn.cluster import MeanShift, estimate_bandwidth
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("sklearn is required for MeanShift clustering")

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    if spatial_coords is not None and spatial_weight > 0:
        spatial_scaled = scaler.fit_transform(spatial_coords)
        features = np.concatenate([
            emb_scaled,
            spatial_scaled * spatial_weight
        ], axis=-1)
    else:
        features = emb_scaled

    if bandwidth is None:
        bandwidth = estimate_bandwidth(features, quantile=0.2)
        if bandwidth <= 0:
            bandwidth = 1.0

    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = clusterer.fit_predict(features)
    return labels.astype(np.int32)


def spectral_cluster(
    embeddings: np.ndarray,
    n_clusters: int = 14,
    spatial_coords: Optional[np.ndarray] = None,
    spatial_weight: float = 0.3
) -> np.ndarray:
    """
    Spectral clustering - good for non-convex clusters.

    Args:
        embeddings: (N, D) normalized embeddings
        n_clusters: Number of clusters (typically 14 teeth per jaw)
        spatial_coords: (N, 3) optional spatial coordinates
        spatial_weight: Weight for spatial coordinates

    Returns:
        labels: (N,) cluster labels
    """
    try:
        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("sklearn is required for Spectral clustering")

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    if spatial_coords is not None and spatial_weight > 0:
        spatial_scaled = scaler.fit_transform(spatial_coords)
        features = np.concatenate([
            emb_scaled,
            spatial_scaled * spatial_weight
        ], axis=-1)
    else:
        features = emb_scaled

    # Subsample if too many points (spectral is expensive)
    N = features.shape[0]
    max_samples = 10000
    if N > max_samples:
        indices = np.random.choice(N, max_samples, replace=False)
        features_sub = features[indices]
    else:
        indices = None
        features_sub = features

    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels='kmeans',
        affinity='nearest_neighbors',
        n_neighbors=30
    )
    labels_sub = clusterer.fit_predict(features_sub)

    # If subsampled, propagate labels to all points using nearest neighbors
    if indices is not None:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features_sub, labels_sub)
        labels = knn.predict(features).astype(np.int32)
    else:
        labels = labels_sub.astype(np.int32)

    return labels


def spatial_meanshift_cluster(
    points: np.ndarray,
    bandwidth: float = 4.0,
    jaw: Optional[str] = None,
) -> np.ndarray:
    """
    MeanShift clustering using only spatial coordinates (XYZ).

    This method achieved Instance Dice = 0.63-0.68 on dental data,
    outperforming embedding-based approaches.

    Per-jaw optimized bandwidth values (2026/02/14):
        - Upper jaw: 3.5 (Dice=0.6546)
        - Lower jaw: 4.5 (Dice=0.6149)
        - Default: 4.0

    Args:
        points: (N, 3) XYZ coordinates of tooth points
        bandwidth: MeanShift kernel bandwidth (default=4.0)
        jaw: Optional jaw type ('upper' or 'lower') for per-jaw optimization

    Returns:
        labels: (N,) cluster labels (0 to num_clusters-1)
    """
    try:
        from sklearn.cluster import MeanShift
    except ImportError:
        raise ImportError("sklearn is required for MeanShift clustering")

    # Per-jaw bandwidth optimization
    if jaw is not None:
        jaw_lower = jaw.lower()
        if 'upper' in jaw_lower:
            bandwidth = 3.5
        elif 'lower' in jaw_lower:
            bandwidth = 4.5

    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = clusterer.fit_predict(points)
    return labels.astype(np.int32)


def split_large_spatial(
    points: np.ndarray,
    labels: np.ndarray,
    size_threshold: int = 4500,
    sub_bandwidth: float = 3.4,
) -> np.ndarray:
    """
    Split oversized clusters using MeanShift with smaller bandwidth.

    Effective for lower jaw where MeanShift(bw=4.5) tends to merge
    adjacent teeth into single clusters.

    Args:
        points: (N, 3) XYZ coordinates
        labels: (N,) cluster labels from initial MeanShift
        size_threshold: clusters larger than this are split candidates
        sub_bandwidth: bandwidth for sub-clustering (smaller = more splits)

    Returns:
        labels: (N,) updated cluster labels
    """
    try:
        from sklearn.cluster import MeanShift
    except ImportError:
        return labels

    new_labels = labels.copy()
    next_id = int(labels.max()) + 1
    for cid in np.unique(labels):
        if cid < 0:
            continue
        mask = labels == cid
        if mask.sum() <= size_threshold:
            continue
        sub_ms = MeanShift(bandwidth=sub_bandwidth, bin_seeding=True)
        sub_labels = sub_ms.fit_predict(points[mask])
        if len(np.unique(sub_labels)) > 1:
            idx = np.where(mask)[0]
            for sub_id in np.unique(sub_labels):
                new_labels[idx[sub_labels == sub_id]] = next_id
                next_id += 1
    return new_labels.astype(np.int32)


def merge_nearby_clusters(
    points: np.ndarray,
    labels: np.ndarray,
    min_size: int = 3500,
    max_merge_dist: float = 5.5,
) -> np.ndarray:
    """
    Merge small clusters into their nearest neighbor cluster.

    Effective for upper jaw where MeanShift(bw=3.5) tends to over-segment
    teeth into multiple fragments.

    Args:
        points: (N, 3) XYZ coordinates
        labels: (N,) cluster labels
        min_size: clusters smaller than this are merge candidates
        max_merge_dist: maximum centroid distance for merging

    Returns:
        labels: (N,) updated cluster labels (relabeled consecutively)
    """
    new_labels = labels.copy()
    unique_ids = [int(i) for i in np.unique(labels) if i >= 0]
    if len(unique_ids) <= 1:
        return new_labels

    centroids = {}
    sizes = {}
    for cid in unique_ids:
        mask = labels == cid
        centroids[cid] = points[mask].mean(axis=0)
        sizes[cid] = mask.sum()

    for small_id in sorted(
        [c for c in unique_ids if sizes[c] < min_size], key=lambda x: sizes[x]
    ):
        if sizes.get(small_id, 0) == 0:
            continue
        sc = centroids[small_id]
        best_target = -1
        best_dist = float('inf')
        for other_id in unique_ids:
            if other_id == small_id or sizes.get(other_id, 0) == 0:
                continue
            dist = np.linalg.norm(centroids[other_id] - sc)
            if dist < max_merge_dist and dist < best_dist:
                best_dist = dist
                best_target = other_id
        if best_target >= 0:
            new_labels[new_labels == small_id] = best_target
            target_mask = new_labels == best_target
            centroids[best_target] = points[target_mask].mean(axis=0)
            sizes[best_target] = target_mask.sum()
            sizes[small_id] = 0

    # Relabel consecutively
    final = -np.ones_like(new_labels)
    for i, cid in enumerate(
        sorted(set(int(x) for x in np.unique(new_labels) if x >= 0))
    ):
        final[new_labels == cid] = i
    return final.astype(np.int32)


def center_voting_cluster(
    embeddings: np.ndarray,
    center_offsets: np.ndarray,
    points: np.ndarray,
    bandwidth: float = 2.0,
    min_votes: int = 100
) -> np.ndarray:
    """
    Center voting based clustering.
    Each point votes for its predicted center, then clusters are formed around center concentrations.

    Args:
        embeddings: (N, D) embeddings (not used directly, but kept for interface consistency)
        center_offsets: (N, 3) predicted offset to cluster center
        points: (N, 3) point coordinates
        bandwidth: Bandwidth for center density estimation
        min_votes: Minimum votes to be a valid center

    Returns:
        labels: (N,) cluster labels (-1 for noise)
    """
    try:
        from sklearn.cluster import MeanShift
    except ImportError:
        raise ImportError("sklearn is required for center voting clustering")

    # Compute voted centers
    voted_centers = points + center_offsets

    # Cluster the voted centers
    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    center_labels = clusterer.fit_predict(voted_centers)

    # Filter out small clusters
    unique, counts = np.unique(center_labels, return_counts=True)
    valid_clusters = unique[counts >= min_votes]

    # Relabel
    labels = center_labels.copy()
    for i, cluster_id in enumerate(valid_clusters):
        labels[center_labels == cluster_id] = i
    labels[~np.isin(center_labels, valid_clusters)] = -1

    return labels.astype(np.int32)


def refine_clusters(
    labels: np.ndarray,
    points: np.ndarray,
    min_cluster_size: int = 100,
    max_cluster_size: int = 10000,
    merge_distance: float = 3.0
) -> np.ndarray:
    """
    Post-process clusters: merge small fragments, split large clusters.

    Args:
        labels: (N,) initial cluster labels
        points: (N, 3) point coordinates
        min_cluster_size: Clusters smaller than this are merged or marked noise
        max_cluster_size: Clusters larger than this are split
        merge_distance: Distance threshold for merging small clusters

    Returns:
        labels: (N,) refined cluster labels
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    # Compute cluster centroids and sizes
    centroids = {}
    sizes = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = points[mask].mean(axis=0)
        sizes[label] = mask.sum()

    new_labels = labels.copy()
    next_label = labels.max() + 1

    # Merge small clusters
    small_clusters = [l for l in unique_labels if sizes[l] < min_cluster_size]
    for small_label in small_clusters:
        if small_label not in centroids:
            continue
        small_centroid = centroids[small_label]

        # Find nearest large enough cluster
        best_target = -1
        best_dist = float('inf')
        for other_label in unique_labels:
            if other_label == small_label:
                continue
            if sizes.get(other_label, 0) < min_cluster_size:
                continue
            dist = np.linalg.norm(centroids[other_label] - small_centroid)
            if dist < merge_distance and dist < best_dist:
                best_dist = dist
                best_target = other_label

        if best_target >= 0:
            new_labels[labels == small_label] = best_target
        else:
            # Mark as noise if no nearby cluster
            new_labels[labels == small_label] = -1

    # Split large clusters (using sub-clustering)
    large_clusters = [l for l in unique_labels if sizes.get(l, 0) > max_cluster_size]
    for large_label in large_clusters:
        mask = new_labels == large_label
        if mask.sum() <= max_cluster_size:
            continue

        # Sub-cluster using DBSCAN with tighter parameters
        sub_points = points[mask]
        try:
            from sklearn.cluster import DBSCAN
            sub_labels = DBSCAN(eps=1.0, min_samples=50).fit_predict(sub_points)
            sub_unique = np.unique(sub_labels)
            sub_unique = sub_unique[sub_unique >= 0]

            # Assign new labels
            for sub_label in sub_unique:
                sub_mask = sub_labels == sub_label
                full_mask = np.zeros(len(labels), dtype=bool)
                full_mask[np.where(mask)[0][sub_mask]] = True
                new_labels[full_mask] = next_label
                next_label += 1
        except ImportError:
            pass  # Keep original label if sklearn not available

    # Relabel to consecutive integers
    final_labels = -np.ones_like(new_labels)
    unique_new = np.unique(new_labels)
    unique_new = unique_new[unique_new >= 0]
    for i, label in enumerate(unique_new):
        final_labels[new_labels == label] = i

    return final_labels.astype(np.int32)


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = 'spatial_meanshift',
    spatial_coords: Optional[np.ndarray] = None,
    center_offsets: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Main entry point for clustering embeddings.

    Args:
        embeddings: (N, D) learned embeddings (ignored for spatial_meanshift)
        method: Clustering method:
            - 'spatial_meanshift': XYZ-only MeanShift (RECOMMENDED, Dice=0.68)
            - 'dbscan': DBSCAN on embeddings
            - 'hdbscan': HDBSCAN on embeddings
            - 'meanshift': MeanShift on embeddings
            - 'spectral': Spectral clustering
            - 'center_voting': Center voting with offsets
        spatial_coords: (N, 3) point coordinates (required for spatial_meanshift)
        center_offsets: (N, 3) predicted offsets to cluster centers (for center_voting)
        config: Configuration dictionary

    Returns:
        labels: (N,) cluster labels
    """
    config = config or {}
    spatial_weight = float(config.get('spatial_weight', 0.3))

    if method == 'spatial_meanshift':
        if spatial_coords is None:
            raise ValueError("spatial_meanshift requires spatial_coords")
        labels = spatial_meanshift_cluster(
            spatial_coords,
            bandwidth=float(config.get('bandwidth', 4.0))
        )
    elif method == 'dbscan':
        labels = dbscan_cluster(
            embeddings,
            eps=float(config.get('eps', 0.7)),
            min_samples=int(config.get('min_samples', 300)),
            spatial_coords=spatial_coords,
            spatial_weight=spatial_weight
        )
    elif method == 'hdbscan':
        labels = hdbscan_cluster(
            embeddings,
            min_cluster_size=int(config.get('min_cluster_size', 200)),
            min_samples=int(config.get('min_samples', 50)),
            spatial_coords=spatial_coords,
            spatial_weight=spatial_weight
        )
    elif method == 'meanshift':
        labels = meanshift_cluster(
            embeddings,
            bandwidth=config.get('bandwidth'),
            spatial_coords=spatial_coords,
            spatial_weight=spatial_weight
        )
    elif method == 'spectral':
        labels = spectral_cluster(
            embeddings,
            n_clusters=int(config.get('n_clusters', 14)),
            spatial_coords=spatial_coords,
            spatial_weight=spatial_weight
        )
    elif method == 'center_voting':
        if center_offsets is None or spatial_coords is None:
            raise ValueError("center_voting requires center_offsets and spatial_coords")
        labels = center_voting_cluster(
            embeddings,
            center_offsets,
            spatial_coords,
            bandwidth=float(config.get('bandwidth', 2.0)),
            min_votes=int(config.get('min_votes', 100))
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Optional refinement
    if config.get('refine', False) and spatial_coords is not None:
        labels = refine_clusters(
            labels,
            spatial_coords,
            min_cluster_size=int(config.get('min_cluster_size', 100)),
            max_cluster_size=int(config.get('max_cluster_size', 10000)),
            merge_distance=float(config.get('merge_distance', 3.0))
        )

    return labels
