#!/usr/bin/env python3
"""
Unified dataset sampling script that creates three splits:
1. Train: Non-outlier users (users that are more similar to the overall population)
2. Random Test: Random sample of data across all users
3. OOD Test: Outlier users (maximally dissimilar to the overall population)

This combines functionality from random_sample_datasets.py and user_emb_sample_datasets.py
All outputs are saved in v{version}/ directory structure.
"""

import json
import numpy as np
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def _temporary_seed(random_seed: Optional[int]):
    """Temporarily set Python and NumPy random seeds, then restore state."""
    if random_seed is None:
        yield
        return
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

def load_jsonl(file_path):
    """Load JSONL file and return list of objects"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to JSONL file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def group_by_user(data, user_key='user_id'):
    """Group data by user_id"""
    user_data = defaultdict(list)
    for item in data:
        user_data[item[user_key]].append(item)
    return user_data

def get_history_length(item, dataset_name):
    """Get history length based on dataset type"""
    if "PRISM" in dataset_name:
        return len(item.get('conversations', []))
    else:  # LaMP and LongLaMP
        return len(item.get('history', []))

def load_user_embeddings(embedding_file: str) -> Tuple[List[str], np.ndarray]:
    """Load user embeddings from npz file"""
    logger.info(f"Loading embeddings from {embedding_file}")
    data = np.load(embedding_file)
    user_ids = list(data['user_ids'])
    embeddings = data['embeddings']
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    logger.info(f"Loaded embeddings for {len(user_ids)} users, shape: {embeddings.shape}")
    return user_ids, embeddings


def select_diverse_users(embeddings: np.ndarray, user_ids: List[str], n_users: int, 
                        random_seed: Optional[int] = None, method: str = "clustering") -> List[str]:
    """
    Select diverse users based on embeddings to ensure representative sampling.
    
    Args:
        embeddings: User embeddings (normalized)
        user_ids: List of user IDs corresponding to embeddings
        n_users: Number of users to select
        random_seed: Random seed for reproducibility
        method: Selection method ("clustering" or "maximin")
    
    Returns:
        List of selected user IDs
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    n_available = len(user_ids)
    if n_users >= n_available:
        return user_ids.copy()
    
    logger.info(f"Selecting {n_users} diverse users from {n_available} available users using {method} method")
    
    if method == "clustering":
        # Use clustering-based diverse sampling
        # Determine optimal number of clusters (aim for reasonable cluster sizes)
        min_cluster_size = max(1, n_users // 20)  # At least 5% of target users per cluster
        max_clusters = min(50, max(5, n_available // min_cluster_size))
        
        # Use fewer clusters for smaller datasets
        n_clusters = min(max_clusters, max(5, min(20, n_users // 2)))
        n_clusters = min(n_clusters, n_available)
        
        logger.info(f"Using {n_clusters} clusters for diverse sampling")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate how many users to sample from each cluster
        cluster_counts = np.bincount(cluster_labels)
        cluster_proportions = cluster_counts / n_available
        
        # Distribute target users across clusters proportionally
        users_per_cluster = np.round(cluster_proportions * n_users).astype(int)
        
        # Adjust for rounding errors
        total_assigned = users_per_cluster.sum()
        if total_assigned < n_users:
            # Add extra users to largest clusters
            diff = n_users - total_assigned
            largest_clusters = np.argsort(cluster_counts)[-diff:]
            users_per_cluster[largest_clusters] += 1
        elif total_assigned > n_users:
            # Remove users from smallest clusters
            diff = total_assigned - n_users
            smallest_clusters = np.argsort(cluster_counts)[:diff]
            users_per_cluster[smallest_clusters] = np.maximum(0, users_per_cluster[smallest_clusters] - 1)
        
        # Sample users from each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            n_from_cluster = users_per_cluster[cluster_id]
            
            if n_from_cluster > 0 and len(cluster_indices) > 0:
                if n_from_cluster >= len(cluster_indices):
                    # Take all users from this cluster
                    selected_indices.extend(cluster_indices)
                else:
                    # Randomly sample from this cluster
                    sampled = np.random.choice(cluster_indices, size=n_from_cluster, replace=False)
                    selected_indices.extend(sampled)
        
        # Ensure we have exactly n_users (handle any remaining discrepancies)
        if len(selected_indices) < n_users:
            # Add more users from remaining pool
            remaining_indices = set(range(n_available)) - set(selected_indices)
            additional_needed = n_users - len(selected_indices)
            if remaining_indices:
                additional = np.random.choice(list(remaining_indices), 
                                            size=min(additional_needed, len(remaining_indices)), 
                                            replace=False)
                selected_indices.extend(additional)
        elif len(selected_indices) > n_users:
            # Randomly remove excess users
            selected_indices = np.random.choice(selected_indices, size=n_users, replace=False)
        
        selected_user_ids = [user_ids[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_user_ids)} diverse users using clustering-based sampling")
        
    else:  # maximin method
        # Use maximin sampling for maximum diversity
        selected_indices = []
        remaining_indices = list(range(n_available))
        
        # Start with a random user
        first_idx = np.random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select users that are furthest from already selected users
        for _ in range(n_users - 1):
            if not remaining_indices:
                break
                
            # Calculate minimum distances to already selected users
            selected_embeddings = embeddings[selected_indices]
            min_distances = []
            
            for idx in remaining_indices:
                distances = np.linalg.norm(embeddings[idx] - selected_embeddings, axis=1)
                min_distances.append(distances.min())
            
            # Select user with maximum minimum distance
            max_min_idx = np.argmax(min_distances)
            selected_idx = remaining_indices[max_min_idx]
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
        
        selected_user_ids = [user_ids[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_user_ids)} diverse users using maximin sampling")
    
    return selected_user_ids


def select_ood_test_users_clustering(embeddings: np.ndarray, n_train: int, n_test: int, 
                                   user_ids: List[str], random_seed: Optional[int] = None, 
                                   min_test_cluster_size: int = 5) -> Tuple[List[int], List[int]]:
    """
    Select test users using EXTREME clustering-based OOD strategy.
    
    1. Cluster users into groups using K-means
    2. Select ENTIRE outlier clusters for testing (no cluster sharing between train/test)
    3. If cluster is too large, select users furthest from cluster centroid
    4. Ensures complete separation between train and test user types
    
    Args:
        embeddings: User embeddings (normalized)
        n_train: Number of train users
        n_test: Number of test users (restricted)
        user_ids: List of user IDs
        random_seed: Random seed for reproducibility
        min_test_cluster_size: Minimum size for clusters to be considered for test selection
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    n_users = embeddings.shape[0]
    
    if n_train + n_test > n_users:
        # Adjust if we don't have enough users
        logger.warning(f"Requested {n_train} train + {n_test} test = {n_train + n_test} users, but only have {n_users}")
        if n_users <= n_test:
            # Very small dataset, just split in half
            n_test = n_users // 2
            n_train = n_users - n_test
        else:
            # Keep test size, adjust train
            n_test = min(n_test, n_users // 5)  # At most 20% for test
            n_train = n_users - n_test
    
    # Restrict test set size to ensure we don't have too large test sets
    max_test_size = min(n_test, max(50, n_users // 5))  # At most 20% or minimum 50
    n_test = min(n_test, max_test_size)
    n_train = min(n_train, n_users - n_test)
    
    logger.info(f"EXTREME OOD clustering-based selection: {n_train} train and {n_test} test users from {n_users} total users")
    
    # Determine optimal number of clusters
    # Use a more adaptive range based on dataset size for better outlier detection
    # Target: Allow for minority clusters while avoiding overly fragmented clustering
    min_clusters = max(5, min(10, n_users // 100))  # Start with more granular clustering
    max_clusters = min(50, max(20, n_users // 50))  # Allow up to ~2% of users per cluster on average
    
    best_k = min_clusters
    best_score = -1
    
    # Find optimal number of clusters using silhouette score
    # Use a coarser search for efficiency when range is large
    search_step = max(1, (max_clusters - min_clusters) // 10)  # Test at most 20 values
    search_range = list(range(min_clusters, max_clusters + 1, search_step))
    if max_clusters not in search_range:
        search_range.append(max_clusters)  # Always include the max value
    
    logger.info(f"Finding optimal number of clusters (testing {len(search_range)} values from {min_clusters}-{max_clusters})...")
    for k in search_range:
        if k >= n_users:
            break
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=5)  # Reduced n_init for speed
            cluster_labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, cluster_labels)
            logger.info(f"k={k}, silhouette_score={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            logger.warning(f"Error with k={k}: {e}")
            continue
    
    logger.info(f"Using k={best_k} clusters (silhouette score: {best_score:.3f})")
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    
    # Analyze cluster properties
    cluster_info = []
    for cluster_id in range(best_k):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        # Calculate average distance from cluster center
        cluster_embs = embeddings[cluster_indices]
        distances_to_center = np.linalg.norm(cluster_embs - cluster_centers[cluster_id], axis=1)
        avg_distance = distances_to_center.mean()
        
        # Calculate average inter-cluster distance (how far this cluster is from others)
        inter_cluster_distances = []
        for other_id in range(best_k):
            if other_id != cluster_id:
                dist = np.linalg.norm(cluster_centers[cluster_id] - cluster_centers[other_id])
                inter_cluster_distances.append(dist)
        avg_inter_cluster_dist = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'indices': cluster_indices,
            'avg_distance_to_center': avg_distance,
            'avg_inter_cluster_distance': avg_inter_cluster_dist
        })
        
        logger.info(f"Cluster {cluster_id}: {cluster_size} users, "
                   f"avg_dist_to_center={avg_distance:.3f}, "
                   f"avg_inter_cluster_dist={avg_inter_cluster_dist:.3f}")
    
    # Strategy: Select test users from minority clusters and clusters that are far from others
    # Sort clusters by a combination of size (smaller better) and inter-cluster distance (farther better)
    # Prioritize smaller, more isolated clusters for test set
    
    cluster_scores = []
    for info in cluster_info:
        # Score = inter_cluster_distance / (1 + cluster_size)
        # Higher score means better for test set (more isolated and smaller)
        score = info['avg_inter_cluster_distance'] / (1 + info['size'])
        cluster_scores.append((score, info))
    
    # Sort by score (descending - highest scores first)
    cluster_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select test users from top-scoring clusters (EXTREME OOD - no cluster sharing)
    test_indices = []
    train_indices = []
    test_cluster_ids = set()  # Track which clusters are used for testing
    
    for score, info in cluster_scores:
        cluster_indices = info['indices']
        cluster_size = info['size']
        cluster_id = info['cluster_id']
        
        # Determine how many test users to take from this cluster
        if len(test_indices) < n_test and cluster_size >= min_test_cluster_size:
            remaining_test_slots = n_test - len(test_indices)
            
            if cluster_size <= remaining_test_slots:
                # Take ALL users from this cluster for test set (entire cluster goes to test)
                n_test_from_cluster = cluster_size
                cluster_test_indices = cluster_indices
                test_cluster_ids.add(cluster_id)
                
                logger.info(f"Selected ALL {n_test_from_cluster} users from cluster {cluster_id} "
                           f"(score={score:.3f}, size={cluster_size}) - ENTIRE CLUSTER for test")
            else:
                # Cluster is larger than remaining slots - select users furthest from centroid
                n_test_from_cluster = remaining_test_slots
                
                # Calculate distances from cluster centroid for all users in this cluster
                cluster_center = cluster_centers[cluster_id]
                cluster_embs = embeddings[cluster_indices]
                distances_to_center = np.linalg.norm(cluster_embs - cluster_center, axis=1)
                
                # Select users with largest distances (furthest from centroid = most extreme)
                furthest_indices_in_cluster = np.argsort(distances_to_center)[-n_test_from_cluster:]
                cluster_test_indices = cluster_indices[furthest_indices_in_cluster]
                test_cluster_ids.add(cluster_id)
                
                logger.info(f"Selected {n_test_from_cluster} most distant users from cluster {cluster_id} "
                           f"(score={score:.3f}, size={cluster_size}) - furthest from centroid")
            
            test_indices.extend(cluster_test_indices)
            
            # EXTREME OOD: NO users from test clusters go to training
            # (break here since we've filled our test quota)
            if len(test_indices) >= n_test:
                break
    
    # All remaining clusters (not used for testing) go entirely to training
    for score, info in cluster_scores:
        cluster_id = info['cluster_id']
        if cluster_id not in test_cluster_ids:
            train_indices.extend(info['indices'])
    
    # Trim train set if we have too many users
    if len(train_indices) > n_train:
        train_indices = random.sample(train_indices, n_train)
    
    # Convert to lists and ensure we have the right number
    test_indices = test_indices[:n_test]
    train_indices = train_indices[:n_train]
    
    # Calculate OOD statistics
    test_emb = embeddings[test_indices]
    train_emb = embeddings[train_indices]
    test_train_sims = cosine_similarity(test_emb, train_emb)
    
    avg_test_train_sim = test_train_sims.mean()
    max_test_train_sim = test_train_sims.max()
    min_test_train_sim = test_train_sims.min()
    
    logger.info(f"EXTREME OOD Test-Train similarity stats: avg={avg_test_train_sim:.3f}, "
                f"max={max_test_train_sim:.3f}, min={min_test_train_sim:.3f}")
    
    # For comparison, calculate train-train similarity
    if len(train_indices) > 1:
        train_train_sims = cosine_similarity(train_emb, train_emb)
        # Exclude diagonal (self-similarity)
        mask = np.ones(train_train_sims.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        train_train_sims_no_diag = train_train_sims[mask]
        
        avg_train_train_sim = train_train_sims_no_diag.mean()
        logger.info(f"Train-Train similarity avg={avg_train_train_sim:.3f} "
                    f"(test-train is {(avg_test_train_sim/avg_train_train_sim):.1%} of train-train)")
    
    # Report cluster distribution
    test_clusters = [cluster_labels[i] for i in test_indices]
    train_clusters = [cluster_labels[i] for i in train_indices]
    
    test_cluster_counts = defaultdict(int)
    train_cluster_counts = defaultdict(int)
    
    for cluster_id in test_clusters:
        test_cluster_counts[cluster_id] += 1
    for cluster_id in train_clusters:
        train_cluster_counts[cluster_id] += 1
    
    logger.info(f"Test set cluster distribution: {dict(test_cluster_counts)}")
    logger.info(f"Train set cluster distribution: {dict(train_cluster_counts)}")
    
    return train_indices, test_indices

def select_ood_test_users(embeddings: np.ndarray, user_ids: List[str], 
                         n_outliers: int, random_seed: Optional[int] = None) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Select OOD test users using EXTREME clustering-based approach.
    Ensures complete cluster separation between train and test sets.
    Returns: (outlier_user_ids, normal_user_ids, outlier_indices, normal_indices)
    """
    n_users = embeddings.shape[0]
    n_train = n_users - n_outliers
    
    if n_outliers >= n_users:
        logger.warning(f"Requested {n_outliers} outliers, but only have {n_users} users. Using all as outliers.")
        outlier_indices = list(range(n_users))
        normal_indices = []
        return user_ids, [], outlier_indices, normal_indices
    
    logger.info(f"Identifying {n_outliers} EXTREME OOD outlier users from {n_users} total users")
    
    train_indices, test_indices = select_ood_test_users_clustering(
        embeddings, n_train, n_outliers, user_ids, random_seed
    )
    
    outlier_user_ids = [user_ids[i] for i in test_indices]
    normal_user_ids = [user_ids[i] for i in train_indices]
    
    return outlier_user_ids, normal_user_ids, test_indices, train_indices

def _compute_history_length(item: dict, dataset_name: str) -> int:
    try:
        return get_history_length(item, dataset_name)
    except Exception:
        return 0

def _assign_bins_by_quantiles(lengths: List[int], n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Create quantile-based bin edges and bin indices using log1p(length) to handle long tails."""
    if len(lengths) == 0:
        return np.array([]), np.array([])
    arr = np.array(lengths, dtype=float)
    # Use log-space quantiles for long-tailed distributions
    arr_log = np.log1p(arr)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(arr_log, quantiles)
    # Ensure strictly increasing edges to avoid zero-width bins
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    bins = np.digitize(arr_log, edges[1:-1], right=True)
    return edges, bins

def _stratified_sample_by_history(train_data: List[dict], candidates: List[dict], dataset_name: str,
                                  sample_size: int, n_bins: int = 10, random_seed: Optional[int] = None) -> List[dict]:
    """Sample candidates so history-length distribution matches train_data distribution."""
    if not candidates or sample_size <= 0:
        return []
    if not train_data:
        # Fallback to random if no train_data available
        with _temporary_seed(random_seed):
            shuffled = candidates.copy()
            random.shuffle(shuffled)
            return shuffled[:min(sample_size, len(shuffled))]

    # Compute lengths and bins on train set
    train_lengths = [_compute_history_length(x, dataset_name) for x in train_data]
    edges, train_bins = _assign_bins_by_quantiles(train_lengths, n_bins=n_bins)
    if edges.size == 0:
        with _temporary_seed(random_seed):
            shuffled = candidates.copy()
            random.shuffle(shuffled)
            return shuffled[:min(sample_size, len(shuffled))]

    # Bin candidates using same edges (log-space)
    cand_lengths = [_compute_history_length(x, dataset_name) for x in candidates]
    cand_bins = np.digitize(np.log1p(np.array(cand_lengths, dtype=float)), edges[1:-1], right=True)

    # Target allocation per bin based on train distribution
    bin_count = n_bins
    train_counts = np.bincount(train_bins, minlength=bin_count).astype(float)
    total_train = max(1, int(train_counts.sum()))
    proportions = train_counts / total_train

    # Initial rounding
    target = np.round(proportions * sample_size).astype(int)

    # Adjust rounding to match exact sample_size
    diff = sample_size - int(target.sum())
    if diff != 0:
        # Distribute remaining based on largest fractional remainders from proportions*sample_size
        remainders = (proportions * sample_size) - np.floor(proportions * sample_size)
        order = np.argsort(remainders)[::-1] if diff > 0 else np.argsort(remainders)
        for i in order:
            if diff == 0:
                break
            target[i] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1

    # Cap by availability per bin
    cand_indices_by_bin: Dict[int, List[int]] = defaultdict(list)
    for idx, b in enumerate(cand_bins.tolist()):
        cand_indices_by_bin[b].append(idx)

    alloc = target.copy()
    for b in range(bin_count):
        alloc[b] = min(alloc[b], len(cand_indices_by_bin.get(b, [])))

    # If under-allocated due to scarcity, fill from bins with extra availability
    remaining = sample_size - int(alloc.sum())
    if remaining > 0:
        # Calculate extra capacity in each bin
        extra = {b: max(0, len(cand_indices_by_bin.get(b, [])) - alloc[b]) for b in range(bin_count)}
        # Fill greedily from bins with most extra
        for b in sorted(extra.keys(), key=lambda x: extra[x], reverse=True):
            if remaining == 0:
                break
            take = min(remaining, extra[b])
            alloc[b] += take
            remaining -= take

    # Collect sampled indices per bin
    chosen_indices: List[int] = []
    for b in range(bin_count):
        bin_candidates = cand_indices_by_bin.get(b, [])
        if not bin_candidates or alloc[b] <= 0:
            continue
        with _temporary_seed(random_seed):
            selected = random.sample(bin_candidates, alloc[b]) if alloc[b] < len(bin_candidates) else bin_candidates
        chosen_indices.extend(selected)

    # If still short (e.g., not enough candidates overall), fill randomly from leftovers
    if len(chosen_indices) < min(sample_size, len(candidates)):
        remaining_pool = list(set(range(len(candidates))) - set(chosen_indices))
        with _temporary_seed(random_seed):
            random.shuffle(remaining_pool)
        needed = min(sample_size, len(candidates)) - len(chosen_indices)
        chosen_indices.extend(remaining_pool[:needed])

    # Final shuffle for randomness
    with _temporary_seed(random_seed):
        random.shuffle(chosen_indices)
    chosen_indices = chosen_indices[:min(sample_size, len(chosen_indices))]
    return [candidates[i] for i in chosen_indices]

def create_random_test_split(data: List[dict], test_size: int, dataset_name: str, random_seed: Optional[int] = None) -> Tuple[List[dict], List[dict]]:
    """Create stratified train/test split by history length with no overlap."""
    if not data or test_size <= 0:
        return data, []
    test_size = min(test_size, len(data))

    # Compute quantile bins from full data, then sample test per-bin proportionally
    n_bins = 10
    lengths = [_compute_history_length(x, dataset_name) for x in data]
    edges, bins = _assign_bins_by_quantiles(lengths, n_bins=n_bins)
    if edges.size == 0:
        with _temporary_seed(random_seed):
            shuffled_data = data.copy()
            random.shuffle(shuffled_data)
        return shuffled_data[test_size:], shuffled_data[:test_size]

    total = len(data)
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    proportions = counts / max(1, total)
    target = np.round(proportions * test_size).astype(int)
    # Adjust rounding to exact test_size
    diff = test_size - int(target.sum())
    if diff != 0:
        remainders = (proportions * test_size) - np.floor(proportions * test_size)
        order = np.argsort(remainders)[::-1] if diff > 0 else np.argsort(remainders)
        for i in order:
            if diff == 0:
                break
            target[i] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1

    # Map from bin to indices
    idx_by_bin: Dict[int, List[int]] = defaultdict(list)
    for idx, b in enumerate(bins.tolist()):
        idx_by_bin[b].append(idx)

    chosen: List[int] = []
    for b in range(n_bins):
        pool = idx_by_bin.get(b, [])
        need = min(target[b], len(pool))
        if need > 0:
            with _temporary_seed(random_seed):
                chosen.extend(random.sample(pool, need) if need < len(pool) else pool)

    # If short due to scarcity, top up randomly from remaining
    if len(chosen) < test_size:
        remaining_pool = list(set(range(len(data))) - set(chosen))
        with _temporary_seed(random_seed):
            random.shuffle(remaining_pool)
        chosen.extend(remaining_pool[:test_size - len(chosen)])

    chosen = chosen[:test_size]
    chosen_set = set(chosen)
    test_data = [data[i] for i in chosen]
    train_data = [data[i] for i in range(len(data)) if i not in chosen_set]
    return train_data, test_data

def check_data_overlap(train_data: List[dict], random_test: List[dict], ood_test: List[dict]) -> Tuple[int, int, int]:
    """
    Check for overlaps between the three datasets
    Note: Only train vs test overlaps are problematic; test vs test overlaps are allowed
    Returns: (train_random_overlap, train_ood_overlap, random_ood_overlap) counts
    """
    # Create sets of unique identifiers for each dataset
    # Use a combination of multiple fields to create unique identifiers
    def create_id(item):
        # Try to create a unique identifier using available fields
        id_parts = []
        if 'user_id' in item:
            id_parts.append(str(item['user_id']))
        if 'query' in item:
            # For datasets with queries, use first part of query
            if isinstance(item['query'], list) and len(item['query']) > 0:
                if isinstance(item['query'][0], dict) and 'input' in item['query'][0]:
                    id_parts.append(str(item['query'][0]['input'])[:50])  # First 50 chars
            elif isinstance(item['query'], str):
                id_parts.append(str(item['query'])[:50])
            else:
                id_parts.append(str(item['query'])[:50])
        if 'input' in item:
            id_parts.append(str(item['input'])[:50])
        if 'output' in item:
            id_parts.append(str(item['output'])[:50])
        if 'conversations' in item:
            # For PRISM dataset that has conversations
            id_parts.append(str(item['conversations'])[:50])
        # If no good identifier found, use string representation
        if not id_parts:
            id_parts.append(str(item)[:100])
        # Ensure all parts are strings
        id_parts = [str(part) for part in id_parts]
        return '|'.join(id_parts)
    
    train_ids = set(create_id(item) for item in train_data)
    random_test_ids = set(create_id(item) for item in random_test)
    ood_test_ids = set(create_id(item) for item in ood_test)
    
    # Count overlaps
    train_random_overlap = len(train_ids.intersection(random_test_ids))
    train_ood_overlap = len(train_ids.intersection(ood_test_ids))
    random_ood_overlap = len(random_test_ids.intersection(ood_test_ids))
    
    return (train_random_overlap, train_ood_overlap, random_ood_overlap)

def process_dataset_unified(file_path: str, dataset_name: str, embedding_file: Optional[str] = None, version: str = "v3", random_test_seed: Optional[int] = None):
    """Process a dataset with unified sampling creating train, random_test, and ood_test splits"""
    
    print(f"\n=== Processing {dataset_name} with unified sampling ===")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
        
    # Load data
    print(f"Loading data from {file_path}")
    data = load_jsonl(file_path)
    print(f"Total records: {len(data)}")
    
    if len(data) == 0:
        print("No data found, skipping...")
        return None
    
    # Check if we have embeddings for OOD splits
    has_embeddings = embedding_file and os.path.exists(embedding_file)
    if embedding_file and not has_embeddings:
        print(f"Embedding file not found: {embedding_file}, will skip OOD test splits")
    
    # Special handling for datasets that don't use user-based splitting
    if dataset_name in ["LaMP Citation", "LaMP Tweet", "LaMP Scholarly Title", "LaMP Product", 
                       "LongLaMP Abstract Generation", "LongLaMP Product Review", "LongLaMP Topic Writing"]:
        # For these datasets, ensure no overlap between splits
        print(f"Using non-overlapping splits for {dataset_name}")
        
        if has_embeddings:
            user_data = group_by_user(data)
            emb_user_ids, embeddings = load_user_embeddings(embedding_file)
            emb_user_id_to_idx = {uid: idx for idx, uid in enumerate(emb_user_ids)}
            
            # Filter to users with embeddings
            user_data = {uid: items for uid, items in user_data.items() if uid in emb_user_id_to_idx}
            dataset_user_ids = list(user_data.keys())
            
            if dataset_user_ids:
                # Get embeddings for these users
                dataset_user_indices = [emb_user_id_to_idx[uid] for uid in dataset_user_ids]
                dataset_embeddings = embeddings[dataset_user_indices]
                
                # Determine OOD test user count following user_emb_sample_datasets.py logic
                # For LaMP/LongLaMP, use 200 test users (or less if not enough users)
                n_test_users = min(200, len(dataset_user_ids))
                # If we have less than 200 users total, use 4:1 ratio
                if len(dataset_user_ids) < 200:
                    n_test_users = max(1, (len(dataset_user_ids)*2)//10)
                
                if n_test_users > 0:
                    # Create clustering-based OOD test (seeded for reproducibility)
                    outlier_user_ids, remaining_user_ids, _, _ = select_ood_test_users(
                        dataset_embeddings, dataset_user_ids, n_test_users, random_seed=random_test_seed
                    )
                    
                    # Create OOD test from outlier users
                    ood_test = []
                    for uid in outlier_user_ids:
                        ood_test.extend(user_data[uid])
                    
                    # Limit OOD test to 200 samples (seeded shuffle)
                    with _temporary_seed(random_test_seed):
                        random.shuffle(ood_test)
                    ood_test = ood_test[:200]
                    
                    # From remaining users, create user-level random test and train splits
                    # Split users (not samples) to ensure no user overlap between train and random test
                    # Use diverse sampling for random test users
                    n_random_users = min(200, len(remaining_user_ids))
                    if n_random_users > 0:
                        # Get embeddings for remaining users for diverse selection
                        remaining_user_indices = [emb_user_id_to_idx[uid] for uid in remaining_user_ids]
                        remaining_embeddings = embeddings[remaining_user_indices]
                        
                        # Select diverse random test users (seed applies only to random_test)
                        with _temporary_seed(random_test_seed):
                            random_test_users = select_diverse_users(
                                remaining_embeddings, remaining_user_ids, n_random_users, random_seed=random_test_seed
                            )
                        train_users = [uid for uid in remaining_user_ids if uid not in random_test_users]
                    else:
                        random_test_users = []
                        train_users = remaining_user_ids.copy()
                    
                    # Create random test and train data from these user splits
                    random_test = []
                    for uid in random_test_users:
                        random_test.extend(user_data[uid])
                    
                    train_data = []
                    for uid in train_users:
                        train_data.extend(user_data[uid])
                    
                    # Sample random_test to match train history-length distribution
                    random_test = _stratified_sample_by_history(train_data, random_test, dataset_name, 200, random_seed=random_test_seed)
                    
                else:
                    # No outliers, create user-level random split with diverse sampling
                    all_users = list(user_data.keys())
                    # Filter to users with embeddings for diverse selection
                    users_with_emb = [uid for uid in all_users if uid in emb_user_id_to_idx]
                    
                    n_random_users = min(200, len(users_with_emb) if users_with_emb else len(all_users))
                    if users_with_emb and n_random_users > 0:
                        # Get embeddings for diverse selection
                        user_indices = [emb_user_id_to_idx[uid] for uid in users_with_emb]
                        user_embeddings = embeddings[user_indices]
                        
                        # Select diverse random test users
                        with _temporary_seed(random_test_seed):
                            random_test_users = select_diverse_users(
                                user_embeddings, users_with_emb, n_random_users, random_seed=random_test_seed
                            )
                        train_users = [uid for uid in all_users if uid not in random_test_users]
                    else:
                        # Fall back to random selection
                        with _temporary_seed(random_test_seed):
                            random.shuffle(all_users)
                            random_test_users = all_users[:200]
                        train_users = all_users[200:]
                    
                    random_test = []
                    for uid in random_test_users:
                        random_test.extend(user_data[uid])
                    # Sample random_test to match train history-length distribution
                    random_test = _stratified_sample_by_history(train_data, random_test, dataset_name, 200, random_seed=random_test_seed)
                    
                    train_data = []
                    for uid in train_users:
                        train_data.extend(user_data[uid])
                    
                    ood_test = []
            else:
                # No users with embeddings, fall back to sample-level split
                train_data, random_test = create_random_test_split(data, 200, dataset_name, random_seed=random_test_seed)
                ood_test = []
        else:
            # No embeddings, fall back to sample-level split
            train_data, random_test = create_random_test_split(data, 200, dataset_name, random_seed=random_test_seed)
            ood_test = []
        
        # Stats
        train_users = len(set(item.get('user_id', 'unknown') for item in train_data)) if train_data else 0
        random_test_users = len(set(item.get('user_id', 'unknown') for item in random_test)) if random_test else 0
        ood_test_users = len(set(item.get('user_id', 'unknown') for item in ood_test)) if ood_test else 0
        
    elif "OpinionQA" in dataset_name:
        # Special case for OpinionQA: ensure user diversity with no overlap
        user_data = group_by_user(data)
        users = list(user_data.keys())
        total_users = len(users)
        
        print(f"Total unique users: {total_users}")
        
        if has_embeddings:
            emb_user_ids, embeddings = load_user_embeddings(embedding_file)
            emb_user_id_to_idx = {uid: idx for idx, uid in enumerate(emb_user_ids)}
            
            # Filter to users with embeddings
            user_data = {uid: items for uid, items in user_data.items() if uid in emb_user_id_to_idx}
            dataset_user_ids = list(user_data.keys())
            
            if dataset_user_ids:
                # Calculate total test users following same logic as standard approach
                total_test_users = 200  # Fixed for OpinionQA
                
                # Split total test users between random_test and ood test
                n_random_test_users = total_test_users // 2
                n_ood_test_users = total_test_users - n_random_test_users
                
                print(f"Total test users: {total_test_users} (Random: {n_random_test_users}, OOD: {n_ood_test_users})")
                
                # Get embeddings for diverse random test user selection
                dataset_user_indices = [emb_user_id_to_idx[uid] for uid in dataset_user_ids]
                dataset_embeddings = embeddings[dataset_user_indices]
                
                # Select diverse random test users using embeddings
                with _temporary_seed(random_test_seed):
                    random_test_users_list = select_diverse_users(
                        dataset_embeddings, dataset_user_ids, n_random_test_users, random_seed=random_test_seed
                    )
                remaining_users_after_random = [uid for uid in dataset_user_ids if uid not in random_test_users_list]
                
                # Create random test candidate pool
                random_test_candidates = []
                for uid in random_test_users_list:
                    random_test_candidates.extend(user_data[uid])
                
                # Get embeddings for remaining users for OOD selection
                remaining_user_indices = [emb_user_id_to_idx[uid] for uid in remaining_users_after_random]
                remaining_embeddings = embeddings[remaining_user_indices]
                
                # Identify clustering-based outlier users from remaining users (seeded)
                actual_n_ood_users = min(n_ood_test_users, len(remaining_users_after_random))
                outlier_user_ids, normal_user_ids, _, _ = select_ood_test_users(
                    remaining_embeddings, remaining_users_after_random, actual_n_ood_users, random_seed=random_test_seed
                )
                
                # Create OOD test from outlier users
                ood_test = []
                for uid in outlier_user_ids:
                    ood_test.extend(user_data[uid])
                
                # Shuffle and limit OOD test to 200 samples (seeded)
                with _temporary_seed(random_test_seed):
                    random.shuffle(ood_test)
                ood_test = ood_test[:200]
                
                # Create train data from normal users (not outliers, not random test users)
                train_data = []
                for uid in normal_user_ids:
                    train_data.extend(user_data[uid])
                
                # Shuffle and limit train data to 10000 samples
                random.shuffle(train_data)
                train_data = train_data[:10000]

                # Now stratify random_test from candidate pool to match train history-length distribution
                random_test = _stratified_sample_by_history(train_data, random_test_candidates, dataset_name, 200, random_seed=random_test_seed)
                
            else:
                # No users with embeddings, fall back to sample-level split
                train_data, random_test = create_random_test_split(data, 200, dataset_name, random_seed=random_test_seed)
                train_data = train_data[:10000]  # Limit train to 10000
                ood_test = []
        else:
            # No embeddings, fall back to sample-level split  
            train_data, random_test = create_random_test_split(data, 200, dataset_name, random_seed=random_test_seed)
            train_data = train_data[:10000]  # Limit train to 10000
            ood_test = []
        
        train_users = len(set(item['user_id'] for item in train_data)) if train_data else 0
        random_test_users = len(set(item['user_id'] for item in random_test)) if random_test else 0
        ood_test_users = len(set(item['user_id'] for item in ood_test)) if ood_test else 0

    elif dataset_name == "ALOE":
        # ALOE: 100 users in random_test, 100 users in ood_test, others for training
        user_data = group_by_user(data)
        users = list(user_data.keys())
        total_users = len(users)
        print(f"Total unique users: {total_users}")

        n_random_test_users = min(100, total_users)
        # Ensure we don't request more OOD users than remain after random_test selection
        n_ood_test_users = min(100, max(0, total_users - n_random_test_users))

        if has_embeddings:
            emb_user_ids, embeddings = load_user_embeddings(embedding_file)
            emb_user_id_to_idx = {uid: idx for idx, uid in enumerate(emb_user_ids)}

            # Filter to users with embeddings
            users_with_emb = [uid for uid in users if uid in emb_user_id_to_idx]

            # Random test selection (prefer diverse via embeddings)
            if users_with_emb:
                # Select up to n_random_test_users using diverse sampling
                user_indices = [emb_user_id_to_idx[uid] for uid in users_with_emb]
                user_embeddings = embeddings[user_indices]
                with _temporary_seed(random_test_seed):
                    random_test_users_list = select_diverse_users(
                        user_embeddings, users_with_emb, n_random_test_users, random_seed=random_test_seed
                    )
            else:
                # Fallback to random selection
                with _temporary_seed(random_test_seed):
                    random.shuffle(users)
                    random_test_users_list = users[:n_random_test_users]

            remaining_users_after_random = [uid for uid in users if uid not in random_test_users_list]

            # OOD selection from remaining users (prefer clustering-based outliers)
            remaining_users_with_emb = [uid for uid in remaining_users_after_random if uid in emb_user_id_to_idx]
            if remaining_users_with_emb and n_ood_test_users > 0:
                remaining_user_indices = [emb_user_id_to_idx[uid] for uid in remaining_users_with_emb]
                remaining_embeddings = embeddings[remaining_user_indices]
                actual_n_ood = min(n_ood_test_users, len(remaining_users_with_emb))
                outlier_user_ids, normal_user_ids, _, _ = select_ood_test_users(
                    remaining_embeddings, remaining_users_with_emb, actual_n_ood, random_seed=random_test_seed
                )
                # Build splits
                random_test = []
                for uid in random_test_users_list:
                    random_test.extend(user_data[uid])
                ood_test = []
                for uid in outlier_user_ids:
                    ood_test.extend(user_data[uid])
                train_data = []
                for uid in normal_user_ids:
                    train_data.extend(user_data[uid])
            else:
                # No embeddings available for OOD selection; split users randomly
                with _temporary_seed(random_test_seed):
                    random.shuffle(remaining_users_after_random)
                ood_test_users_list = remaining_users_after_random[:n_ood_test_users]
                train_users_list = remaining_users_after_random[n_ood_test_users:]
                random_test = []
                for uid in random_test_users_list:
                    random_test.extend(user_data[uid])
                ood_test = []
                for uid in ood_test_users_list:
                    ood_test.extend(user_data[uid])
                train_data = []
                for uid in train_users_list:
                    train_data.extend(user_data[uid])
        else:
            # No embeddings; simple random split by users
            with _temporary_seed(random_test_seed):
                random.shuffle(users)
            random_test_users_list = users[:n_random_test_users]
            remaining_users_after_random = users[n_random_test_users:]
            ood_test_users_list = remaining_users_after_random[:n_ood_test_users]
            train_users_list = remaining_users_after_random[n_ood_test_users:]

            random_test = []
            for uid in random_test_users_list:
                random_test.extend(user_data[uid])
            ood_test = []
            for uid in ood_test_users_list:
                ood_test.extend(user_data[uid])
            train_data = []
            for uid in train_users_list:
                train_data.extend(user_data[uid])

        train_users = len(set(item.get('user_id', 'unknown') for item in train_data)) if train_data else 0
        random_test_users = len(set(item.get('user_id', 'unknown') for item in random_test)) if random_test else 0
        ood_test_users = len(set(item.get('user_id', 'unknown') for item in ood_test)) if ood_test else 0
        
    else:
        # Standard user-based splitting (EC, PersonalReddit, PRISM, etc.)
        user_data = group_by_user(data)
        users = list(user_data.keys())
        total_users = len(users)
        
        print(f"Total unique users: {total_users}")
        
        # Special case for EC: sample 10 users for random_test and 10 for ood_test; others train
        if dataset_name == "EC":
            print("EC dataset: 10 users random_test, 10 users ood_test, rest train")
            
            # Fixed target counts, but cap by availability
            total_test_users = min(100, total_users)
            n_random_test_users = min(10, total_test_users // 2)
            n_ood_test_users = min(10, total_test_users - n_random_test_users)
            print(f"Total test users: {total_test_users} (Random: {n_random_test_users}, OOD: {n_ood_test_users})")
            
            # Select random_test users (prefer diverse selection using embeddings)
            if has_embeddings:
                emb_user_ids, embeddings = load_user_embeddings(embedding_file)
                emb_user_id_to_idx = {uid: idx for idx, uid in enumerate(emb_user_ids)}
                users_with_emb = [uid for uid in users if uid in emb_user_id_to_idx]
                if len(users_with_emb) >= n_random_test_users and n_random_test_users > 0:
                    user_indices = [emb_user_id_to_idx[uid] for uid in users_with_emb]
                    user_embeddings = embeddings[user_indices]
                    with _temporary_seed(random_test_seed):
                        random_test_users_list = select_diverse_users(
                            user_embeddings, users_with_emb, n_random_test_users, random_seed=random_test_seed
                        )
                else:
                    with _temporary_seed(random_test_seed):
                        random.shuffle(users)
                    random_test_users_list = users[:n_random_test_users]
            else:
                with _temporary_seed(random_test_seed):
                    random.shuffle(users)
                random_test_users_list = users[:n_random_test_users]
            
            # Remaining users after random_test selection
            remaining_users_after_random = [uid for uid in users if uid not in set(random_test_users_list)]
            
            # Select OOD users (prefer clustering-based outliers using embeddings), then top-up randomly to reach target
            ood_test_users_list: List[str] = []
            if has_embeddings and n_ood_test_users > 0:
                # Use embeddings if available for OOD selection
                if 'emb_user_ids' not in locals():
                    emb_user_ids, embeddings = load_user_embeddings(embedding_file)
                    emb_user_id_to_idx = {uid: idx for idx, uid in enumerate(emb_user_ids)}
                remaining_users_with_emb = [uid for uid in remaining_users_after_random if uid in emb_user_id_to_idx]
                if len(remaining_users_with_emb) > 0:
                    remaining_user_indices = [emb_user_id_to_idx[uid] for uid in remaining_users_with_emb]
                    remaining_embeddings = embeddings[remaining_user_indices]
                    actual_n_ood = min(n_ood_test_users, len(remaining_users_with_emb))
                    outlier_user_ids, _normal_user_ids, _, _ = select_ood_test_users(
                        remaining_embeddings, remaining_users_with_emb, actual_n_ood, random_seed=random_test_seed
                    )
                    ood_test_users_list = list(outlier_user_ids)
            
            # Top up OOD users randomly if needed (no embeddings or not enough outliers found)
            if n_ood_test_users > len(ood_test_users_list):
                need = n_ood_test_users - len(ood_test_users_list)
                pool = [uid for uid in remaining_users_after_random if uid not in set(ood_test_users_list)]
                with _temporary_seed(random_test_seed):
                    random.shuffle(pool)
                ood_test_users_list.extend(pool[:need])
            
            # Build splits
            random_test = []
            for uid in random_test_users_list:
                random_test.extend(user_data[uid])
            ood_test = []
            for uid in ood_test_users_list:
                ood_test.extend(user_data[uid])
            train_user_set = set(users) - set(random_test_users_list) - set(ood_test_users_list)
            train_data = []
            for uid in train_user_set:
                train_data.extend(user_data[uid])
            
            # Stats
            train_users = len(train_user_set)
            random_test_users = len(set(random_test_users_list))
            ood_test_users = len(set(ood_test_users_list))
            
        else:
            # Calculate total test users using random_sample_datasets.py logic
            if dataset_name in ["PersonalReddit"]:
                # Use 4:1 ratio for train:test (same as random_sample_datasets.py)
                total_test_users = max(1, (total_users * 2) // 10)  # 20% for testing
            else:
                # For other datasets, follow sample_users_even_distribution logic
                target_train_users = 1000
                target_test_users = 200
                if total_users < (target_train_users + target_test_users):
                    # Use 4:1 ratio for smaller datasets
                    total_test_users = max(1, (total_users * 2)// 10)  # 20% for testing
                else:
                    total_test_users = target_test_users  # 200 test users
        
            # Split total test users between random_test and ood test
            n_random_test_users = total_test_users // 2
            n_ood_test_users = total_test_users - n_random_test_users
            
            print(f"Total test users: {total_test_users} (Random: {n_random_test_users}, OOD: {n_ood_test_users})")
            
            # First try to select diverse random test users using embeddings if available
            if has_embeddings:
                emb_user_ids, embeddings = load_user_embeddings(embedding_file)
                emb_user_id_to_idx = {uid: idx for idx, uid in enumerate(emb_user_ids)}
                
                # Filter to users with embeddings for random test selection
                users_with_emb = [uid for uid in users if uid in emb_user_id_to_idx]
                
                if len(users_with_emb) >= n_random_test_users:
                    # Get embeddings for diverse random test user selection
                    user_indices = [emb_user_id_to_idx[uid] for uid in users_with_emb]
                    user_embeddings = embeddings[user_indices]
                    
                    # Select diverse random test users using embeddings
                    with _temporary_seed(random_test_seed):
                        random_test_users_list = select_diverse_users(
                            user_embeddings, users_with_emb, n_random_test_users, random_seed=random_test_seed
                        )
                    remaining_users_after_random = [uid for uid in users if uid not in random_test_users_list]
                else:
                    # Fall back to random selection if not enough users have embeddings
                    with _temporary_seed(random_test_seed):
                        random.shuffle(users)
                        random_test_users_list = users[:n_random_test_users]
                    remaining_users_after_random = users[n_random_test_users:]
            else:
                # Fall back to random selection if no embeddings
                with _temporary_seed(random_test_seed):
                    random.shuffle(users)
                    random_test_users_list = users[:n_random_test_users]
                remaining_users_after_random = users[n_random_test_users:]
            
            # Create random test data
            random_test = []
            for uid in random_test_users_list:
                random_test.extend(user_data[uid])
            
            if has_embeddings:
                # Only load embeddings if not already loaded for random test selection
                if 'emb_user_ids' not in locals():
                    emb_user_ids, embeddings = load_user_embeddings(embedding_file)
                    emb_user_id_to_idx = {uid: idx for idx, uid in enumerate(emb_user_ids)}
                
                # Filter remaining users to those with embeddings
                remaining_users_with_emb = [uid for uid in remaining_users_after_random if uid in emb_user_id_to_idx]
                
                if remaining_users_with_emb and n_ood_test_users > 0:
                    # Get embeddings for remaining users
                    remaining_user_indices = [emb_user_id_to_idx[uid] for uid in remaining_users_with_emb]
                    remaining_embeddings = embeddings[remaining_user_indices]
                    
                    # Use the calculated n_ood_test_users, but limit to available users with embeddings
                    actual_n_ood_users = min(n_ood_test_users, len(remaining_users_with_emb))
                    
                    # Identify clustering-based outlier users from remaining users (seeded)
                    outlier_user_ids, normal_user_ids, _, _ = select_ood_test_users(
                        remaining_embeddings, remaining_users_with_emb, actual_n_ood_users, random_seed=random_test_seed
                    )
                    
                    # Create splits
                    train_data = []
                    ood_test = []
                    
                    for uid in normal_user_ids:
                        train_data.extend(user_data[uid])
                    for uid in outlier_user_ids:
                        ood_test.extend(user_data[uid])
                        
                else:
                    # No embeddings available or no OOD test users needed, use remaining users for train
                    train_data = []
                    for uid in remaining_users_after_random:
                        train_data.extend(user_data[uid])
                    ood_test = []
            else:
                # No embeddings, use remaining users for train
                train_data = []
                for uid in remaining_users_after_random:
                    train_data.extend(user_data[uid])
                ood_test = []
            
            train_users = len(set(item.get('user_id', 'unknown') for item in train_data)) if train_data else 0
            random_test_users = len(set(item.get('user_id', 'unknown') for item in random_test)) if random_test else 0
            ood_test_users = len(set(item.get('user_id', 'unknown') for item in ood_test)) if ood_test else 0
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)
    
    if base_name.endswith("_data"):
        base_name = base_name[:-5]  # Remove "_data" suffix
    
    train_file = os.path.join(output_dir, f"{version}/{base_name}_train.jsonl")
    random_test_file = os.path.join(output_dir, f"{version}/{base_name}_random_test.jsonl")
    ood_test_file = os.path.join(output_dir, f"{version}/{base_name}_ood_test.jsonl")
    
    # Check for overlaps between splits
    (train_random_overlap, train_ood_overlap, random_ood_overlap) = check_data_overlap(
        train_data, random_test, ood_test)
    
    # Report problematic overlaps (only train vs test overlaps are issues)
    train_test_overlaps = train_random_overlap + train_ood_overlap
    if train_test_overlaps == 0:
        print("✓ No problematic overlaps detected between train and test splits")
        if random_ood_overlap > 0:
            print(f"   Test-Test overlap: Random/OOD={random_ood_overlap} (allowed)")
    else:
        print(f"⚠️  WARNING: Problematic overlaps detected!")
        if train_random_overlap > 0:
            print(f"   Train/Random_test overlap: {train_random_overlap} samples")
        if train_ood_overlap > 0:
            print(f"   Train/OOD_test overlap: {train_ood_overlap} samples")
        if random_ood_overlap > 0:
            print(f"   Test-Test overlap: Random/OOD={random_ood_overlap} (allowed)")
    
    # Save files
    save_jsonl(train_data, train_file)
    save_jsonl(random_test, random_test_file)
    if ood_test:  # Only save if we have OOD test data
        save_jsonl(ood_test, ood_test_file)
    
    print(f"Saved train data to: {train_file} ({len(train_data)} samples from {train_users} users)")
    print(f"Saved random test data to: {random_test_file} ({len(random_test)} samples from {random_test_users} users)")
    if ood_test:
        print(f"Saved OOD test data to: {ood_test_file} ({len(ood_test)} samples from {ood_test_users} users)")
    else:
        print(f"No OOD test data created (no embeddings available)")
    
    return (train_users, random_test_users, ood_test_users,
            len(train_data), len(random_test), len(ood_test))

def main():
    """Main function to process all datasets with unified sampling"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified dataset sampling with configurable version and embedding directory')
    parser.add_argument('--version', default='v5', help='Version directory for output')
    parser.add_argument('--embedding-dir', default='user_gen_profile_embeddings', 
                       help='Directory containing user embeddings (default: user_gen_profile_embeddings)')
    parser.add_argument('--random-test-seed', type=int, default=None, 
                        help='Random seed applied to both random_test and OOD splits')
    args = parser.parse_args()
    
    # Do not set global RNG seeds here. If provided, --random-test-seed
    # is applied locally to both random_test and OOD sampling via context or function args.
    
    # Base directory for embeddings
    embedding_dir = args.embedding_dir
    
    # Dataset configurations with optional embedding files
    datasets = [
        # # LaMP datasets
        # ("LaMP/LaMP_processed_tweet_data.jsonl", "LaMP Tweet", 
        #  f"{embedding_dir}/lamp_tweet_user_embeddings.npz"),
        # ("LaMP/LaMP_processed_scholarly_title_data.jsonl", "LaMP Scholarly Title",
        #  f"{embedding_dir}/lamp_scholarly_title_user_embeddings.npz"),
        # ("LaMP/LaMP_processed_product_data.jsonl", "LaMP Product",
        #  f"{embedding_dir}/lamp_product_user_embeddings.npz"),
        # ("LaMP/LaMP_processed_news_headline_data.jsonl", "LaMP News Headline",
        #  f"{embedding_dir}/lamp_news_headline_user_embeddings.npz"),
        # ("LaMP/LaMP_processed_movie_data.jsonl", "LaMP Movie",
        #  f"{embedding_dir}/lamp_movie_user_embeddings.npz"),
        # ("LaMP/LaMP_processed_news_cat_data.jsonl", "LaMP News Category",
        #  f"{embedding_dir}/lamp_news_cat_user_embeddings.npz"),
        # ("LaMP/LaMP_processed_citation_data.jsonl", "LaMP Citation",
        #  f"{embedding_dir}/lamp_citation_user_embeddings.npz"),
         
        # # LongLaMP datasets
        # ("LongLaMP/LongLaMP_abstract_generation_data.jsonl", "LongLaMP Abstract Generation",
        #  f"{embedding_dir}/longlamp_abstract_generation_user_embeddings.npz"),
        # ("LongLaMP/LongLaMP_product_review_data.jsonl", "LongLaMP Product Review",
        #  f"{embedding_dir}/longlamp_product_review_user_embeddings.npz"),
        # ("LongLaMP/LongLaMP_topic_writing_data.jsonl", "LongLaMP Topic Writing",
        #  f"{embedding_dir}/longlamp_topic_writing_user_embeddings.npz"),
         
        # Other datasets
        # ("PRISM/PRISM_data.jsonl", "PRISM",
        #  f"{embedding_dir}/prism_user_embeddings.npz"),
        # ("OpinionQA/OpinionQA_data.jsonl", "OpinionQA",
        #  f"{embedding_dir}/opinionqa_user_embeddings.npz"),
        # ("PersonalReddit/PersonalReddit_data.jsonl", "PersonalReddit",
        #  f"{embedding_dir}/personalreddit_user_embeddings.npz"),
        ("EC/EC_data.jsonl", "EC",
         f"{embedding_dir}/ec_user_embeddings.npz"),
        # ("ALOE/ALOE_data.jsonl", "ALOE",
        #  f"{embedding_dir}/aloe_user_embeddings.npz")
    ]
    
    results = []
    
    for file_path, dataset_name, embedding_file in datasets:
        try:
            result = process_dataset_unified(file_path, dataset_name, embedding_file, args.version, args.random_test_seed)
            if result:
                results.append((dataset_name, *result))
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n=== SUMMARY (Unified Splits) ===")
    print(f"{'Dataset':<30} {'Train Users':<12} {'RndT Users':<11} {'OOD Users':<11} {'Train Recs':<11} {'RndT Recs':<10} {'OOD Recs':<10}")
    print("-" * 110)
    for dataset_name, train_users, random_test_users, ood_users, train_records, random_test_records, ood_records in results:
        print(f"{dataset_name:<30} {train_users:<12} {random_test_users:<11} {ood_users:<11} {train_records:<11} {random_test_records:<10} {ood_records:<10}")
    
    print("\nNote: Train = normal users, RndT = random test sample")
    print("      OOD = EXTREME clustering-based outlier users (complete cluster separation)")
    print("Overlap policy: No overlap between train and test sets; test sets may overlap with each other")
    print(f"Files saved in {args.version}/ directory with naming pattern: *_train.jsonl, *_random_test.jsonl, *_ood_test.jsonl")

if __name__ == "__main__":
    main() 
