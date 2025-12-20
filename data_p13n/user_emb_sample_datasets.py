#!/usr/bin/env python3
"""
Split datasets based on user embeddings to create out-of-distribution test sets.

This script uses pre-computed user embeddings to split users into train/test sets
where test users are maximally dissimilar to train users (out-of-distribution).
It follows the same train/test split sizes as random_sample_datasets.py.
"""

import json
import numpy as np
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def group_by_user(data, user_key='user_id'):
    """Group data by user_id"""
    user_data = defaultdict(list)
    for item in data:
        user_data[item[user_key]].append(item)
    return user_data

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

def select_ood_test_users_clustering(embeddings: np.ndarray, n_train: int, n_test: int, 
                                   user_ids: List[str], random_seed: int = 42, 
                                   min_test_cluster_size: int = 5) -> Tuple[List[int], List[int]]:
    """
    Select test users using clustering-based OOD strategy.
    
    1. Cluster users into groups using K-means
    2. Select test users from minority/distant clusters to ensure OOD property
    3. Restrict test set size as specified
    
    Args:
        embeddings: User embeddings (normalized)
        n_train: Number of train users
        n_test: Number of test users (restricted)
        user_ids: List of user IDs
        random_seed: Random seed for reproducibility
        min_test_cluster_size: Minimum size for clusters to be considered for test selection
    """
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
    max_test_size = min(n_test, max(50, n_users // 10))  # At most 10% or minimum 50
    n_test = min(n_test, max_test_size)
    n_train = min(n_train, n_users - n_test)
    
    logger.info(f"Clustering-based selection: {n_train} train and {n_test} test users from {n_users} total users")
    
    # Determine optimal number of clusters
    # Use a reasonable range based on dataset size
    min_clusters = max(2, min(10, n_users // 20))
    max_clusters = min(20, n_users // 5)
    
    best_k = min_clusters
    best_score = -1
    
    # Find optimal number of clusters using silhouette score
    logger.info(f"Finding optimal number of clusters (range: {min_clusters}-{max_clusters})...")
    for k in range(min_clusters, max_clusters + 1):
        if k >= n_users:
            break
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
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
    
    # Select test users from top-scoring clusters
    test_indices = []
    train_indices = []
    
    for score, info in cluster_scores:
        cluster_indices = info['indices']
        cluster_size = info['size']
        
        # Determine how many test users to take from this cluster
        if len(test_indices) < n_test and cluster_size >= min_test_cluster_size:
            # Take users from this cluster for test set
            remaining_test_slots = n_test - len(test_indices)
            n_test_from_cluster = min(remaining_test_slots, cluster_size // 2, cluster_size)
            
            # Randomly select test users from this cluster
            cluster_test_indices = np.random.choice(cluster_indices, size=n_test_from_cluster, replace=False)
            test_indices.extend(cluster_test_indices)
            
            # Remaining users from this cluster go to train (if we need more train users)
            remaining_cluster_indices = [idx for idx in cluster_indices if idx not in cluster_test_indices]
            train_indices.extend(remaining_cluster_indices)
            
            logger.info(f"Selected {n_test_from_cluster} test users from cluster {info['cluster_id']} "
                       f"(score={score:.3f}, size={cluster_size})")
        else:
            # All users from this cluster go to train set
            train_indices.extend(cluster_indices)
    
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
    
    logger.info(f"Clustering-based OOD Test-Train similarity stats: avg={avg_test_train_sim:.3f}, "
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

def select_ood_test_users(embeddings: np.ndarray, n_train: int, n_test: int, 
                         user_ids: List[str], random_seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Select test users that are out-of-distribution using clustering approach.
    
    This function now uses clustering to identify groups of similar users,
    then selects test users from minority/distant clusters to ensure OOD property.
    """
    return select_ood_test_users_clustering(embeddings, n_train, n_test, user_ids, random_seed)

def process_dataset_with_embeddings(file_path: str, dataset_name: str, embedding_file: str, 
                                   target_train_users: int = None, target_test_users: int = None,
                                   target_train_samples: int = None, target_test_samples: int = None):
    """Process a dataset using user embeddings for OOD train/test split"""
    
    print(f"\n=== Processing {dataset_name} with embeddings ===")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
        
    if not os.path.exists(embedding_file):
        print(f"Embedding file not found: {embedding_file}")
        return None
    
    # Load data
    print(f"Loading data from {file_path}")
    data = load_jsonl(file_path)
    print(f"Total records: {len(data)}")
    
    # Load embeddings
    emb_user_ids, embeddings = load_user_embeddings(embedding_file)
    emb_user_id_to_idx = {uid: idx for idx, uid in enumerate(emb_user_ids)}
    
    # Special handling for different datasets
    if "OpinionQA" in dataset_name:
        # Special case for OpinionQA: maximize user diversity in test set
        user_data = group_by_user(data)
        
        # Filter to users with embeddings
        user_data = {uid: items for uid, items in user_data.items() if uid in emb_user_id_to_idx}
        dataset_user_ids = list(user_data.keys())
        
        if not dataset_user_ids:
            logger.warning("No users with embeddings found")
            return None
        
        # Get embeddings for these users
        dataset_user_indices = [emb_user_id_to_idx[uid] for uid in dataset_user_ids]
        dataset_embeddings = embeddings[dataset_user_indices]
        
        # Select diverse test users
        n_test_users = min(250, len(dataset_user_ids))  
        n_train_users = len(dataset_user_ids) - n_test_users
        
        train_idx, test_idx = select_ood_test_users(
            dataset_embeddings, n_train_users, n_test_users, dataset_user_ids
        )
        
        train_user_ids = [dataset_user_ids[i] for i in train_idx]
        test_user_ids = [dataset_user_ids[i] for i in test_idx]
        
        # Collect questions from these users
        train_data = []
        test_data = []
        
        for uid in test_user_ids:
            test_data.extend(user_data[uid][:1])  # One question per test user for diversity
        
        for uid in train_user_ids:
            train_data.extend(user_data[uid])
        
        # If we need more samples, add more from users
        if len(test_data) < 250:
            for uid in test_user_ids:
                available = [q for q in user_data[uid] if q not in test_data]
                test_data.extend(available[:250-len(test_data)])
                if len(test_data) >= 250:
                    break
        
        # Shuffle and trim
        random.shuffle(train_data)
        random.shuffle(test_data)
        train_data = train_data[:10000]
        test_data = test_data[:250]
        
        train_users = len(set(item['user_id'] for item in train_data))
        test_users = len(set(item['user_id'] for item in test_data))
        
        print(f"OOD user split for {dataset_name}: {len(train_data)} train from {train_users} users, "
              f"{len(test_data)} test from {test_users} users")
        
    else:
        # Standard user-based splitting (EC, PersonalReddit, PRISM, LaMP, LongLaMP, etc.)
        user_data = group_by_user(data)
        
        # Filter to users with embeddings
        user_data = {uid: items for uid, items in user_data.items() if uid in emb_user_id_to_idx}
        dataset_user_ids = list(user_data.keys())
        
        if not dataset_user_ids:
            logger.warning("No users with embeddings found")
            return None
            
        print(f"Found {len(dataset_user_ids)} users with embeddings")
        
        # Get embeddings for these users
        dataset_user_indices = [emb_user_id_to_idx[uid] for uid in dataset_user_ids]
        dataset_embeddings = embeddings[dataset_user_indices]
        
        # Determine target sizes
        if dataset_name in ["EC", "PersonalReddit"]:
            # Use 4:1 ratio
            n_test_users = max(1, len(dataset_user_ids) // 5)
            n_train_users = len(dataset_user_ids) - n_test_users
        elif dataset_name in ["LaMP Citation", "LaMP Tweet", "LaMP Scholarly Title", "LaMP Product",
                           "LongLaMP Abstract Generation", "LongLaMP Product Review", "LongLaMP Topic Writing"]:
            # For LaMP/LongLaMP, use 250 test users (or less if not enough users)
            n_test_users = min(250, len(dataset_user_ids))
            n_train_users = len(dataset_user_ids) - n_test_users
            
            # If we have less than 250 users total, use 4:1 ratio
            if len(dataset_user_ids) < 250:
                n_test_users = max(1, len(dataset_user_ids) // 5)
                n_train_users = len(dataset_user_ids) - n_test_users
        else:
            # Use target sizes or adjust based on available data
            n_train_users = target_train_users or 1000
            n_test_users = target_test_users or 250
            
            if len(dataset_user_ids) < n_train_users + n_test_users:
                # Use 4:1 ratio
                n_test_users = max(1, len(dataset_user_ids) // 5)
                n_train_users = len(dataset_user_ids) - n_test_users
        
        # Select OOD test users
        train_idx, test_idx = select_ood_test_users(
            dataset_embeddings, n_train_users, n_test_users, dataset_user_ids
        )
        
        train_user_ids = [dataset_user_ids[i] for i in train_idx]
        test_user_ids = [dataset_user_ids[i] for i in test_idx]
        
        # Create train and test datasets
        train_data = []
        test_data = []
        
        for uid in train_user_ids:
            train_data.extend(user_data[uid])
        for uid in test_user_ids:
            test_data.extend(user_data[uid])
        
        # No need to trim for LaMP/LongLaMP anymore - we want all data from the 250 test users
        
        train_users = len(train_user_ids)
        test_users = len(test_user_ids)
        
        print(f"Train data: {len(train_data)} records from {train_users} users")
        print(f"Test data: {len(test_data)} records from {test_users} users")
    
    # Generate output filenames (using 'ood' instead of 'random')
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)
    
    # Remove "_data" from base_name if it exists to match the renamed OOD files
    if base_name.endswith("_data"):
        base_name = base_name[:-5]  # Remove "_data" suffix
    
    train_file = os.path.join(output_dir, f"v2/{base_name}_ood_train.jsonl")
    test_file = os.path.join(output_dir, f"v2/{base_name}_ood_test.jsonl")
    
    # Save files
    save_jsonl(train_data, train_file)
    save_jsonl(test_data, test_file)
    
    print(f"Saved OOD train data to: {train_file}")
    print(f"Saved OOD test data to: {test_file}")
    
    return train_users, test_users, len(train_data), len(test_data)

def main():
    """Main function to process all datasets with embedding-based OOD splits"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Base directory for embeddings
    embedding_dir = "user_gen_profile_embeddings_task_specific"
    
    # Dataset configurations
    datasets = [
        # LaMP datasets - sample-based splitting
        ("LaMP/LaMP_processed_tweet_data.jsonl", "LaMP Tweet", 
         f"{embedding_dir}/lamp_tweet_user_embeddings.npz", None, None, None, 250),
        ("LaMP/LaMP_processed_scholarly_title_data.jsonl", "LaMP Scholarly Title",
         f"{embedding_dir}/lamp_scholarly_title_user_embeddings.npz", None, None, None, 250),
        ("LaMP/LaMP_processed_product_data.jsonl", "LaMP Product",
         f"{embedding_dir}/lamp_product_user_embeddings.npz", None, None, None, 250),
        ("LaMP/LaMP_processed_news_headline_data.jsonl", "LaMP News Headline",
         f"{embedding_dir}/lamp_news_headline_user_embeddings.npz", None, None, None, None),
        ("LaMP/LaMP_processed_movie_data.jsonl", "LaMP Movie",
         f"{embedding_dir}/lamp_movie_user_embeddings.npz", None, None, None, None),
        ("LaMP/LaMP_processed_news_cat_data.jsonl", "LaMP News Category",
         f"{embedding_dir}/lamp_news_cat_user_embeddings.npz", None, None, None, None),
        ("LaMP/LaMP_processed_citation_data.jsonl", "LaMP Citation",
         f"{embedding_dir}/lamp_citation_user_embeddings.npz", None, None, None, 250),
         
        # LongLaMP datasets - sample-based splitting
        ("LongLaMP/LongLaMP_abstract_generation_data.jsonl", "LongLaMP Abstract Generation",
         f"{embedding_dir}/longlamp_abstract_generation_user_embeddings.npz", None, None, None, 250),
        ("LongLaMP/LongLaMP_product_review_data.jsonl", "LongLaMP Product Review",
         f"{embedding_dir}/longlamp_product_review_user_embeddings.npz", None, None, None, 250),
        ("LongLaMP/LongLaMP_topic_writing_data.jsonl", "LongLaMP Topic Writing",
         f"{embedding_dir}/longlamp_topic_writing_user_embeddings.npz", None, None, None, 250),
         
        # Other datasets - user-based splitting
        ("PRISM/PRISM_data.jsonl", "PRISM",
         f"{embedding_dir}/prism_user_embeddings.npz", 1000, 250, None, None),
        ("OpinionQA/OpinionQA_data.jsonl", "OpinionQA",
         f"{embedding_dir}/opinionqa_user_embeddings.npz", None, None, 10000, 250),
        ("PersonalReddit/PersonalReddit_data.jsonl", "PersonalReddit",
         f"{embedding_dir}/personalreddit_user_embeddings.npz", None, None, None, None),  # 4:1 ratio
        ("EC/EC_data.jsonl", "EC",
         f"{embedding_dir}/ec_user_embeddings.npz", None, None, None, None)  # 4:1 ratio
    ]
    
    results = []
    
    for file_path, dataset_name, embedding_file, target_train_users, target_test_users, target_train_samples, target_test_samples in datasets:
        try:
            # Calculate target train samples for LaMP/LongLaMP if not specified
            if target_train_samples is None and target_test_samples is not None:
                # For LaMP/LongLaMP, use all data minus test samples
                data = load_jsonl(file_path) if os.path.exists(file_path) else []
                target_train_samples = max(0, len(data) - target_test_samples)
            
            result = process_dataset_with_embeddings(
                file_path, dataset_name, embedding_file,
                target_train_users, target_test_users,
                target_train_samples, target_test_samples
            )
            if result:
                results.append((dataset_name, *result))
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n=== SUMMARY (OOD Splits) ===")
    print(f"{'Dataset':<30} {'Train Users':<12} {'Test Users':<11} {'Train Records':<13} {'Test Records':<12}")
    print("-" * 80)
    for dataset_name, train_users, test_users, train_records, test_records in results:
        print(f"{dataset_name:<30} {train_users:<12} {test_users:<11} {train_records:<13} {test_records:<12}")
    
    print("\nNote: Test users are selected to be maximally dissimilar (out-of-distribution) from train users based on embeddings.")

if __name__ == "__main__":
    main() 