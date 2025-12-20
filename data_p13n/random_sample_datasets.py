#!/usr/bin/env python3
import json
import random
from collections import defaultdict
import os

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

def filter_users_with_sufficient_history(user_data, dataset_name, min_history=0):
    """Filter users who have at least min_history items"""
    filtered_users = {}
    users_with_sufficient_history = 0
    
    for user_id, user_items in user_data.items():
        # For each user, check the first item's history length (all items from same user should have same history)
        if user_items:
            history_length = get_history_length(user_items[0], dataset_name)
            if history_length >= min_history:
                filtered_users[user_id] = user_items
                users_with_sufficient_history += 1
    
    print(f"Users with {min_history}+ history items: {users_with_sufficient_history} out of {len(user_data)}")
    return filtered_users

def sample_users_even_distribution(user_data, dataset_name, target_train_users=1000, target_test_users=250, min_history_for_test=0):
    """Sample users with stratification by user history length so train/test have similar distributions.

    - Determine each user's history length (based on dataset type).
    - Allocate test users proportionally across history-length buckets (respecting min_history_for_test).
    - Allocate train users from remaining users using the same stratification.
    """

    def get_user_history_len(user_id):
        items = user_data[user_id]
        return get_history_length(items[0], dataset_name) if items else 0

    users = list(user_data.keys())
    total_users = len(users)
    print(f"Total unique users: {total_users}")

    # Filter users with sufficient history for test set
    users_with_sufficient_history = filter_users_with_sufficient_history(user_data, dataset_name, min_history_for_test)
    eligible_test_users = list(users_with_sufficient_history.keys())
    if not eligible_test_users:
        print(f"No users found with {min_history_for_test}+ history items. Proceeding without history filter.")
        eligible_test_users = users

    # Decide split sizes
    if total_users < (target_train_users + target_test_users):
        # Use 5:1 ratio for smaller datasets
        test_users = max(1, min(len(eligible_test_users), max(1, total_users // 5)))
        train_users = max(0, total_users - test_users)
        print(f"Dataset has less than {target_train_users + target_test_users} users. Using 5:1 ratio: {train_users} train, {test_users} test")
    else:
        test_users = min(target_test_users, len(eligible_test_users))
        train_users = min(target_train_users, max(0, len(users) - test_users))
        print(f"Using target split: {train_users} train, {test_users} test")

    # Build buckets by exact history length
    bucket_to_users = defaultdict(list)
    for u in users:
        bucket_to_users[get_user_history_len(u)].append(u)

    bucket_to_eligible = defaultdict(list)
    for u in eligible_test_users:
        bucket_to_eligible[get_user_history_len(u)].append(u)

    # Helper: proportional allocation with capping and remainder distribution
    def proportional_allocate(bucket_to_pool, total_pool_count, n_target):
        # First pass: floor allocation
        allocations = {}
        remainders = []
        assigned = 0
        for b, pool in bucket_to_pool.items():
            desired = (len(bucket_to_users[b]) / total_users) * n_target if total_users > 0 else 0
            alloc = int(desired)
            # Cap to available in this pool
            alloc = min(alloc, len(pool))
            allocations[b] = alloc
            assigned += alloc
            remainders.append((desired - int(desired), b))
        # Distribute remaining slots by largest remainder where capacity exists
        remaining = n_target - assigned
        if remaining > 0:
            # Sort by remainder descending
            remainders.sort(key=lambda x: x[0], reverse=True)
            for _, b in remainders:
                if remaining <= 0:
                    break
                cap = len(bucket_to_pool[b]) - allocations[b]
                if cap > 0:
                    take = min(cap, remaining)
                    allocations[b] += take
                    remaining -= take
        return allocations

    # Allocate and sample test users per bucket
    test_alloc = proportional_allocate(bucket_to_eligible, len(eligible_test_users), test_users)
    test_users_list = []
    for b, alloc in test_alloc.items():
        cand = bucket_to_eligible[b]
        random.shuffle(cand)
        test_users_list.extend(cand[:alloc])

    # If still short due to caps, fill from any remaining eligible users regardless of bucket
    if len(test_users_list) < test_users:
        remaining_needed = test_users - len(test_users_list)
        remaining_pool = [u for u in eligible_test_users if u not in set(test_users_list)]
        random.shuffle(remaining_pool)
        test_users_list.extend(remaining_pool[:remaining_needed])

    # Training users: stratified sampling from remaining users
    remaining_users = [u for u in users if u not in set(test_users_list)]
    # Build remaining buckets
    rem_bucket_to_users = defaultdict(list)
    for u in remaining_users:
        rem_bucket_to_users[get_user_history_len(u)].append(u)
    train_alloc = proportional_allocate(rem_bucket_to_users, len(remaining_users), train_users)
    train_users_list = []
    for b, alloc in train_alloc.items():
        cand = rem_bucket_to_users[b]
        random.shuffle(cand)
        train_users_list.extend(cand[:alloc])
    if len(train_users_list) < train_users:
        remaining_needed = train_users - len(train_users_list)
        leftover_pool = [u for u in remaining_users if u not in set(train_users_list)]
        random.shuffle(leftover_pool)
        train_users_list.extend(leftover_pool[:remaining_needed])

    return train_users_list, test_users_list

def process_dataset(file_path, dataset_name):
    """Process a single dataset"""
    print(f"\n=== Processing {dataset_name} ===")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    if os.path.getsize(file_path) == 0:
        print(f"File is empty: {file_path}")
        return
    
    # Load data
    print(f"Loading data from {file_path}")
    data = load_jsonl(file_path)
    print(f"Total records: {len(data)}")
    
    # Special case for LaMP Citation and LaMP Tweet: sample entire dataset and split into training and 250 testing
    if dataset_name in ["LaMP Citation", "LaMP Tweet", "LaMP Scholarly Title", "LaMP Product", "LongLaMP Abstract Generation", "LongLaMP Product Review", "LongLaMP Topic Writing"]:
        random.shuffle(data)
        n_total = len(data)
        n_test = min(250, n_total)
        n_train = n_total - n_test
        train_data = data[:n_train]
        test_data = data[n_train:]
        print(f"Special split for {dataset_name}: {n_train} train, {n_test} test (250 test or all if less)")
        # For reporting, fake "users" as 1 (since we don't care about user split here)
        train_users = 1
        test_users = 1
    elif "OpinionQA" in dataset_name:
        # Special case for OpinionQA: sample 10000 questions for training and 250 for testing with user diversity
        user_data = group_by_user(data)
        users = list(user_data.keys())
        total_users = len(users)
        
        print(f"Total unique users: {total_users}")
        
        # Shuffle users to ensure random sampling
        random.shuffle(users)
        
        # Collect train and test data while ensuring user diversity
        train_data = []
        test_data = []
        train_target = 10000
        test_target = 250
        
        # First, sample test data (smaller, so easier to ensure diversity)
        test_users_used = set()
        for user in users:
            if len(test_data) >= test_target:
                break
            user_questions = user_data[user]
            if user_questions and user not in test_users_used:
                # Sample one question per user for test set to maximize diversity
                random.shuffle(user_questions)
                test_data.append(user_questions[0])
                test_users_used.add(user)
        
        # If we need more test data and have exhausted unique users, sample more from existing users
        remaining_users = [u for u in users if u in test_users_used]
        while len(test_data) < test_target and remaining_users:
            random.shuffle(remaining_users)
            for user in remaining_users:
                if len(test_data) >= test_target:
                    break
                user_questions = user_data[user]
                # Find questions not already in test_data
                used_questions = {id(q) for q in test_data if q.get('user_id') == user}
                available_questions = [q for q in user_questions if id(q) not in used_questions]
                if available_questions:
                    test_data.append(available_questions[0])
        
        # Now sample train data from remaining users and questions
        train_users_used = set()
        for user in users:
            if len(train_data) >= train_target:
                break
            if user not in test_users_used:  # Avoid users already in test set
                user_questions = user_data[user]
                if user_questions:
                    # Sample multiple questions per user for training, but spread across users
                    random.shuffle(user_questions)
                    questions_to_add = min(len(user_questions), max(1, train_target // total_users))
                    train_data.extend(user_questions[:questions_to_add])
                    train_users_used.add(user)
        
        # If we need more training data, sample from all available users (including test users but different questions)
        if len(train_data) < train_target:
            all_remaining_questions = []
            for user in users:
                user_questions = user_data[user]
                # For test users, exclude questions already in test set
                if user in test_users_used:
                    test_question_ids = {id(q) for q in test_data if q.get('user_id') == user}
                    available_questions = [q for q in user_questions if id(q) not in test_question_ids]
                else:
                    # For train-only users, exclude questions already in train set
                    train_question_ids = {id(q) for q in train_data if q.get('user_id') == user}
                    available_questions = [q for q in user_questions if id(q) not in train_question_ids]
                all_remaining_questions.extend(available_questions)
            
            # Shuffle and add remaining questions to reach target
            random.shuffle(all_remaining_questions)
            needed = train_target - len(train_data)
            train_data.extend(all_remaining_questions[:needed])
        
        train_users = len(train_users_used)
        test_users = len(test_users_used)
        
        print(f"Diverse sampling for {dataset_name}: {len(train_data)} train questions from {train_users} users, {len(test_data)} test questions from {test_users} users")
    elif any(ds in dataset_name for ds in ["EC", "PersonalReddit"]):
        # Special case for EC and PersonalReddit: sample users in 5:1 ratio for training and testing
        user_data = group_by_user(data)
        users = list(user_data.keys())
        total_users = len(users)
        
        # Use 5:1 ratio for train:test
        test_users_count = max(1, total_users // 5)
        train_users_count = total_users - test_users_count
        
        print(f"Total unique users: {total_users}")
        print(f"Using 5:1 ratio: {train_users_count} train users, {test_users_count} test users")
        
        # Randomly shuffle and split users
        random.shuffle(users)
        train_users_list = users[:train_users_count]
        test_users_list = users[train_users_count:train_users_count + test_users_count]
        
        # Create train and test datasets
        train_data = []
        test_data = []
        for user in train_users_list:
            train_data.extend(user_data[user])
        for user in test_users_list:
            test_data.extend(user_data[user])
        
        train_users = len(train_users_list)
        test_users = len(test_users_list)
        print(f"Train data: {len(train_data)} records from {train_users} users")
        print(f"Test data: {len(test_data)} records from {test_users} users")
        if test_data:
            sample_test_history = get_history_length(test_data[0], dataset_name)
            print(f"Sample test user history length: {sample_test_history}")
    else:
        # Group by user
        user_data = group_by_user(data)
        # Sample users with history requirement for test users
        train_users_list, test_users_list = sample_users_even_distribution(user_data, dataset_name)
        # Create train and test datasets
        train_data = []
        test_data = []
        for user in train_users_list:
            train_data.extend(user_data[user])
        for user in test_users_list:
            test_data.extend(user_data[user])
        train_users = len(train_users_list)
        test_users = len(test_users_list)
        print(f"Train data: {len(train_data)} records from {train_users} users")
        print(f"Test data: {len(test_data)} records from {test_users} users")
        # Verify test users have sufficient history
        if test_data:
            sample_test_history = get_history_length(test_data[0], dataset_name)
            print(f"Sample test user history length: {sample_test_history}")
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)
    
    if base_name.endswith("_data"):
        base_name = base_name[:-5]  # Remove "_data" suffix

    train_file = os.path.join(output_dir, f"v2/{base_name}_random_train.jsonl")
    test_file = os.path.join(output_dir, f"v2/{base_name}_random_test.jsonl")
    
    # Save files
    save_jsonl(train_data, train_file)
    save_jsonl(test_data, test_file)
    
    print(f"Saved train data to: {train_file}")
    print(f"Saved test data to: {test_file}")
    
    return train_users, test_users, len(train_data), len(test_data)

def main():
    """Main function to process all datasets"""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Dataset files to process
    datasets = [
        ("LaMP/LaMP_processed_tweet_data.jsonl", "LaMP Tweet"),
        ("LaMP/LaMP_processed_scholarly_title_data.jsonl", "LaMP Scholarly Title"),
        ("LaMP/LaMP_processed_product_data.jsonl", "LaMP Product"),
        ("LaMP/LaMP_processed_news_headline_data.jsonl", "LaMP News Headline"),
        ("LaMP/LaMP_processed_movie_data.jsonl", "LaMP Movie"),
        ("LaMP/LaMP_processed_news_cat_data.jsonl", "LaMP News Category"),
        ("LaMP/LaMP_processed_citation_data.jsonl", "LaMP Citation"),
        ("LongLaMP/LongLaMP_abstract_generation_data.jsonl", "LongLaMP Abstract Generation"),
        ("LongLaMP/LongLaMP_product_review_data.jsonl", "LongLaMP Product Review"),
        ("LongLaMP/LongLaMP_topic_writing_data.jsonl", "LongLaMP Topic Writing"),
        ("PRISM/PRISM_data.jsonl", "PRISM"),
        ("OpinionQA/OpinionQA_data.jsonl", "OpinionQA"),
        ("PersonalReddit/PersonalReddit_data.jsonl", "PersonalReddit"),
        ("EC/EC_data.jsonl", "EC")
    ]
    
    results = []
    
    for file_path, dataset_name in datasets:
        try:
            result = process_dataset(file_path, dataset_name)
            if result:
                results.append((dataset_name, *result))
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'Dataset':<30} {'Train Users':<12} {'Test Users':<11} {'Train Records':<13} {'Test Records':<12}")
    print("-" * 80)
    for dataset_name, train_users, test_users, train_records, test_records in results:
        print(f"{dataset_name:<30} {train_users:<12} {test_users:<11} {train_records:<13} {test_records:<12}")

if __name__ == "__main__":
    main() 
