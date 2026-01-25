import json
import pandas as pd
import random
import gzip
from collections import defaultdict
from tqdm import tqdm


def load_amazon_data(file_path):
    """Load Amazon review data from gzip file"""
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)


def preprocess_amazon_with_candidates(category='Luxury_Beauty', neg_ratio=19):
    """
    Preprocess Amazon data with candidate sets
    category: 'Luxury_Beauty' or 'Software'
    neg_ratio: 19 for 1:19, 99 for 1:99
    """
    # Load data
    print(f"Loading {category} data...")
    review_file = f'{category}.json.gz'
    meta_file = f'meta_{category}.json.gz'

    reviews = load_amazon_data(review_file)
    metadata = load_amazon_data(meta_file)

    # Create item mapping
    item_to_idx = {asin: idx for idx, asin in enumerate(metadata['asin'].unique())}
    all_items = set(item_to_idx.keys())

    # Get item titles
    item_titles = {}
    for _, row in metadata.iterrows():
        if 'title' in row and pd.notna(row['title']):
            item_titles[row['asin']] = row['title']
        else:
            item_titles[row['asin']] = row['asin']

    metadata[['asin', 'title']].to_csv(f'{category.lower()}_item_mapping.csv', index=False)

    # Build user interaction dict
    user_dict = defaultdict(lambda: {'asin': [], 'rating': [], 'time': [], 'title': []})

    for _, row in tqdm(reviews.iterrows(), desc="Processing reviews", total=len(reviews)):
        if row['asin'] not in item_to_idx:
            continue
        user_id = row['reviewerID']
        user_dict[user_id]['asin'].append(row['asin'])
        user_dict[user_id]['rating'].append(int(row['overall'] > 3))
        user_dict[user_id]['time'].append(row['unixReviewTime'])
        user_dict[user_id]['title'].append(item_titles.get(row['asin'], row['asin']))

    # Filter users with more than 3 interactions
    filtered_users = {k: v for k, v in user_dict.items() if len(v['asin']) > 3}

    # Sort by timestamp
    sequential_data = []
    seq_len = 10

    for user_id, interactions in tqdm(filtered_users.items(), desc="Creating sequences"):
        indices = sorted(range(len(interactions['asin'])), key=lambda i: interactions['time'][i])
        sorted_asin = [interactions['asin'][i] for i in indices]
        sorted_rating = [interactions['rating'][i] for i in indices]
        sorted_title = [interactions['title'][i] for i in indices]

        for i in range(seq_len, len(sorted_asin)):
            user_history = set(sorted_asin[:i + 1])
            neg_candidates = list(all_items - user_history)

            sequential_data.append({
                'user_id': user_id,
                'history_asin': sorted_asin[max(0, i - seq_len):i],
                'history_rating': sorted_rating[max(0, i - seq_len):i],
                'history_title': sorted_title[max(0, i - seq_len):i],
                'target_asin': sorted_asin[i],
                'target_rating': sorted_rating[i],
                'target_title': sorted_title[i],
                'neg_candidates': neg_candidates
            })

    # Split data
    random.shuffle(sequential_data)
    train_size = int(len(sequential_data) * 0.8)
    valid_size = int(len(sequential_data) * 0.9)

    train_data = sequential_data[:train_size]
    valid_data = sequential_data[train_size:valid_size]
    test_data = sequential_data[valid_size:]

    def create_json_with_candidates(data_list, output_path, neg_ratio):
        json_list = []
        for item in tqdm(data_list, desc=f"Generating {output_path}"):
            # Only use positive targets
            if item['target_rating'] != 1:
                continue

            # Build preference strings
            preference = [f'"{title}"' for title, rating in
                          zip(item['history_title'], item['history_rating']) if rating == 1]
            unpreference = [f'"{title}"' for title, rating in
                            zip(item['history_title'], item['history_rating']) if rating == 0]

            # Sample negative candidates
            random.shuffle(item['neg_candidates'])
            neg_samples = item['neg_candidates'][:neg_ratio]

            # Create candidate list
            candidates_asin = [item['target_asin']] + neg_samples
            random.shuffle(candidates_asin)
            target_index = candidates_asin.index(item['target_asin'])

            # Get candidate titles
            candidate_titles = [f'"{item_titles.get(asin, asin)}"' for asin in candidates_asin]

            preference_str = ", ".join(preference) if preference else "None"
            unpreference_str = ", ".join(unpreference) if unpreference else "None"
            candidates_str = ", ".join(candidate_titles)

            json_list.append({
                "instruction": f"Given the user's preference and unpreference, select the product the user will most likely enjoy from the following {neg_ratio + 1} candidates. Answer with the product title only.",
                "input": f"User Preference: {preference_str}\nUser Unpreference: {unpreference_str}\nCandidates: {candidates_str}\nWhich product will the user most likely enjoy?",
                "output": candidate_titles[target_index],
                "candidates": candidate_titles,
                "target_index": target_index
            })

        with open(output_path, 'w') as f:
            json.dump(json_list, f, indent=4)

    # Generate datasets
    create_json_with_candidates(train_data, f'./data/train_{category.lower()}_{neg_ratio + 1}.json', neg_ratio)
    create_json_with_candidates(valid_data, f'./data/valid_{category.lower()}_{neg_ratio + 1}.json', neg_ratio)
    create_json_with_candidates(test_data, f'./data/test_{category.lower()}_{neg_ratio + 1}.json', neg_ratio)


if __name__ == "__main__":
    # Process Luxury Beauty
    for neg_ratio in [19, 99]:
        print(f"\n{'=' * 50}")
        print(f"Processing Luxury_Beauty with 1:{neg_ratio} ratio")
        print(f"{'=' * 50}")
        preprocess_amazon_with_candidates('Luxury_Beauty', neg_ratio)

    # Process Software
    for neg_ratio in [19, 99]:
        print(f"\n{'=' * 50}")
        print(f"Processing Software with 1:{neg_ratio} ratio")
        print(f"{'=' * 50}")
        preprocess_amazon_with_candidates('Software', neg_ratio)