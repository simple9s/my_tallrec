import json
import pandas as pd
import random
import gzip
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import os


def load_amazon_data(file_path):
    """Load Amazon review data from gzip file"""
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)


def filter_k_core(user_dict, item_counts, k=5):
    """Filter k-core: users and items with at least k interactions"""
    while True:
        new_user_dict = {}
        for user, interactions in user_dict.items():
            if len(interactions['asin']) >= k:
                new_user_dict[user] = interactions

        new_item_counts = defaultdict(int)
        for user, interactions in new_user_dict.items():
            valid_items = []
            valid_ratings = []
            valid_times = []
            valid_titles = []
            for i, item in enumerate(interactions['asin']):
                if item_counts[item] >= k:
                    valid_items.append(item)
                    valid_ratings.append(interactions['rating'][i])
                    valid_times.append(interactions['time'][i])
                    valid_titles.append(interactions['title'][i])
                    new_item_counts[item] += 1

            if len(valid_items) >= k:
                new_user_dict[user] = {
                    'asin': valid_items,
                    'rating': valid_ratings,
                    'time': valid_times,
                    'title': valid_titles
                }

        if len(new_user_dict) == len(user_dict):
            break

        user_dict = new_user_dict
        item_counts = new_item_counts

    return user_dict


def preprocess_amazon_with_candidates(category='Luxury_Beauty', neg_ratio=19):
    print(f"Loading {category} data...")
    review_file = f'{category}.json.gz'
    meta_file = f'meta_{category}.json.gz'

    reviews = load_amazon_data(review_file)
    metadata = load_amazon_data(meta_file)

    if category == 'Luxury_Beauty' and neg_ratio == 19:
        print("Filtering luxury_beauty: overall >= 3")
        reviews = reviews[reviews['overall'] >= 3]

    item_to_idx = {asin: idx for idx, asin in enumerate(metadata['asin'].unique())}
    all_items = set(item_to_idx.keys())

    item_titles = {}
    for _, row in metadata.iterrows():
        if 'title' in row and pd.notna(row['title']):
            item_titles[row['asin']] = row['title']
        else:
            item_titles[row['asin']] = row['asin']

    user_dict = defaultdict(lambda: {'asin': [], 'rating': [], 'time': [], 'title': []})
    item_counts = defaultdict(int)

    for _, row in tqdm(reviews.iterrows(), desc="Processing reviews", total=len(reviews)):
        if row['asin'] not in item_to_idx:
            continue
        user_id = row['reviewerID']
        user_dict[user_id]['asin'].append(row['asin'])
        user_dict[user_id]['rating'].append(int(row['overall'] > 3))
        user_dict[user_id]['time'].append(row['unixReviewTime'])
        user_dict[user_id]['title'].append(item_titles.get(row['asin'], row['asin']))
        item_counts[row['asin']] += 1

    if neg_ratio == 19:
        if category == 'Luxury_Beauty':
            min_interactions = 4
        else:
            min_interactions = 5
        print(f"Filtering users with at least {min_interactions} interactions...")
        filtered_users = {k: v for k, v in user_dict.items() if len(v['asin']) >= min_interactions}
    else:
        print("Applying 5-core filtering...")
        filtered_users = filter_k_core(dict(user_dict), item_counts, k=5)

    all_items = set()
    for user_data in filtered_users.values():
        all_items.update(user_data['asin'])

    metadata[['asin', 'title']].to_csv(f'{category.lower()}_item_mapping.csv', index=False)

    sequential_data = []
    seq_len = 10

    for user_id, interactions in tqdm(filtered_users.items(), desc="Creating sequences"):
        indices = sorted(range(len(interactions['asin'])), key=lambda i: interactions['time'][i])
        sorted_asin = [interactions['asin'][i] for i in indices]
        sorted_rating = [interactions['rating'][i] for i in indices]
        sorted_title = [interactions['title'][i] for i in indices]
        sorted_time = [interactions['time'][i] for i in indices]

        for i in range(min(seq_len, len(sorted_asin) - 1), len(sorted_asin)):
            user_history = set(sorted_asin[:i + 1])
            neg_candidates = list(all_items - user_history)

            sequential_data.append({
                'user_id': user_id,
                'history_asin': sorted_asin[max(0, i - seq_len):i],
                'history_rating': sorted_rating[max(0, i - seq_len):i],
                'history_title': sorted_title[max(0, i - seq_len):i],
                'history_time': sorted_time[max(0, i - seq_len):i],
                'target_asin': sorted_asin[i],
                'target_rating': sorted_rating[i],
                'target_title': sorted_title[i],
                'neg_candidates': neg_candidates
            })

    random.shuffle(sequential_data)
    train_size = int(len(sequential_data) * 0.8)
    valid_size = int(len(sequential_data) * 0.9)

    train_data = sequential_data[:train_size]
    valid_data = sequential_data[train_size:valid_size]
    test_data = sequential_data[valid_size:]

    def create_json_with_candidates(data_list, output_path, neg_ratio):
        json_list = []
        for item in tqdm(data_list, desc=f"Generating {output_path}"):
            if item['target_rating'] != 1:
                continue

            history_items = []
            for idx, (title, timestamp) in enumerate(zip(item['history_title'], item['history_time'])):
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y/%m/%d')
                history_items.append(f"No.{idx + 1} Time: {date_str} Title: {title}")

            history_str = ", ".join(history_items) if history_items else "None"

            random.shuffle(item['neg_candidates'])
            neg_samples = item['neg_candidates'][:neg_ratio]

            candidates_asin = [item['target_asin']] + neg_samples
            random.shuffle(candidates_asin)
            target_index = candidates_asin.index(item['target_asin'])

            candidate_titles = [item_titles.get(asin, asin) for asin in candidates_asin]

            json_list.append({
                "instruction": f"This user has made a series of purchases in the following order: [{history_str}]. Based on this sequence of purchases, generate user representation token: [UserRep].",
                "input": f"Given the {neg_ratio + 1} candidate items below, select which item the user will most likely purchase next:\n" + "\n".join(
                    [f'{i + 1}. Title: {title}' for i, title in enumerate(candidate_titles)]),
                "output": str(target_index + 1),
                "candidates": candidate_titles,
                "target_index": target_index
            })

        with open(output_path, 'w') as f:
            json.dump(json_list, f, indent=4)

        print(f"Generated {len(json_list)} samples to {output_path}")

    create_json_with_candidates(train_data, f'./data/train_{category.lower()}_{neg_ratio + 1}.json', neg_ratio)
    create_json_with_candidates(valid_data, f'./data/valid_{category.lower()}_{neg_ratio + 1}.json', neg_ratio)
    create_json_with_candidates(test_data, f'./data/test_{category.lower()}_{neg_ratio + 1}.json', neg_ratio)


if __name__ == "__main__":
    os.makedirs('./data', exist_ok=True)
    import multiprocessing
    from functools import partial


    def process_dataset(args):
        """每个进程执行的任务函数"""
        dataset_name, neg_ratio = args
        print(f"\n{'=' * 50}")
        print(f"Processing {dataset_name} with 1:{neg_ratio} ratio")
        print(f"{'=' * 50}")

        # 调用你的预处理函数
        preprocess_amazon_with_candidates(dataset_name, neg_ratio)

        print(f"Completed {dataset_name} with 1:{neg_ratio} ratio")
        return f"{dataset_name}_{neg_ratio}"


    def main():
        # 定义所有要处理的任务组合
        datasets = ['Luxury_Beauty', 'Software', 'Video_Games']
        neg_ratios = [19, 99]

        # 创建所有任务参数组合
        tasks = [(dataset, ratio) for dataset in datasets for ratio in neg_ratios]

        print(f"开始处理 {len(tasks)} 个任务")
        print(f"可用CPU核心数: {multiprocessing.cpu_count()}")

        # 设置进程池大小（通常为CPU核心数或稍少）
        # 对于I/O密集型任务，可以设置稍多一些
        pool_size = min(multiprocessing.cpu_count(), len(tasks))

        # 使用进程池并行执行
        with multiprocessing.Pool(processes=pool_size) as pool:
            results = pool.map(process_dataset, tasks)

        print(f"\n所有任务完成: {results}")


    # # Process Luxury Beauty
    # for neg_ratio in [19, 99]:
    #     print(f"\n{'='*50}")
    #     print(f"Processing Luxury_Beauty with 1:{neg_ratio} ratio")
    #     print(f"{'='*50}")
    #     preprocess_amazon_with_candidates('Luxury_Beauty', neg_ratio)

    # # Process Software
    # for neg_ratio in [19, 99]:
    #     print(f"\n{'='*50}")
    #     print(f"Processing Software with 1:{neg_ratio} ratio")
    #     print(f"{'='*50}")
    #     preprocess_amazon_with_candidates('Software', neg_ratio)

    # for neg_ratio in [19, 99]:
    #     print(f"\n{'='*50}")
    #     print(f"Processing Software with 1:{neg_ratio} ratio")
    #     print(f"{'='*50}")
    #     preprocess_amazon_with_candidates('Video_Games', neg_ratio)
    main()