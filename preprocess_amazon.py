import json
import random
import gzip
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import os


def load_amazon_data(file_path):
    """加载Amazon评论数据"""
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                continue

    import pandas as pd
    return pd.DataFrame(data)


def filter_k_core(user_dict, item_counts, k=5):
    """K-core过滤"""
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


def preprocess_amazon(category='Luxury_Beauty', neg_per_pos=1):
    """
    预处理Amazon数据

    Args:
        category: 数据集类别
        neg_per_pos: 每个正样本对应的负样本数量
    """
    print(f"Processing {category}...")

    review_file = f'{category}.json.gz'
    meta_file = f'meta_{category}.json.gz'

    # 加载数据
    reviews = load_amazon_data(review_file)
    metadata = load_amazon_data(meta_file)

    # 对Luxury_Beauty特殊处理
    if category == 'Luxury_Beauty':
        reviews = reviews[reviews['overall'] >= 3]

    # 创建item映射
    item_titles = {}
    for _, row in metadata.iterrows():
        if 'title' in row and row['title'] is not None:
            item_titles[row['asin']] = str(row['title'])
        else:
            item_titles[row['asin']] = row['asin']

    # 构建用户交互字典
    user_dict = defaultdict(lambda: {'asin': [], 'rating': [], 'time': [], 'title': []})
    item_counts = defaultdict(int)

    for _, row in tqdm(reviews.iterrows(), desc="Processing reviews", total=len(reviews)):
        if row['asin'] not in item_titles:
            continue

        user_id = row['reviewerID']
        user_dict[user_id]['asin'].append(row['asin'])
        user_dict[user_id]['rating'].append(int(row['overall'] > 3))
        user_dict[user_id]['time'].append(row['unixReviewTime'])
        user_dict[user_id]['title'].append(item_titles[row['asin']])
        item_counts[row['asin']] += 1

    # 过滤
    min_interactions = 4 if category == 'Luxury_Beauty' else 5
    print(f"Filtering users with at least {min_interactions} interactions...")
    filtered_users = {k: v for k, v in user_dict.items() if len(v['asin']) >= min_interactions}

    # 获取所有items
    all_items = set()
    for user_data in filtered_users.values():
        all_items.update(user_data['asin'])

    print(f"Total users: {len(filtered_users)}")
    print(f"Total items: {len(all_items)}")

    # 创建序列数据
    sequential_data = []
    seq_len = 10

    for user_id, interactions in tqdm(filtered_users.items(), desc="Creating sequences"):
        # 按时间排序
        indices = sorted(range(len(interactions['asin'])), key=lambda i: interactions['time'][i])
        sorted_asin = [interactions['asin'][i] for i in indices]
        sorted_rating = [interactions['rating'][i] for i in indices]
        sorted_title = [interactions['title'][i] for i in indices]
        sorted_time = [interactions['time'][i] for i in indices]

        # 为每个位置创建序列
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

    print(f"Total sequences: {len(sequential_data)}")

    # 划分数据集
    random.shuffle(sequential_data)
    train_size = int(len(sequential_data) * 0.8)
    valid_size = int(len(sequential_data) * 0.9)

    train_data = sequential_data[:train_size]
    valid_data = sequential_data[train_size:valid_size]
    test_data = sequential_data[valid_size:]

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    # 生成对比学习格式的数据
    def create_contrastive_samples(data_list, output_path, neg_per_pos):
        """
        为每个序列生成对比样本
        正样本：用户历史 + 目标item + label=1
        负样本：用户历史 + 随机item + label=0
        """
        samples = []

        for item in tqdm(data_list, desc=f"Generating {output_path}"):
            # 只处理正样本（用户确实喜欢的）
            if item['target_rating'] != 1:
                continue

            # 构建历史描述
            history_items = []
            for idx, (title, timestamp) in enumerate(zip(item['history_title'], item['history_time'])):
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y/%m/%d')
                history_items.append(f"No.{idx + 1} Time: {date_str} Title: {title}")

            history_str = ", ".join(history_items) if history_items else "None"

            # 正样本
            samples.append({
                "history": history_str,
                "candidate": item['target_title'],
                "label": 1
            })

            # 负样本
            random.shuffle(item['neg_candidates'])
            for neg_asin in item['neg_candidates'][:neg_per_pos]:
                neg_title = item_titles.get(neg_asin, neg_asin)
                samples.append({
                    "history": history_str,
                    "candidate": neg_title,
                    "label": 0
                })

        # 打乱顺序
        random.shuffle(samples)

        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2)

        pos_count = sum(1 for s in samples if s['label'] == 1)
        neg_count = sum(1 for s in samples if s['label'] == 0)
        print(f"Saved {len(samples)} samples to {output_path}")
        print(f"  Positive: {pos_count}, Negative: {neg_count}")

    # 生成训练/验证/测试数据
    os.makedirs('./data', exist_ok=True)

    # 训练和验证：对比学习格式（固定1:1负样本比例）
    create_contrastive_samples(train_data, f'./data/train_{category.lower()}.json', neg_per_pos=1)
    create_contrastive_samples(valid_data, f'./data/valid_{category.lower()}.json', neg_per_pos=1)

    # 测试数据需要保留完整候选列表用于ranking
    def create_test_samples(data_list, output_path, num_candidates):
        """
        测试数据格式：
        {
            "history": "...",
            "candidates": ["item1", "item2", ...],
            "target_index": 5
        }
        """
        samples = []

        for item in tqdm(data_list, desc=f"Generating {output_path}"):
            if item['target_rating'] != 1:
                continue

            history_items = []
            for idx, (title, timestamp) in enumerate(zip(item['history_title'], item['history_time'])):
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y/%m/%d')
                history_items.append(f"No.{idx + 1} Time: {date_str} Title: {title}")

            history_str = ", ".join(history_items) if history_items else "None"

            # 采样负样本
            random.shuffle(item['neg_candidates'])
            neg_samples = item['neg_candidates'][:num_candidates - 1]

            # 组合候选并打乱
            candidates_asin = [item['target_asin']] + neg_samples
            random.shuffle(candidates_asin)
            target_index = candidates_asin.index(item['target_asin'])

            candidate_titles = [item_titles.get(asin, asin) for asin in candidates_asin]

            samples.append({
                "history": history_str,
                "candidates": candidate_titles,
                "target_index": target_index
            })

        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2)

        print(f"Saved {len(samples)} test samples to {output_path}")

    # 生成不同候选数的测试数据
    create_test_samples(test_data, f'./data/test_{category.lower()}_20.json', num_candidates=20)
    create_test_samples(test_data, f'./data/test_{category.lower()}_100.json', num_candidates=100)

    print(f"\nPreprocessing complete for {category}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='Luxury_Beauty',
                       choices=['Luxury_Beauty', 'Software', 'Video_Games'],
                       help='Dataset category')
    parser.add_argument('--neg_ratio', type=int, default=19,
                       choices=[19, 99],
                       help='Negative samples per positive (19 or 99)')

    args = parser.parse_args()

    # 处理数据集
    preprocess_amazon(args.category, neg_per_pos=args.neg_ratio)