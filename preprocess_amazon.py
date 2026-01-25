import json
import random
import sys

from tqdm import tqdm
from collections import defaultdict
import gzip


def load_amazon_data(rating_file, meta_file):
    """加载Amazon 2018数据集"""
    print("Loading ratings...")
    ratings = []
    with gzip.open(rating_file, 'r') as f:
        for line in f:
            try:
                ratings.append(json.loads(line.strip()))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                continue

    print("Loading metadata...")
    meta_data = {}
    with gzip.open(meta_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if 'asin' in item:
                    meta_data[item['asin']] = item
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                continue

    return ratings, meta_data


def preprocess_amazon(rating_file, meta_file, output_prefix, min_interactions=10, seq_len=10):
    """预处理Amazon数据集，生成1正19负的训练数据"""
    ratings, meta_data = load_amazon_data(rating_file, meta_file)

    print("Building user interaction dictionary...")
    interaction_dicts = defaultdict(lambda: {'asin': [], 'rating': [], 'timestamp': []})

    for rating in tqdm(ratings):
        user_id = rating['reviewerID']
        asin = rating['asin']
        rate = rating['overall']
        timestamp = rating['unixReviewTime']

        interaction_dicts[user_id]['asin'].append(asin)
        interaction_dicts[user_id]['rating'].append(int(rate >= 4))  # 4-5星为正样本
        interaction_dicts[user_id]['timestamp'].append(timestamp)

    print(f"Filtering users with at least {min_interactions} interactions...")
    filtered_users = {uid: data for uid, data in interaction_dicts.items()
                      if len(data['asin']) >= min_interactions}
    print(f"Filtered users: {len(filtered_users)}")

    print("Creating item mapping...")
    all_items = list(set([asin for user_data in filtered_users.values()
                          for asin in user_data['asin']]))
    item_to_idx = {asin: idx for idx, asin in enumerate(all_items)}

    print(f"Total items: {len(all_items)}")

    # 保存物品映射，正确处理category字段
    with open(f'{output_prefix}_item_mapping.json', 'w') as f:
        item_mapping = []
        for asin, idx in item_to_idx.items():
            item_info = meta_data.get(asin, {})

            # 处理title
            title = item_info.get('title', 'Unknown')
            if isinstance(title, list):
                title = ' '.join(title)

            # 处理category - Amazon数据集中category可能有多种格式
            category = item_info.get('category', [])
            if not category:
                category = item_info.get('categories', [])

            # 如果category是嵌套列表，提取所有类别
            flat_category = []
            if isinstance(category, list):
                for cat in category:
                    if isinstance(cat, list):
                        flat_category.extend(cat)
                    else:
                        flat_category.append(cat)

            item_mapping.append({
                'item_id': idx,
                'asin': asin,
                'title': title,
                'category': flat_category
            })
        json.dump(item_mapping, f, indent=4)

    print("Creating sequential interactions...")
    sequential_interaction_list = []

    for user_id, user_data in tqdm(filtered_users.items()):
        # 按时间戳排序
        temp = list(zip(user_data['asin'], user_data['rating'], user_data['timestamp']))
        temp.sort(key=lambda x: x[2])
        asins, ratings, timestamps = zip(*temp)

        # 构建序列
        for i in range(seq_len, len(asins)):
            sequential_interaction_list.append({
                'user_id': user_id,
                'history_asin': list(asins[i - seq_len:i]),
                'history_rating': list(ratings[i - seq_len:i]),
                'target_asin': asins[i],
                'target_rating': ratings[i],
                'timestamp': timestamps[i]
            })

    print(f"Total sequences: {len(sequential_interaction_list)}")

    # 划分数据集：80% train, 10% valid, 10% test
    random.seed(42)
    random.shuffle(sequential_interaction_list)

    train_size = int(len(sequential_interaction_list) * 0.8)
    valid_size = int(len(sequential_interaction_list) * 0.9)

    train_data = sequential_interaction_list[:train_size]
    valid_data = sequential_interaction_list[train_size:valid_size]
    test_data = sequential_interaction_list[valid_size:]

    print(f"Train sequences: {len(train_data)}")
    print(f"Valid sequences: {len(valid_data)}")
    print(f"Test sequences: {len(test_data)}")

    def create_json_with_negatives(data, all_items, meta_data, item_to_idx, neg_ratio=19):
        """为每个序列创建1正19负的样本"""
        json_list = []

        for seq in tqdm(data, desc="Creating samples"):
            # 用户交互过的所有物品（历史+目标）
            user_items = set(seq['history_asin'] + [seq['target_asin']])

            # 负样本候选集：所有物品中排除用户交互过的
            neg_candidates = [item for item in all_items if item not in user_items]

            if len(neg_candidates) < neg_ratio:
                print(f"Warning: Not enough negative candidates for user {seq['user_id']}")
                continue

            # 采样19个负样本
            neg_samples = random.sample(neg_candidates, neg_ratio)

            # 构建偏好和非偏好字符串
            preference = []
            unpreference = []
            for asin, rating in zip(seq['history_asin'], seq['history_rating']):
                item_info = meta_data.get(asin, {})
                title = item_info.get('title', 'Unknown Item')
                if isinstance(title, list):
                    title = ' '.join(title)

                if rating == 1:
                    preference.append(f'"{title}"')
                else:
                    unpreference.append(f'"{title}"')

            preference_str = ", ".join(preference) if preference else "None"
            unpreference_str = ", ".join(unpreference) if unpreference else "None"

            # 正样本
            target_info = meta_data.get(seq['target_asin'], {})
            target_title = target_info.get('title', 'Unknown Item')
            if isinstance(target_title, list):
                target_title = ' '.join(target_title)

            json_list.append({
                "instruction": "Given the user's preference and unpreference, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                "input": f"User Preference: {preference_str}\nUser Unpreference: {unpreference_str}\nWhether the user will like the target item \"{target_title}\"?",
                "output": "Yes.",
                "item_id": item_to_idx.get(seq['target_asin'], -1),
                "label": 1
            })

            # 19个负样本
            for neg_asin in neg_samples:
                neg_info = meta_data.get(neg_asin, {})
                neg_title = neg_info.get('title', 'Unknown Item')
                if isinstance(neg_title, list):
                    neg_title = ' '.join(neg_title)

                json_list.append({
                    "instruction": "Given the user's preference and unpreference, identify whether the user will like the target item by answering \"Yes.\" or \"No.\".",
                    "input": f"User Preference: {preference_str}\nUser Unpreference: {unpreference_str}\nWhether the user will like the target item \"{neg_title}\"?",
                    "output": "No.",
                    "item_id": item_to_idx.get(neg_asin, -1),
                    "label": 0
                })

        return json_list

    print("\nCreating training samples with negatives (1:19)...")
    train_json = create_json_with_negatives(train_data, all_items, meta_data, item_to_idx, 19)

    print("Creating validation samples with negatives (1:19)...")
    valid_json = create_json_with_negatives(valid_data, all_items, meta_data, item_to_idx, 19)

    print("Creating test samples with negatives (1:19)...")
    test_json = create_json_with_negatives(test_data, all_items, meta_data, item_to_idx, 19)

    print("Saving files...")
    with open(f'{output_prefix}_train.json', 'w') as f:
        json.dump(train_json, f, indent=4)

    with open(f'{output_prefix}_valid.json', 'w') as f:
        json.dump(valid_json, f, indent=4)

    with open(f'{output_prefix}_test.json', 'w') as f:
        json.dump(test_json, f, indent=4)

    print(f"\nDataset statistics:")
    print(
        f"Train samples: {len(train_json)} (positive: {sum(1 for x in train_json if x['label'] == 1)}, negative: {sum(1 for x in train_json if x['label'] == 0)})")
    print(
        f"Valid samples: {len(valid_json)} (positive: {sum(1 for x in valid_json if x['label'] == 1)}, negative: {sum(1 for x in valid_json if x['label'] == 0)})")
    print(
        f"Test samples: {len(test_json)} (positive: {sum(1 for x in test_json if x['label'] == 1)}, negative: {sum(1 for x in test_json if x['label'] == 0)})")
    print(f"Negative ratio: 1:19")
    print(f"\nFiles saved:")
    print(f"  - {output_prefix}_train.json")
    print(f"  - {output_prefix}_valid.json")
    print(f"  - {output_prefix}_test.json")
    print(f"  - {output_prefix}_item_mapping.json")

if __name__ == "__main__":
    # 修改这里的路径
    base_path = "data/amazon_2018"
    rating_file = f"data/amazon_2018/Luxury_Beauty.json.gz"  # Amazon评分文件路径
    meta_file = "data/amazon_2018/meta_Luxury_Beauty.json.gz"  # Amazon元数据文件路径
    output_prefix = "./data/amazon_beauty"  # 输出文件前缀

    preprocess_amazon(rating_file, meta_file, output_prefix,
                      min_interactions=4, seq_len=10)