# Amazon推荐系统 - 对比学习

## 数据集

支持3个数据集，每个数据集生成20和100候选两种测试集：
- `luxury_beauty` - Luxury Beauty
- `software` - Software
- `video_games` - Video Games

## 使用

### 1. 数据预处理

```bash
# 下载Amazon数据（.json.gz文件）放到当前目录

# 处理Luxury Beauty - 生成20和100候选测试集
python preprocess_amazon.py --category Luxury_Beauty --neg_ratio 19

# 处理Software
python preprocess_amazon.py --category Software --neg_ratio 19

# 处理Video Games
python preprocess_amazon.py --category Video_Games --neg_ratio 19
```

生成文件：
```
./data/train_luxury_beauty.json
./data/valid_luxury_beauty.json  
./data/test_luxury_beauty_20.json
./data/test_luxury_beauty_100.json
```

### 2. 训练

```bash
# 格式：bash train.sh GPU_ID SEED BASE_MODEL DATASET
bash train.sh 0 42 meta-llama/Llama-3.2-3B-Instruct luxury_beauty
```

### 3. 评估

```bash
# 格式：bash evaluate.sh GPU_ID BASE_MODEL LORA_WEIGHTS DATASET CANDIDATES
# 20候选
bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/luxury_beauty_seed42 luxury_beauty 20

# 100候选
bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/luxury_beauty_seed42 luxury_beauty 100
```

### 快速开始

```bash
# 一键运行（数据预处理+训练+评估20候选）
bash quickstart.sh luxury_beauty 20

# 100候选
bash quickstart.sh luxury_beauty 100
```

## 结果

结果保存在：
- `./output/luxury_beauty_seed42/results_20.json` - 20候选指标
- `./output/luxury_beauty_seed42/results_100.json` - 100候选指标
- `./output/luxury_beauty_seed42/results_20_debug.json` - 详细结果

## 指标

- Hit@1, Hit@5, Hit@10, Hit@20
- NDCG@1, NDCG@5, NDCG@10, NDCG@20

预期（20候选）：
- Hit@1: 0.35+
- Hit@5: 0.65+
- NDCG@5: 0.50+


# 1. 预处理（生成20和100候选测试集）
python preprocess_amazon.py --category Luxury_Beauty --neg_ratio 19

# 2. 训练
bash train.sh 0 42 meta-llama/Llama-3.2-3B-Instruct luxury_beauty

# 3. 评估20候选
bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/luxury_beauty_seed42 luxury_beauty 20

# 4. 评估100候选
bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/luxury_beauty_seed42 luxury_beauty 100

# 或一键运行
bash quickstart.sh luxury_beauty 20