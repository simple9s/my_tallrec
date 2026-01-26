# 使用方法

## 1. 预处理数据

```bash
# 电影
python preprocess_movie_candidates.py

# 书籍
python preprocess_book_candidates.py

# Amazon（需先下载.json.gz文件）
python preprocess_amazon_candidates.py
```

## 2. 训练

```bash
bash train.sh GPU_ID SEED BASE_MODEL CANDIDATES DATASET
```

**示例：**
```bash
# 电影 + LLaMA 3.2B + 20候选
bash train.sh 0 42 meta-llama/Llama-3.2-3B-Instruct 20 movie

# 书籍 + OPT 6.7B + 100候选
bash train.sh 0 42 facebook/opt-6.7b 100 book

# Luxury Beauty + LLaMA + 20候选
bash train.sh 0 42 meta-llama/Llama-3.2-3B-Instruct 20 luxury_beauty

# Software + OPT + 100候选
bash train.sh 0 42 facebook/opt-6.7b 100 software
```

## 3. 评估

```bash
bash evaluate.sh GPU_ID BASE_MODEL LORA_WEIGHTS CANDIDATES DATASET
```

**示例：**
```bash
# 电影
bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/movie_20_seed42 20 movie

# Luxury Beauty
bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/luxury_beauty_20_seed42 20 luxury_beauty
```

## 参数

**DATASET选项：** movie, book, luxury_beauty, software  
**CANDIDATES选项：** 20（1:19比例）, 100（1:99比例）  
**BASE_MODEL选项：** meta-llama/Llama-3.2-3B-Instruct, facebook/opt-6.7b

## 评估指标

- Hit@1, Hit@5, Hit@10, Hit@20
- NDCG@1, NDCG@5, NDCG@10, NDCG@20

结果保存在：`{LORA_WEIGHTS}/results.json`