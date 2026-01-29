#!/bin/bash
apt install aria2 -y
# 定义要下载的文件URL列表
urls=(
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Luxury_Beauty.json.gz"
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Luxury_Beauty.json.gz"
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Software.json.gz"
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Software.json.gz"
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Video_Games.json.gz"
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz"
)

# 使用 aria2 并发下载所有文件
for url in "${urls[@]}"; do
    aria2c -x 16 -s 16 "$url" &
done

# 等待所有后台下载任务完成
wait
echo "所有文件下载完成！"