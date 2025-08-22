#!/bin/bash

# 进入 diff-gaussian-rasterization 的 build 目录
cd submodules/diff-gaussian-rasterization/build/ || {
    echo "Error: 无法进入目录 submodules/diff-gaussian-rasterization/build/"
    exit 1
}

# 执行 cmake 构建命令
echo "开始根据你的改动进行动态构建项目..."
cmake --build . || {
    echo "Error: cmake 构建失败"
    exit 1
}

# 返回到原始目录（脚本执行前的目录）
cd ../../.. || {
    echo "Error: 无法返回上级目录"
    exit 1
}
echo "动态构建完成，开始测试"

echo "==========Running train script=========="
python train.py -s data/Hub -m data/Hub/output --eval --iterations 30000 --optimizer_type sparse_adam --test_iterations 10000 20000 30000 --save_iterations 30000

echo "==========Running render script=========="
python render.py -m ./data/Hub/output --eval

echo "==========Running metrics script=========="
python metrics.py -m ./data/Hub/output

echo "==========Done, 你真厉害!=========="
