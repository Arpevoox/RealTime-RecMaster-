# RealTime-RecMaster - 万亿级实时推荐系统

万亿级实时推荐系统，采用业界领先的架构设计，包含完整的推荐链路：从数据仿真、特征工程、召回、排序到在线学习闭环。

## 项目结构

```
RealTime-RecMaster/
├── deployment/             # 基础设施部署
│   └── docker-compose.yml
├── data_stream/            # 数据仿真与采集
│   └── data_generator.py
├── feature_engineering/    # Flink 实时特征工程
│   └── feature_engineering.py
├── recall/                 # 召回层 (GNN + Milvus)
│   ├── gnn_model.py        # GNN 模型定义 (GraphSAGE, BPR Loss, Inductive 推理)
│   └── milvus_client.py    # 向量存储与检索 (HNSW 索引)
├── ranking/                # 排序层 (MMoE/PLE)
│   └── ranking_mmoe.py     # MMoE 多目标精排模型 (动态权重, ESMM 架构)
├── online_learning/        # 在线学习闭环
│   ├── sample_joiner.py    # 实时样本拼接
│   └── online_trainer.py   # 增量训练器与模型热加载
├── serving/                # 推理服务与压测
│   ├── inference_server.py # 高性能推理服务 (Locality, 并发执行, 模型优化)
│   └── locustfile.py       # 压力测试脚本
├── requirements.txt        # 依赖列表
├── .gitignore              # 忽略文件
└── README.md               # 项目说明文档
```

## 核心特性

### 1. 基础设施 (Deployment)
- **Kafka/Zookeeper**: 承载用户行为日志流
- **Redis**: 作为实时特征库 (Feature Store)
- **Milvus**: 向量数据库，支持 HNSW 索引实现高效相似性搜索

### 2. 数据仿真 (Data Stream)
- **行为模拟**: 生成用户点击、曝光、点赞、完播等行为日志
- **业务逻辑**: 模拟用户偏好，小 ID 用户对特定品类有更高点击率
- **持续运行**: 支持长时间数据生成，用于系统测试

### 3. 实时特征工程 (Feature Engineering)
- **滑动窗口**: 每10秒计算一次过去1分钟的用户行为统计
- **乱序处理**: 使用 EventTime 和 Watermark 处理数据乱序
- **高性能**: Redis 连接池和 Pipeline 批量提交，支撑万亿级吞吐

### 4. 异构图召回 (Recall)
- **GraphSAGE**: 基于图神经网络的召回模型
- **Inductive 推理**: 支持冷启动，无需重新训练即可为新物品生成 Embedding
- **HNSW 索引**: Milvus 中的高效相似性搜索，亿级商品毫秒级召回

### 5. 多目标精排 (Ranking)
- **MMoE 架构**: 多专家多任务学习，共享知识同时保持任务独立性
- **动态权重**: Uncertainty Weighting 自适应平衡不同任务损失
- **ESMM 架构**: 解决 CTR/CVR 的选择偏差问题

### 6. 在线学习 (Online Learning)
- **样本拼接**: Joiner 实时拼接曝光和点击样本
- **增量训练**: Online Trainer 实时更新模型参数
- **热加载**: 模型权重零停机更新

### 7. 高性能推理 (Serving)
- **Locality 优化**: LRU 缓存高频物品特征，减少 Redis 查询
- **并发执行**: asyncio.gather 同时启动 Redis 和 Milvus 查询
- **模型优化**: PyTorch JIT 和批处理提升推理速度
- **预取机制**: 异步预取推荐物品详细信息

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd RealTime-RecMaster

# 安装依赖
pip install -r requirements.txt

# 启动基础设施
docker-compose -f deployment/docker-compose.yml up -d
```

### 2. 启动数据仿真

```bash
# 在 data_stream 目录下运行
cd data_stream
python data_generator.py
```

### 3. 启动实时特征工程

```bash
# 在 feature_engineering 目录下运行
cd feature_engineering
python feature_engineering.py
```

### 4. 训练召回模型

```bash
# 在 recall 目录下运行
cd recall
python gnn_model.py
```

### 5. 训练精排模型

```bash
# 在 ranking 目录下运行
cd ranking
python ranking_mmoe.py
```

### 6. 启动在线学习

```bash
# 启动样本拼接器
cd online_learning
python sample_joiner.py

# 启动在线训练器
python online_trainer.py
```

### 7. 启动推理服务

```bash
# 在 serving 目录下运行
cd serving
python inference_server.py
```

## 压力测试

```bash
# 安装 Locust
pip install locust

# 运行压测
cd serving
locust -f locustfile.py
```

## 技术亮点

### 1. 计算局部性优化
- 高频物品特征缓存在推理服务内存中
- 使用 LRU Cache 减少 Redis 查询延迟

### 2. 并发执行策略
- 使用 asyncio.gather 同时执行 Redis 和 Milvus 查询
- 显著降低整体响应时间

### 3. 模型优化技术
- PyTorch JIT 编译加速模型推理
- ONNX Runtime/TensorRT 进一步压缩推理耗时

### 4. 预取机制
- 异步预取推荐结果的详细信息
- 提升用户体验，降低感知延迟

### 5. 混合召回策略
- 协同过滤 + 向量召回的结合
- 多路召回结果去重和融合排序

## 架构设计原则

1. **高可用性**: 微服务架构，各组件独立部署
2. **高性能**: 异步处理，批量操作，缓存优化
3. **可扩展性**: 水平扩展能力，支持万亿级数据
4. **实时性**: 流式处理，秒级特征更新
5. **鲁棒性**: 完善的异常处理和监控

## 项目贡献

本项目展示了现代推荐系统的完整技术栈，涵盖了从数据流处理到深度学习模型的各个方面，适合学习和实践大规模推荐系统的设计与实现。

## 许可证

Apache License 2.0