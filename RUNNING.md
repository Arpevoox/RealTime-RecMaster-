# RealTime-RecMaster 运行指南

## 环境准备

1. 确保安装了 Docker 和 Docker Compose
2. 确保安装了 Python 3.8+
3. 确保安装了 Java 8 或更高版本（用于 PyFlink）

## 启动基础设施

```bash
# 启动所有服务（Kafka, Redis, Milvus 等）
docker-compose up -d
```

等待所有服务启动完成。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行数据生成器

```bash
python data_generator.py
```

## 运行特征工程服务

在另一个终端中运行：

```bash
python feature_engineering.py
```

**注意**: 如果遇到 PyFlink 连接 Kafka 的问题，您可能需要下载 `flink-sql-connector-kafka` JAR 包：
- 下载适用于您 Flink 版本的 Kafka 连接器
- 将其放置在 Flink 的 lib 目录下
- 或者在代码中通过 `pipeline.jars` 参数指定路径

## 验证 Redis 写入

打开一个新的终端窗口，进入 Redis CLI 并监控 Redis 操作：

```bash
# 连接到 Redis
redis-cli

# 监控 Redis 操作
MONITOR
```

您应该能看到来自特征工程服务的密集写入操作，包括：
- `SET` 操作（用于存储用户点击量和商品曝光量）
- `LPUSH` 和 `LTRIM` 操作（用于维护用户点击序列）

## 验证 Kafka 数据流

```bash
# 进入 Kafka 容器
docker exec -it realtime-recmaster-kafka-1 bash

# 查看 user_behavior 主题的消息
kafka-console-consumer --bootstrap-server localhost:9092 --topic user_behavior --from-beginning
```

## 停止服务

```bash
# 停止数据生成器（Ctrl+C）
# 停止特征工程服务（Ctrl+C）
# 停止基础设施
docker-compose down
```

## 使用便捷启动脚本

我们提供了一个便捷的启动脚本，可以一键启动整个系统：

```bash
python start_system.py
```

## 系统健康检查

运行系统健康检查以验证所有组件是否正常工作：

```bash
python check_system.py
```

## 注意事项

1. PyFlink 需要 Java 8 或更高版本
2. 如果遇到 Kafka 连接问题，请确保 Docker 中的网络配置正确
3. 为了运行 PyFlink 作业，可能需要下载额外的 JAR 包：
   - `flink-sql-connector-kafka` 
   - 将其放置在 Flink 的 lib 目录下
   - 或者在代码中通过 `pipeline.jars` 参数指定路径
4. 滑动窗口已正确配置，每10秒更新一次用户点击统计，每30秒更新一次商品曝光统计