"""
实时特征工程脚本 - 使用 PyFlink 处理用户行为数据并计算实时特征

功能说明：
1. 时间特性与乱序处理：使用 EventTime 和 WatermarkStrategy 处理数据乱序
2. 实时计数特征：计算用户点击量和商品曝光量
3. 实时用户行为序列：维护用户最近点击的商品序列
4. 高性能 Redis Sink：使用连接池和 Pipeline 批量更新 Redis

依赖：
- PyFlink >= 1.17.1
- apache-flink == 1.17.1
- redis == 4.5.4
- pyflink-connectors (flink-sql-connector-kafka)

Flink 连接器依赖：
- flink-sql-connector-kafka
"""

from pyflink.datastream import StreamExecutionEnvironment, RuntimeContext
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.watermark_strategy import WatermarkStrategy
from pyflink.common import Time
import json
import redis
import logging
from datetime import datetime
from typing import Dict, Any
import traceback


class RedisSinkFunction:
    """
    自定义 Redis Sink 函数，用于将计算出的特征写入 Redis
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_client = None
        self.pipeline = None

    def open(self, runtime_context: RuntimeContext):
        """
        初始化 Redis 连接
        """
        # 创建 Redis 连接池
        pool = redis.ConnectionPool(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=True,
            max_connections=10
        )
        self.redis_client = redis.Redis(connection_pool=pool)
        
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def invoke(self, value, context=None):
        """
        处理单条记录并写入 Redis
        value: 包含特征计算结果的字典
        """
        try:
            # 开始 Redis Pipeline 操作
            pipe = self.redis_client.pipeline()
            
            # 解析特征类型和数据
            feature_type = value.get('feature_type')
            user_id = value.get('user_id')
            item_id = value.get('item_id')
            feature_value = value.get('feature_value')
            
            if feature_type == 'user_click_count_1m':
                # 用户 1 分钟点击量特征
                key = f"u_c_1m:{user_id}"
                pipe.setex(key, 120, feature_value)  # 设置 2 分钟过期
                
            elif feature_type == 'item_expose_count_5m':
                # 商品 5 分钟曝光量特征
                key = f"i_e_5m:{item_id}"
                pipe.setex(key, 600, feature_value)  # 设置 10 分钟过期
                
            elif feature_type == 'user_click_sequence':
                # 用户点击序列特征
                key = f"u_seq:{user_id}"
                pipe.lpush(key, item_id)  # 将新的 item_id 添加到列表头部
                pipe.ltrim(key, 0, 9)     # 保留最多 10 个项目
                pipe.expire(key, 3600)    # 设置 1 小时过期
                
            # 执行 Pipeline 操作
            pipe.execute()
            
        except Exception as e:
            self.logger.error(f"Error writing to Redis: {str(e)}, value: {value}")
            self.logger.error(traceback.format_exc())

    def close(self):
        """
        关闭 Redis 连接
        """
        if self.redis_client:
            self.redis_client.close()


def parse_timestamp(timestamp_str):
    """
    解析时间戳字符串为毫秒时间戳
    """
    try:
        dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
        return int(dt.timestamp() * 1000)
    except ValueError:
        try:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            return int(dt.timestamp() * 1000)
        except ValueError:
            # 如果解析失败，返回当前时间戳
            return int(datetime.now().timestamp() * 1000)


def process_user_behavior_stream():
    """
    主处理函数：从 Kafka 读取数据，计算特征并写入 Redis
    """
    # 创建执行环境
    env = StreamExecutionEnvironment.get_execution_environment()
    
    # 设置并行度
    env.set_parallelism(1)  # 为了演示目的，实际生产中可调整
    
    # 配置 Kafka 消费者
    kafka_props = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'feature_engineering_group'
    }
    
    # 创建 WatermarkStrategy 以处理事件时间和乱序数据
    watermark_strategy = WatermarkStrategy \
        .for_bounded_out_of_orderness(Time.seconds(5)) \
        .with_timestamp_assigner(lambda event, timestamp: parse_timestamp(event['timestamp']))
    
    # 创建 Kafka 消费者
    kafka_consumer = FlinkKafkaConsumer(
        topics='user_behavior',
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_props
    )
    
    # 添加水印策略
    kafka_consumer.set_start_from_latest()
    
    # 从 Kafka 读取数据流
    stream = env.add_source(kafka_consumer) \
        .map(lambda x: json.loads(x), output_type=Types.PICKLED_BYTE_ARRAY()) \
        .assign_timestamps_and_watermarks(watermark_strategy)
    
    # 处理数据流并计算特征
    def process_element(value):
        """
        处理单个元素并生成特征
        """
        try:
            # 解析 JSON 数据
            data = value
            
            user_id = data.get('user_id')
            item_id = data.get('item_id')
            behavior_type = data.get('behavior_type')
            
            # 生成特征结果列表
            results = []
            
            # 1. 用户 1 分钟点击量特征
            if behavior_type == 'click':
                results.append({
                    'feature_type': 'user_click_count_1m',
                    'user_id': user_id,
                    'item_id': None,
                    'feature_value': 1  # 这里简化处理，实际应聚合窗口统计
                })
                
                # 2. 用户点击序列特征
                results.append({
                    'feature_type': 'user_click_sequence',
                    'user_id': user_id,
                    'item_id': item_id,
                    'feature_value': None
                })
            
            # 3. 商品 5 分钟曝光量特征
            elif behavior_type == 'expose':
                results.append({
                    'feature_type': 'item_expose_count_5m',
                    'user_id': None,
                    'item_id': item_id,
                    'feature_value': 1  # 这里简化处理，实际应聚合窗口统计
                })
            
            return results
            
        except Exception as e:
            print(f"Error processing element: {str(e)}")
            print(f"Element: {value}")
            return []  # 返回空列表避免流中断
    
    # 对流应用处理函数
    processed_stream = stream.map(process_element) \
        .flat_map(lambda x: x)  # 展平列表
    
    # 添加 Redis Sink
    redis_sink = RedisSinkFunction(redis_host='localhost', redis_port=6379)
    processed_stream.add_sink(redis_sink)
    
    # 执行任务
    env.execute("Real-time Feature Engineering Job")


# 以下是一个替代实现，使用更完整的窗口聚合逻辑
def process_with_window_aggregation():
    """
    使用滑动窗口聚合的完整实现
    """
    from pyflink.datastream.window import SlidingEventTimeWindows, TumblingEventTimeWindows
    from pyflink.common.time import Time as FlinkTime
    
    env = StreamExecutionEnvironment.get_execution_environment()
    
    # 设置程序属性，指定连接器JAR包路径（如果需要）
    # env.get_config().set_global_job_parameters({'pipeline.jars': 'file:///path/to/flink-sql-connector-kafka-*.jar'})
    
    env.set_parallelism(1)
    
    # 配置 Kafka 消费者
    kafka_props = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'feature_engineering_sliding_window_group'
    }
    
    # Watermark 策略
    watermark_strategy = WatermarkStrategy \
        .for_bounded_out_of_orderness(Time.seconds(5)) \
        .with_timestamp_assigner(lambda event, timestamp: parse_timestamp(event['timestamp']))
    
    # Kafka 消费者
    kafka_consumer = FlinkKafkaConsumer(
        topics='user_behavior',
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_props
    )
    kafka_consumer.set_start_from_latest()
    
    # 从 Kafka 读取数据流
    stream = env.add_source(kafka_consumer) \
        .map(lambda x: json.loads(x), output_type=Types.PICKLED_BYTE_ARRAY()) \
        .assign_timestamps_and_watermarks(watermark_strategy)
    
    # 分离不同的行为类型
    clicks_stream = stream.filter(lambda x: x.get('behavior_type') == 'click')
    exposes_stream = stream.filter(lambda x: x.get('behavior_type') == 'expose')
    
    # 用户 1 分钟点击量滑动窗口聚合 (每10秒更新一次)
    user_click_counts = clicks_stream \
        .key_by(lambda x: x['user_id']) \
        .window(SlidingEventTimeWindows.of(
            size=FlinkTime.minutes(1),     # 窗口大小：1分钟
            slide=FlinkTime.seconds(10)    # 滑动步长：10秒
        )) \
        .aggregate(
            # 聚合函数：累加点击数
            lambda acc, value: acc + 1,
            lambda key, window, agg_result: {
                'feature_type': 'user_click_count_1m',
                'user_id': key,
                'item_id': None,
                'feature_value': agg_result
            },
            create_accumulator=lambda: 0
        )
    
    # 商品 5 分钟曝光量滑动窗口聚合 (每30秒更新一次)
    item_expose_counts = exposes_stream \
        .key_by(lambda x: x['item_id']) \
        .window(SlidingEventTimeWindows.of(
            size=FlinkTime.minutes(5),     # 窗口大小：5分钟
            slide=FlinkTime.seconds(30)    # 滑动步长：30秒
        )) \
        .aggregate(
            # 聚合函数：累加曝光数
            lambda acc, value: acc + 1,
            lambda key, window, agg_result: {
                'feature_type': 'item_expose_count_5m',
                'user_id': None,
                'item_id': key,
                'feature_value': agg_result
            },
            create_accumulator=lambda: 0
        )
    
    # 用户点击序列处理（无需窗口，直接处理）
    user_click_sequences = clicks_stream.map(
        lambda x: {
            'feature_type': 'user_click_sequence',
            'user_id': x['user_id'],
            'item_id': x['item_id'],
            'feature_value': None
        }
    )
    
    # 合并所有特征流
    all_features = user_click_counts.union(item_expose_counts, user_click_sequences)
    
    # 添加 Redis Sink
    redis_sink = RedisSinkFunction(redis_host='localhost', redis_port=6379)
    all_features.add_sink(redis_sink)
    
    # 执行任务
    env.execute("Real-time Feature Engineering with Sliding Window Aggregation Job")


if __name__ == '__main__':
    # 运行带滑动窗口聚合的版本
    process_with_window_aggregation()