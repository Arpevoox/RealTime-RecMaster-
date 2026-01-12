"""
实时样本流拼接器
用于 RealTime-RecMaster 项目第五阶段
监听 Kafka 中的曝光行为和点击行为，进行特征拼接并生成训练样本
"""

import json
import time
import threading
from kafka import KafkaConsumer, KafkaProducer
import redis
import logging
from datetime import datetime, timedelta
import pickle
import uuid
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleJoiner:
    """
    样本拼接器：监听曝光和点击流，拼接特征并生成训练样本
    """
    
    def __init__(self, kafka_bootstrap_servers='localhost:9092', redis_host='localhost', redis_port=6379):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=False)
        
        # Kafka 配置
        self.exposure_consumer = KafkaConsumer(
            'user_behavior',  # 假设曝光和点击都在同一个 topic 中，通过 behavior_type 区分
            bootstrap_servers=[kafka_bootstrap_servers],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='sample_joiner_group',
            auto_offset_reset='latest'
        )
        
        self.sample_producer = KafkaProducer(
            bootstrap_servers=[kafka_bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # 存储曝光信息的字典
        self.exposure_cache = {}
        
        # 设置 Redis 过期键监听器
        self.pubsub = self.redis_client.pubsub()
        # 监听 Redis 键过期事件
        self.pubsub.psubscribe('__keyevent@0__:expired')
        
        # 过期处理线程
        self.expired_handler_thread = threading.Thread(target=self.handle_expired_keys, daemon=True)
        self.expired_handler_thread.start()
        
        logger.info("样本拼接器初始化完成")
    
    def start_listening(self):
        """
        开始监听 Kafka 消息
        """
        logger.info("开始监听 Kafka 消息...")
        
        for message in self.exposure_consumer:
            try:
                behavior_data = message.value
                behavior_type = behavior_data.get('behavior_type', '')
                user_id = behavior_data.get('user_id', '')
                item_id = behavior_data.get('item_id', '')
                
                if behavior_type == 'expose':
                    # 处理曝光事件
                    self.handle_exposure(behavior_data)
                elif behavior_type in ['click', 'like', 'finish']:
                    # 处理点击事件
                    self.handle_click(behavior_data)
                    
            except Exception as e:
                logger.error(f"处理消息时出错: {e}")
    
    def handle_exposure(self, exposure_data):
        """
        处理曝光事件：将特征存储在 Redis 中，等待点击事件
        """
        user_id = exposure_data['user_id']
        item_id = exposure_data['item_id']
        key = f"{user_id}:{item_id}:{exposure_data.get('timestamp', datetime.now().isoformat())}"
        
        # 存储曝光特征到 Redis，5分钟过期
        feature_data = {
            'user_id': user_id,
            'item_id': item_id,
            'item_category': exposure_data['item_category'],
            'timestamp': exposure_data['timestamp'],
            'features': {
                'user_age': exposure_data.get('user_age', 25),
                'user_gender': exposure_data.get('user_gender', 0),
                'user_region': exposure_data.get('user_region', 1),
                'click_count_1m': exposure_data.get('click_count_1m', 0),
                'expose_count_5m': exposure_data.get('expose_count_5m', 0),
                'avg_stay_time': exposure_data.get('avg_stay_time', 0),
                'item_category_id': hash(exposure_data['item_category']) % 1000
            }
        }
        
        # 存储到 Redis，5分钟过期
        self.redis_client.setex(
            key.encode('utf-8'),
            300,  # 5分钟 = 300秒
            pickle.dumps(feature_data)
        )
        
        logger.debug(f"存储曝光数据: {key}")
    
    def handle_click(self, click_data):
        """
        处理点击事件：查找对应的曝光，拼接特征生成正样本
        """
        user_id = click_data['user_id']
        item_id = click_data['item_id']
        
        # 查找最近的曝光记录（这里简化为查找所有匹配的曝光）
        pattern = f"{user_id}:{item_id}:*".encode('utf-8')
        keys = self.redis_client.keys(pattern)
        
        if keys:
            # 找到匹配的曝光记录
            for key in keys:
                try:
                    exposure_data = pickle.loads(self.redis_client.get(key))
                    if exposure_data:
                        # 生成正样本 (Label=1)
                        training_sample = self.create_training_sample(exposure_data, click_data, label=1)
                        
                        # 发送到训练样本 topic
                        self.sample_producer.send('training_samples', value=training_sample)
                        logger.info(f"发送正样本: {training_sample['label']}")
                        
                        # 删除已处理的曝光记录
                        self.redis_client.delete(key)
                        
                except Exception as e:
                    logger.error(f"处理点击事件时出错: {e}")
        else:
            logger.debug(f"未找到用户 {user_id} 对商品 {item_id} 的曝光记录")
    
    def create_training_sample(self, exposure_data, click_data, label):
        """
        创建训练样本
        """
        sample = {
            'user_id': exposure_data['user_id'],
            'item_id': exposure_data['item_id'],
            'label': label,  # 1 表示正样本，0 表示负样本
            'features': exposure_data['features'],
            'timestamp': datetime.now().isoformat(),
            'behavior_type': click_data.get('behavior_type', 'expose') if label == 1 else 'expose',
            'stay_time': click_data.get('stay_time', 0) if label == 1 else 0
        }
        return sample
    
    def handle_expired_keys(self):
        """
        处理过期的 Redis 键，生成负样本
        """
        logger.info("开始监听 Redis 键过期事件...")
        
        for message in self.pubsub.listen():
            try:
                if message['type'] == 'pmessage':
                    expired_key = message['data'].decode('utf-8')
                    logger.debug(f"检测到过期键: {expired_key}")
                    
                    # 由于键已过期，我们需要从键名中提取信息
                    # 键格式: user_id:item_id:timestamp
                    parts = expired_key.split(':')
                    if len(parts) >= 3:
                        user_id = parts[0]
                        item_id = parts[1]
                        
                        # 尝试从过期键重建特征（实际应用中可能需要更复杂的机制）
                        # 这里我们简单地生成一个负样本
                        sample = {
                            'user_id': user_id,
                            'item_id': item_id,
                            'label': 0,  # 未点击，负样本
                            'features': {
                                'user_age': 25,  # 默认值
                                'user_gender': 0,  # 默认值
                                'user_region': 1,  # 默认值
                                'click_count_1m': 0,  # 默认值
                                'expose_count_5m': 0,  # 默认值
                                'avg_stay_time': 0,  # 默认值
                                'item_category_id': hash('unknown') % 1000  # 默认值
                            },
                            'timestamp': datetime.now().isoformat(),
                            'behavior_type': 'expose',
                            'stay_time': 0
                        }
                        
                        # 发送到训练样本 topic
                        self.sample_producer.send('training_samples', value=sample)
                        logger.info(f"发送负样本 (过期曝光): {sample['label']}")
                        
            except Exception as e:
                logger.error(f"处理过期键时出错: {e}")
    
    def cleanup(self):
        """
        清理资源
        """
        if self.exposure_consumer:
            self.exposure_consumer.close()
        if self.sample_producer:
            self.sample_producer.close()
        if self.pubsub:
            self.pubsub.close()


def main():
    """
    主函数：启动样本拼接器
    """
    logger.info("=== RealTime-RecMaster: 实时样本流拼接器 ===")
    
    joiner = SampleJoiner()
    
    try:
        joiner.start_listening()
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭...")
        joiner.cleanup()


if __name__ == "__main__":
    main()