import json
import random
import time
from datetime import datetime
from kafka import KafkaProducer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, kafka_server='localhost:9092'):
        """
        初始化数据生成器
        :param kafka_server: Kafka服务器地址
        """
        self.kafka_server = kafka_server
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_server],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            api_version=(0, 10, 1)
        )
        # 定义商品类别
        self.categories = ['electronics', 'clothing', 'books', 'home', 'sports', 'beauty', 'food', 'automotive']
        
        # 为不同用户群体设置偏好，较小的用户ID对特定类别有更高点击率
        self.user_preferences = {}
        for i in range(1, 1001):
            # 用户ID越小，对前几个类别的偏好越高
            if i <= 250:
                self.user_preferences[f'user_{i}'] = ['electronics', 'automotive']
            elif i <= 500:
                self.user_preferences[f'user_{i}'] = ['clothing', 'beauty']
            elif i <= 750:
                self.user_preferences[f'user_{i}'] = ['books', 'home']
            else:
                self.user_preferences[f'user_{i}'] = ['sports', 'food']

    def generate_user_behavior(self):
        """
        生成用户行为数据
        :return: 用户行为字典
        """
        user_id = f'user_{random.randint(1, 1000)}'
        item_id = f'item_{random.randint(1, 5000)}'
        
        # 根据用户偏好调整行为类型
        preferred_categories = self.user_preferences[user_id]
        # 有一定概率选择偏好类别
        if random.random() < 0.3:  # 30% 概率选择偏好类别
            item_category = random.choice(preferred_categories)
        else:
            item_category = random.choice(self.categories)
        
        # 行为类型及其权重（模拟真实场景）
        behavior_weights = [('expose', 0.6), ('click', 0.25), ('like', 0.1), ('finish', 0.05)]
        behavior_type = random.choices(
            population=[behavior[0] for behavior in behavior_weights],
            weights=[behavior[1] for behavior in behavior_weights]
        )[0]
        
        # 停留时间（秒），根据行为类型调整
        if behavior_type == 'expose':
            stay_time = random.randint(0, 10)  # 曝光停留时间较短
        elif behavior_type == 'click':
            stay_time = random.randint(5, 30)  # 点击后通常会浏览一段时间
        elif behavior_type == 'like':
            stay_time = random.randint(10, 60)  # 点赞通常表示更深入的参与
        else:  # finish（观看完成）
            stay_time = random.randint(30, 120)  # 完成行为通常伴随着较长的停留时间
        
        behavior_data = {
            'user_id': user_id,
            'item_id': item_id,
            'behavior_type': behavior_type,
            'stay_time': stay_time,
            'item_category': item_category,
            'timestamp': datetime.now().isoformat()
        }
        
        return behavior_data

    def send_to_kafka(self, topic='user_behavior', interval=0.1):
        """
        持续生成数据并发送到Kafka
        :param topic: Kafka主题名称
        :param interval: 发送间隔（秒）
        """
        logger.info(f"开始向Kafka主题 '{topic}' 发送数据...")
        logger.info("按 Ctrl+C 停止发送")
        
        try:
            while True:
                behavior_data = self.generate_user_behavior()
                
                # 发送消息到Kafka
                self.producer.send(topic, value=behavior_data)
                
                logger.info(f"已发送: {behavior_data}")
                
                # 控制发送速率
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("停止发送数据")
        finally:
            self.producer.close()

if __name__ == "__main__":
    # 创建数据生成器实例
    generator = DataGenerator()
    
    # 启动数据发送
    # 可以通过修改interval参数来控制发送频率
    generator.send_to_kafka(topic='user_behavior', interval=0.5)