"""
在线训练器
用于 RealTime-RecMaster 项目第五阶段
实时消费训练样本，增量训练 MMoE 模型并实现模型热加载
"""

import json
import time
import threading
from kafka import KafkaConsumer
import redis
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
import pickle
from ranking_mmoe import MMoE  # 导入第四阶段的 MMoE 模型

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnlineTrainer:
    """
    在线训练器：实时消费训练样本，增量训练模型
    """
    
    def __init__(self, kafka_bootstrap_servers='localhost:9092', redis_host='localhost', redis_port=6379):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        
        # Kafka 配置
        self.training_consumer = KafkaConsumer(
            'training_samples',
            bootstrap_servers=[kafka_bootstrap_servers],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='online_trainer_group',
            auto_offset_reset='latest'
        )
        
        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.optimizer = torch.optim.Adam([
            {'params': self.model.embedding_layers.parameters(), 'lr': 0.001},  # Embedding层
            {'params': self.model.task_layers.parameters(), 'lr': 0.001},      # Task层（Towers）
            {'params': self.model.feature_interaction.parameters(), 'lr': 0.0005}, # 特征交互层
        ])
        
        # 存储最近的样本用于指标计算
        self.sample_buffer = []
        self.max_buffer_size = 1000
        
        # 批处理参数
        self.batch_size = 50
        self.sample_count = 0
        
        # 指标跟踪
        self.loss_history = []
        self.auc_history = []
        
        logger.info("在线训练器初始化完成")
    
    def init_model(self):
        """
        初始化 MMoE 模型
        """
        model = MMoE(
            discrete_feature_sizes=[1000, 5000, 8],  # user_id, item_id, category
            continuous_feature_size=6,  # 连续特征数量
            num_experts=4,
            num_tasks=3  # CTR, WatchTime, CVR
        )
        
        # 冻结专家网络层，只训练 Embedding 和 Task 层
        for expert in model.experts:
            for param in expert.parameters():
                param.requires_grad = False
        
        # 冻结门控网络，只训练 Embedding 和 Task 层
        for gate in model.gate_networks:
            for param in gate.parameters():
                param.requires_grad = False
        
        model.to(self.device)
        return model
    
    def preprocess_sample(self, sample):
        """
        预处理样本
        """
        features = sample['features']
        
        # 离散特征
        discrete_features = torch.tensor([
            int(hash(sample['user_id']) % 1000),      # user_id
            int(hash(sample['item_id']) % 5000),      # item_id
            int(features['item_category_id'])         # category
        ], dtype=torch.long).unsqueeze(0)
        
        # 连续特征
        continuous_features = torch.tensor([
            features['user_age'],
            features['user_gender'],
            features['user_region'],
            features['click_count_1m'],
            features['expose_count_5m'],
            features['avg_stay_time']
        ], dtype=torch.float).unsqueeze(0)
        
        # 标签
        label = torch.tensor([sample['label']], dtype=torch.float)
        stay_time = torch.tensor([sample['stay_time']], dtype=torch.float)
        
        return discrete_features, continuous_features, label, stay_time
    
    def compute_loss(self, task_outputs, true_labels, true_stay_times):
        """
        计算损失
        """
        ctr_pred = task_outputs[0].squeeze()
        watch_time_pred = task_outputs[1].squeeze()
        cvr_pred = task_outputs[2].squeeze()
        
        # CTR 任务损失
        ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred, true_labels)
        
        # WatchTime 任务损失
        watch_time_loss = F.mse_loss(watch_time_pred, true_stay_times)
        
        # CVR 任务损失 (只对点击样本计算)
        clicked_mask = (true_labels == 1).float()
        if clicked_mask.sum() > 0:
            cvr_loss = F.binary_cross_entropy_with_logits(cvr_pred * clicked_mask, 
                                                          (true_stay_times > 10).float() * clicked_mask)
        else:
            cvr_loss = torch.tensor(0.0, device=self.device)
        
        # 组合损失
        total_loss = ctr_loss + 0.5 * watch_time_loss + 0.5 * cvr_loss
        
        return total_loss, ctr_loss, watch_time_loss, cvr_loss
    
    def update_metrics(self, task_outputs, true_labels):
        """
        更新指标
        """
        with torch.no_grad():
            ctr_pred = torch.sigmoid(task_outputs[0].squeeze()).cpu().numpy()
            true_labels_np = true_labels.cpu().numpy()
            
            try:
                auc = roc_auc_score(true_labels_np, ctr_pred)
                self.auc_history.append(auc)
            except ValueError:
                # 如果只有一个类别，无法计算 AUC
                self.auc_history.append(0.5)
    
    def train_step(self, batch_samples):
        """
        单步训练
        """
        if not batch_samples:
            return
        
        self.model.train()
        
        # 准备批量数据
        discrete_features_list = []
        continuous_features_list = []
        labels_list = []
        stay_times_list = []
        
        for sample in batch_samples:
            discrete_f, continuous_f, label, stay_time = self.preprocess_sample(sample)
            discrete_features_list.append(discrete_f.squeeze(0))
            continuous_features_list.append(continuous_f.squeeze(0))
            labels_list.append(label)
            stay_times_list.append(stay_time)
        
        # 合并批量
        discrete_features = torch.stack(discrete_features_list).to(self.device)
        continuous_features = torch.stack(continuous_features_list).to(self.device)
        labels = torch.stack(labels_list).to(self.device).squeeze(1)
        stay_times = torch.stack(stay_times_list).to(self.device).squeeze(1)
        
        # 前向传播
        task_outputs, _ = self.model(discrete_features, continuous_features)
        
        # 计算损失
        total_loss, ctr_loss, watch_time_loss, cvr_loss = self.compute_loss(task_outputs, labels, stay_times)
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 更新参数
        self.optimizer.step()
        
        # 更新指标
        self.update_metrics(task_outputs, labels)
        self.loss_history.append(total_loss.item())
        
        # 保持历史记录在合理范围内
        if len(self.loss_history) > self.max_buffer_size:
            self.loss_history = self.loss_history[-self.max_buffer_size:]
        if len(self.auc_history) > self.max_buffer_size:
            self.auc_history = self.auc_history[-self.max_buffer_size:]
        
        logger.info(f"批次训练完成 - Loss: {total_loss.item():.4f}, "
                   f"CTR Loss: {ctr_loss.item():.4f}, "
                   f"WatchTime Loss: {watch_time_loss.item():.4f}, "
                   f"CVR Loss: {cvr_loss.item():.4f}")
    
    def save_model_weights(self):
        """
        保存模型权重到 Redis
        """
        try:
            # 序列化模型权重
            model_state_dict = self.model.state_dict()
            serialized_weights = pickle.dumps({
                'model_state_dict': model_state_dict,
                'timestamp': datetime.now().isoformat(),
                'version': f"v{int(time.time())}"
            })
            
            # 存储到 Redis
            self.redis_client.set('mmoe_model_weights', serialized_weights)
            logger.info("模型权重已保存到 Redis")
        except Exception as e:
            logger.error(f"保存模型权重时出错: {e}")
    
    def print_metrics(self):
        """
        打印最近的指标
        """
        if self.loss_history:
            avg_loss = np.mean(self.loss_history[-100:])  # 最近100个样本的平均损失
            logger.info(f"最近100个样本平均 Loss: {avg_loss:.4f}")
        
        if self.auc_history:
            avg_auc = np.mean(self.auc_history[-100:])  # 最近100个样本的平均 AUC
            logger.info(f"最近100个样本平均 AUC: {avg_auc:.4f}")
    
    def start_training(self):
        """
        开始在线训练
        """
        logger.info("开始在线训练...")
        
        batch_samples = []
        
        for message in self.training_consumer:
            try:
                sample = message.value
                self.sample_buffer.append(sample)
                
                if len(self.sample_buffer) > self.max_buffer_size:
                    self.sample_buffer = self.sample_buffer[-self.max_buffer_size:]
                
                batch_samples.append(sample)
                self.sample_count += 1
                
                # 每积累 50 个样本就执行一次训练步骤
                if len(batch_samples) >= self.batch_size:
                    self.train_step(batch_samples)
                    batch_samples = []  # 清空批次
                    
                    # 每训练 10 个批次保存一次权重
                    if self.sample_count % (self.batch_size * 10) == 0:
                        self.save_model_weights()
                
                # 每 100 个样本打印一次指标
                if self.sample_count % 100 == 0:
                    self.print_metrics()
                    
            except Exception as e:
                logger.error(f"处理训练样本时出错: {e}")
    
    def cleanup(self):
        """
        清理资源
        """
        if self.training_consumer:
            self.training_consumer.close()


class ModelHotSwapper:
    """
    模型热加载器：定期检查 Redis 中的模型权重并更新在线模型
    """
    
    def __init__(self, model, redis_host='localhost', redis_port=6379, check_interval=30):
        self.model = model
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=False)
        self.check_interval = check_interval  # 检查间隔（秒）
        self.last_version = None
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self.monitor_weights, daemon=True)
        self.monitor_thread.start()
        
        logger.info("模型热加载器初始化完成")
    
    def monitor_weights(self):
        """
        监控 Redis 中的模型权重更新
        """
        logger.info(f"开始监控模型权重更新，检查间隔: {self.check_interval}秒")
        
        while True:
            try:
                # 从 Redis 获取模型权重
                serialized_weights = self.redis_client.get('mmoe_model_weights')
                
                if serialized_weights:
                    weights_data = pickle.loads(serialized_weights)
                    current_version = weights_data.get('version', 'unknown')
                    
                    # 检查版本是否更新
                    if current_version != self.last_version:
                        logger.info(f"检测到模型权重更新: {current_version}")
                        
                        # 加载新权重
                        self.model.load_state_dict(weights_data['model_state_dict'])
                        self.last_version = current_version
                        
                        logger.info(f"模型权重已更新: {current_version}")
                
                # 等待下一个检查周期
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"监控模型权重时出错: {e}")
                time.sleep(self.check_interval)


def main():
    """
    主函数：启动在线训练器和模型热加载器
    """
    logger.info("=== RealTime-RecMaster: 在线学习与模型实时更新闭环 ===")
    
    # 初始化在线训练器
    trainer = OnlineTrainer()
    
    # 初始化模型热加载器
    hot_swapper = ModelHotSwapper(trainer.model)
    
    try:
        trainer.start_training()
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭...")
        trainer.cleanup()


if __name__ == "__main__":
    main()