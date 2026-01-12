"""
压测脚本
用于 RealTime-RecMaster 项目的性能测试
"""

from locust import HttpUser, task, between
import json
import random
import requests


class InferenceUser(HttpUser):
    """
    模拟推理服务的用户行为
    """
    wait_time = between(0.1, 0.5)  # 用户等待时间（秒）
    
    def on_start(self):
        """
        用户开始时的初始化
        """
        # 模拟一些用户和物品ID
        self.user_ids = [f"user_{i}" for i in range(1, 1001)]  # 1000个用户
        self.item_ids = [f"item_{i}" for i in range(1, 5001)]  # 5000个物品
    
    @task(70)  # 70% 的请求是获取推荐
    def get_recommendations(self):
        """
        获取推荐列表
        """
        # 随机选择一个用户
        user_id = random.choice(self.user_ids)
        
        # 模拟请求数据
        data = {
            "user_id": user_id,
            "num_candidates": 500,
            "context_features": {
                "hour_of_day": random.randint(0, 23),
                "day_of_week": random.randint(0, 6),
                "device_type": random.choice(["mobile", "desktop", "tablet"]),
                "location": random.choice(["US", "CN", "EU", "JP"])
            }
        }
        
        # 发送请求到推理服务
        # 注意：这里需要根据实际部署的推理服务地址修改
        try:
            response = self.client.post(
                "/api/recommend",  # 推理服务的推荐API端点
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"成功获取推荐: {len(result.get('recommendations', []))} 个物品")
            else:
                print(f"推荐请求失败: {response.status_code}")
        except Exception as e:
            print(f"请求异常: {str(e)}")
    
    @task(20)  # 20% 的请求是获取用户特征
    def get_user_features(self):
        """
        获取用户特征
        """
        user_id = random.choice(self.user_ids)
        
        try:
            response = self.client.get(f"/api/features/user/{user_id}")
            
            if response.status_code == 200:
                features = response.json()
                print(f"成功获取用户特征: {len(features)} 个特征")
            else:
                print(f"用户特征请求失败: {response.status_code}")
        except Exception as e:
            print(f"请求异常: {str(e)}")
    
    @task(10)  # 10% 的请求是获取物品特征
    def get_item_features(self):
        """
        获取物品特征
        """
        item_id = random.choice(self.item_ids)
        
        try:
            response = self.client.get(f"/api/features/item/{item_id}")
            
            if response.status_code == 200:
                features = response.json()
                print(f"成功获取物品特征: {len(features)} 个特征")
            else:
                print(f"物品特征请求失败: {response.status_code}")
        except Exception as e:
            print(f"请求异常: {str(e)}")


# 如果直接运行此脚本，启动压测
if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py")