"""
系统健康检查脚本
用于验证各个组件是否正常工作
"""

import subprocess
import sys
import time
import socket
import requests
import redis
import json


def check_port(host, port):
    """检查端口是否开放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False


def check_docker_containers():
    """检查 Docker 容器状态"""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker 未运行或未安装")
            return False
        
        containers = result.stdout
        services = {
            'zookeeper': 'zookeeper',
            'kafka': 'kafka',
            'redis': 'redis',
            'etcd': 'etcd',
            'minio': 'minio',
            'milvus-standalone': 'milvus'
        }
        
        print("🔍 检查 Docker 容器状态:")
        for service, container_name in services.items():
            if container_name in containers:
                print(f"✅ {service} 服务正在运行")
            else:
                print(f"❌ {service} 服务未运行")
        
        return True
    except FileNotFoundError:
        print("❌ Docker 未安装")
        return False


def check_infrastructure():
    """检查基础设施状态"""
    print("\n🔍 检查基础设施状态:")
    
    # 检查端口
    ports = {
        2181: "Zookeeper",
        9092: "Kafka",
        6379: "Redis",
        2379: "Etcd",
        9000: "MinIO",
        19530: "Milvus"
    }
    
    for port, service in ports.items():
        if check_port('localhost', port):
            print(f"✅ {service} ({port}) 端口开放")
        else:
            print(f"❌ {service} ({port}) 端口未开放")
    
    # 检查 Redis 连接
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        print("✅ Redis 连接正常")
        
        # 检查一些特征键是否存在
        keys = r.keys('*')
        if keys:
            print(f"✅ Redis 中存在 {len(keys)} 个键")
            recent_keys = r.keys('*')[0:5]  # 显示前5个键
            print(f"   最近的键: {recent_keys}")
        else:
            print("ℹ️  Redis 中暂无数据")
        
        r.close()
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")


def check_python_dependencies():
    """检查 Python 依赖"""
    print("\n🔍 检查 Python 依赖:")
    
    dependencies = [
        ('json', 'built-in'),
        ('redis', 'redis'),
        ('kafka', 'kafka-python'),
        ('pymilvus', 'pymilvus'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('requests', 'requests'),
        ('pyflink', 'apache-flink')
    ]
    
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")


def main():
    print("🏥 RealTime-RecMaster 系统健康检查")
    print("="*50)
    
    check_docker_containers()
    check_infrastructure()
    check_python_dependencies()
    
    print("\n💡 运行建议:")
    print("1. 如果基础设施未启动，请运行: docker-compose up -d")
    print("2. 如果依赖未安装，请运行: pip install -r requirements.txt")
    print("3. 要开始数据生成，请运行: python data_generator.py")
    print("4. 要开始特征工程，请运行: python feature_engineering.py")
    print("5. 要监控 Redis，请运行: redis-cli MONITOR")


if __name__ == "__main__":
    main()