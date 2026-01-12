"""
启动 RealTime-RecMaster 系统的便捷脚本
"""

import subprocess
import sys
import os
import threading
import time
import signal
import atexit


def start_infrastructure():
    """启动基础设施"""
    print("🚀 启动基础设施 (Docker Compose)...")
    try:
        result = subprocess.run(['docker-compose', 'up', '-d'], 
                              cwd=os.path.dirname(__file__), 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 基础设施启动成功")
            return True
        else:
            print(f"❌ 基础设施启动失败: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Docker 或 docker-compose 未安装或未在 PATH 中")
        return False
    except Exception as e:
        print(f"❌ 启动基础设施时发生错误: {e}")
        return False


def install_dependencies():
    """安装 Python 依赖"""
    print("📦 安装 Python 依赖...")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                              cwd=os.path.dirname(__file__), 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 依赖安装成功")
            return True
        else:
            print(f"❌ 依赖安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 安装依赖时发生错误: {e}")
        return False


def run_data_generator():
    """运行数据生成器"""
    print("🔄 启动数据生成器...")
    try:
        process = subprocess.Popen([sys.executable, 'data_generator.py'], 
                                 cwd=os.path.dirname(__file__))
        return process
    except Exception as e:
        print(f"❌ 启动数据生成器时发生错误: {e}")
        return None


def run_feature_engineering():
    """运行特征工程"""
    print("⚙️ 启动特征工程服务...")
    try:
        process = subprocess.Popen([sys.executable, 'feature_engineering.py'], 
                                 cwd=os.path.dirname(__file__))
        return process
    except Exception as e:
        print(f"❌ 启动特征工程服务时发生错误: {e}")
        return None


def monitor_redis():
    """监控 Redis"""
    print("🔍 启动 Redis 监控...")
    try:
        process = subprocess.Popen(['redis-cli', 'MONITOR'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f"❌ 启动 Redis 监控时发生错误: {e}")
        print("💡 提示: 如果 redis-cli 未找到，请确保 Redis 已安装并在 PATH 中")
        return None


def main():
    print("🏥 启动 RealTime-RecMaster 系统")
    print("="*50)
    
    # 安装依赖
    if not install_dependencies():
        print("❌ 无法安装依赖，退出")
        return
    
    # 启动基础设施
    if not start_infrastructure():
        print("❌ 无法启动基础设施，退出")
        return
    
    print("\n⏳ 等待基础设施启动 (预计需要 30 秒)...")
    time.sleep(30)  # 等待基础设施完全启动
    
    # 启动数据生成器
    data_gen_process = run_data_generator()
    if data_gen_process is None:
        print("⚠️ 数据生成器启动失败")
    
    # 启动特征工程
    feature_eng_process = run_feature_engineering()
    if feature_eng_process is None:
        print("⚠️ 特征工程服务启动失败")
    
    print("\n✅ 系统已启动!")
    print("📊 数据生成器和特征工程服务正在运行")
    print("💡 按 Ctrl+C 停止所有服务")
    
    try:
        # 等待用户中断
        while True:
            time.sleep(1)
            # 检查进程是否还在运行
            if data_gen_process and data_gen_process.poll() is not None:
                print("⚠️ 数据生成器已停止")
                break
            if feature_eng_process and feature_eng_process.poll() is not None:
                print("⚠️ 特征工程服务已停止")
                break
    except KeyboardInterrupt:
        print("\n🛑 正在停止系统...")
        
        # 终止子进程
        if data_gen_process:
            data_gen_process.terminate()
        if feature_eng_process:
            feature_eng_process.terminate()
        
        print("✅ 系统已停止")
    
    print("\n 若要完全关闭基础设施，请运行: docker-compose down")


if __name__ == "__main__":
    main()