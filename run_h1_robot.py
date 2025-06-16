import sys
sys.path.append('.')
import torch
import numpy as np
from envs.h1.h1_env import H1Env
import time

print("🤖 加载训练好的H1机器人...")

# 加载训练好的actor模型
try:
    actor = torch.load('./experiments/h1/actor.pt', map_location='cpu')
    print("✓ 成功加载 actor.pt")
except:
    try:
        actor = torch.load('./experiments/h1/actor_19999.pt', map_location='cpu')
        print("✓ 成功加载 actor_19999.pt")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        exit()

actor.eval()

# 创建环境
print("🔧 创建H1环境...")
env = H1Env()

print("🚀 开始运行机器人! (按Ctrl+C停止)")
print("-" * 50)

episode = 1
try:
    while True:
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"📍 Episode {episode} 开始")
        
        for step in range(1000):  # 最多1000步
            # 使用训练好的策略
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = actor(obs_tensor).numpy()[0]
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # 渲染显示
            env.render()
            time.sleep(0.01)  # 稍微减慢速度以便观察
            
            if done:
                break
        
        print(f"✅ Episode {episode} 完成: {steps} 步, 奖励: {total_reward:.2f}")
        episode += 1
        time.sleep(1)  # episode间隔

except KeyboardInterrupt:
    print("\n🛑 用户停止运行")
    
print("👋 程序结束")
