import sys
sys.path.append('.')
import torch
import numpy as np
from envs.jvrc import JvrcStepEnv
import time

print("🦶 启动训练好的JVRC步进机器人...")

# 加载训练好的模型
try:
    actor = torch.load('./experiments/jvrc_step/actor.pt', map_location='cpu')
    print("✓ 成功加载 actor.pt")
except:
    # 尝试加载最新的检查点
    import glob
    actor_files = glob.glob('./experiments/jvrc_step/actor_*.pt')
    if actor_files:
        latest_actor = max(actor_files)
        actor = torch.load(latest_actor, map_location='cpu')
        print(f"✓ 成功加载 {latest_actor}")
    else:
        print("❌ 找不到训练好的模型")
        exit()

actor.eval()

# 创建JVRC步进环境
print("🔧 创建JVRC步进环境...")
env = JvrcStepEnv()

print("🚀 开始运行JVRC步进机器人!")
print("这个机器人会根据预定义的足迹进行步进移动")
print("-" * 60)

episode = 1
try:
    while True:
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"📍 Episode {episode} 开始")
        
        for step in range(2000):  # JVRC步进任务通常需要更多步数
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
            time.sleep(0.01)  # 控制播放速度
            
            # 每100步输出一次进度
            if step % 100 == 0:
                print(f"  步数: {step}, 当前奖励: {reward:.3f}")
            
            if done:
                break
        
        print(f"✅ Episode {episode} 完成:")
        print(f"   总步数: {steps}")
        print(f"   总奖励: {total_reward:.2f}")
        print(f"   平均奖励: {total_reward/steps:.3f}")
        
        episode += 1
        time.sleep(2)  # episode间隔

except KeyboardInterrupt:
    print("\n🛑 用户停止运行")
    
print("👋 程序结束")
