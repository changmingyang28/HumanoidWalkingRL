import sys
sys.path.append('.')
import torch
import numpy as np
from envs.jvrc import JvrcStepEnv

print("📊 JVRC步进机器人性能测试...")

# 加载模型
actor = torch.load('./experiments/jvrc_step/actor.pt', map_location='cpu')
actor.eval()

env = JvrcStepEnv()
results = []

for i in range(5):  # 测试5个episode
    obs = env.reset()
    total_reward = 0
    steps = 0
    successful_steps = 0
    
    print(f"\n🧪 测试Episode {i+1}")
    
    for step in range(1000):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = actor(obs_tensor).numpy()[0]
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # 计算成功的步进
        if reward > 0:
            successful_steps += 1
        
        if step % 200 == 0:
            print(f"  进度: {step}/1000, 当前总奖励: {total_reward:.2f}")
        
        if done:
            break
    
    success_rate = successful_steps / steps if steps > 0 else 0
    results.append((steps, total_reward, success_rate))
    
    print(f"  完成: {steps} 步, 总奖励: {total_reward:.2f}, 成功率: {success_rate:.1%}")

# 统计结果
print(f"\n📈 性能统计:")
avg_steps = np.mean([r[0] for r in results])
avg_reward = np.mean([r[1] for r in results])
avg_success = np.mean([r[2] for r in results])

print(f"平均episode长度: {avg_steps:.1f} 步")
print(f"平均总奖励: {avg_reward:.2f}")
print(f"平均成功率: {avg_success:.1%}")
print(f"最长episode: {max(r[0] for r in results)} 步")
print(f"最高总奖励: {max(r[1] for r in results):.2f}")
