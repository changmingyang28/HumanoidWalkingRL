#!/usr/bin/env python3
"""
调试stepping任务
"""
import torch
import pickle
import numpy as np
from pathlib import Path
from envs.h1 import H1Env

def debug_stepping():
    # 加载模型
    model_dir = "experiments/h1_step_20250821_001704"
    actor_path = Path(model_dir, "actor_19999.pt")
    pkl_path = Path(model_dir, "experiment.pkl")
    
    print("📁 加载配置...")
    run_args = pickle.load(open(pkl_path, "rb"))
    print(f"环境: {run_args.env}")
    print(f"配置文件: {getattr(run_args, 'yaml', 'None')}")
    
    print("\n📁 加载模型...")
    policy = torch.load(actor_path, weights_only=False, map_location='cpu')
    policy.eval()
    print(f"模型类型: {type(policy)}")
    print(f"模型参数数量: {sum(p.numel() for p in policy.parameters())}")
    
    print("\n🌍 创建环境...")
    env = H1Env()
    print(f"环境类型: {type(env)}")
    print(f"动作空间: {env.action_space.shape if hasattr(env.action_space, 'shape') else len(env.action_space)}")
    print(f"观测空间: {env.observation_space.shape if hasattr(env.observation_space, 'shape') else len(env.observation_space)}")
    
    print("\n🔄 重置环境...")
    obs = env.reset()
    print(f"初始观测形状: {obs.shape}")
    print(f"初始观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
    
    print("\n🎯 检查任务配置...")
    if hasattr(env, 'task'):
        task = env.task
        print(f"任务类型: {type(task)}")
        
        if hasattr(task, 'sequence'):
            print(f"步进序列长度: {len(task.sequence) if task.sequence is not None else 'None'}")
            if task.sequence is not None and len(task.sequence) > 0:
                print(f"第一个目标: {task.sequence[0]}")
        
        if hasattr(task, 't1') and hasattr(task, 't2'):
            print(f"当前目标索引: t1={task.t1}, t2={task.t2}")
    
    print("\n🤖 测试策略...")
    for step in range(5):
        print(f"\n--- 步骤 {step+1} ---")
        
        # 获取动作
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = policy.forward(obs_tensor, deterministic=True).detach().numpy()
        
        print(f"动作形状: {action.shape}")
        print(f"动作范围: [{action.min():.3f}, {action.max():.3f}]")
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        print(f"奖励: {reward:.4f}")
        print(f"完成: {done}")
        print(f"奖励详情: {info if isinstance(info, dict) else 'N/A'}")
        
        # 检查机器人状态
        if hasattr(env, 'interface'):
            root_pos = env.interface.get_qpos()[:3]
            print(f"机器人位置: [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}]")
        
        if done:
            print("🔄 环境重置")
            obs = env.reset()
    
    print("\n✅ 调试完成")

if __name__ == "__main__":
    debug_stepping()