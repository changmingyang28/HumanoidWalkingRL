import sys
sys.path.append('.')
import numpy as np

print("=== Testing H1Env reset ===")

try:
    from envs.h1.h1_env import H1Env
    
    env = H1Env()
    print("✓ H1Env created")
    
    # 测试reset
    obs = env.reset()
    print(f"✓ Reset successful, observation shape: {obs.shape}")
    print(f"  First 5 obs values: {obs[:5]}")
    
    # 测试step
    action = env.action_space.copy()  # 零动作
    obs, reward, done, info = env.step(action)
    print(f"✓ Step successful, reward: {reward}, done: {done}")
    
    print("✓ All environment methods working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
