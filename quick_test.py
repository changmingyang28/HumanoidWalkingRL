import sys
sys.path.append('.')
import numpy as np

try:
    from envs.h1.h1_env import H1Env
    
    env = H1Env()
    obs = env.reset()
    print(f"✓ Reset: obs shape {obs.shape}")
    
    for i in range(5):
        action = np.zeros(10)  # 零动作
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, done={done}")
        if done:
            print("Episode ended, resetting...")
            obs = env.reset()
    
    print("✅ Quick test passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
