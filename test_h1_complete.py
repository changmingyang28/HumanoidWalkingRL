import sys
sys.path.append('.')
import numpy as np

print("=== Complete H1Env test ===")

try:
    from envs.h1.h1_env import H1Env
    
    print("1. Creating environment...")
    env = H1Env()
    print("✓ Environment created")
    
    print("2. Checking attributes...")
    print(f"✓ action_space shape: {env.action_space.shape}")
    print(f"✓ observation_space shape: {env.observation_space.shape}")
    
    print("3. Testing reset...")
    obs = env.reset()
    print(f"✓ Reset successful, obs shape: {obs.shape}")
    
    print("4. Testing step...")
    action = np.zeros_like(env.action_space)
    obs, reward, done, info = env.step(action)
    print(f"✓ Step successful")
    print(f"  - obs shape: {obs.shape}")
    print(f"  - reward: {reward:.4f}")
    print(f"  - done: {done}")
    print(f"  - info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
    
    print("5. Testing multiple steps...")
    for i in range(3):
        obs, reward, done, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.4f}, done={done}")
    
    print("\n✅ All tests passed! Environment is ready for training.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
