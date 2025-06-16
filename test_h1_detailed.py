import sys
sys.path.append('.')

try:
    print("Importing H1Env...")
    from envs.h1 import H1Env
    print("✓ H1Env imported successfully")
    
    print("Creating H1Env instance (this might take a while)...")
    env = H1Env()
    print("✓ H1Env created successfully!")
    
    print("Testing environment methods...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    print("✓ H1 environment test completed successfully!")
    
except Exception as e:
    print(f"❌ Error at step: {e}")
    import traceback
    traceback.print_exc()
