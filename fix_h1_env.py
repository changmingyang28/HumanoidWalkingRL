import sys
sys.path.append('.')

# 检查H1Env缺少的方法
print("=== Checking H1Env methods ===")

from envs.h1.h1_env import H1Env

env = H1Env()

# 检查基础方法
methods_to_check = ['reset', 'step', 'render', 'close']
for method in methods_to_check:
    if hasattr(env, method):
        print(f"✓ {method}: exists")
    else:
        print(f"❌ {method}: missing")

# 检查父类方法
print(f"\nH1Env MRO: {H1Env.__mro__}")

# 查看基类是否有reset
from envs.common.mujoco_env import MujocoEnv
if hasattr(MujocoEnv, 'reset'):
    print("✓ MujocoEnv has reset method")
else:
    print("❌ MujocoEnv missing reset method")
