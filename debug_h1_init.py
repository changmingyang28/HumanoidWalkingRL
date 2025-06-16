import sys
sys.path.append('.')
import os
import mujoco as mj

print("=== Debugging H1 initialization ===")

# 1. 检查原始模型
print("1. Checking original H1 model...")
orig_path = "models/mujoco_menagerie/unitree_h1/h1.xml"
if os.path.exists(orig_path):
    model = mj.MjModel.from_xml_path(orig_path)
    print(f"   ✓ Original model has {model.nbody} bodies")
    for i in range(min(5, model.nbody)):
        print(f"     Body {i}: {model.body(i).name}")
else:
    print("   ❌ Original model not found!")

# 2. 模拟H1环境的初始化过程
print("\n2. Simulating H1Env initialization...")

try:
    from envs.h1.gen_xml import builder
    from envs.h1.h1_env import H1_DESCRIPTION_PATH
    
    print(f"   H1_DESCRIPTION_PATH: {H1_DESCRIPTION_PATH}")
    
    # 检查源文件是否存在
    if os.path.exists(H1_DESCRIPTION_PATH):
        print("   ✓ Source XML exists")
        model = mj.MjModel.from_xml_path(H1_DESCRIPTION_PATH)
        print(f"   ✓ Source model has {model.nbody} bodies")
    else:
        print("   ❌ Source XML not found!")
    
    # 3. 运行builder
    print("\n3. Running XML builder...")
    export_dir = "/tmp/mjcf-export/h1"
    config = {}
    
    # 手动运行builder来看看发生了什么
    builder(export_dir, config)
    
    # 4. 检查导出的文件
    print("\n4. Checking exported XML...")
    exported_path = os.path.join(export_dir, "h1.xml")
    if os.path.exists(exported_path):
        model = mj.MjModel.from_xml_path(exported_path)
        print(f"   ✓ Exported model has {model.nbody} bodies")
        for i in range(min(5, model.nbody)):
            print(f"     Body {i}: {model.body(i).name}")
        
        # 检查是否有pelvis
        try:
            pelvis = model.body("pelvis")
            print(f"   ✓ Pelvis found in exported model")
        except:
            print(f"   ❌ Pelvis NOT found in exported model")
    else:
        print("   ❌ Exported XML not found!")

except Exception as e:
    print(f"❌ Error during builder test: {e}")
    import traceback
    traceback.print_exc()

# 5. 查看到底H1Env加载的是什么文件
print("\n5. Checking what H1Env actually loads...")
try:
    from envs.h1.h1_env import H1Env
    from envs.common.mujoco_env import MujocoEnv
    
    # 查看MujocoEnv.__init__的参数
    print("   Looking at the initialization path...")
    
    # 我们需要手动追踪路径
    import tempfile
    export_dir = tempfile.mkdtemp(prefix="mjcf-export", dir="/tmp")
    h1_dir = os.path.join(export_dir, "h1")
    os.makedirs(h1_dir, exist_ok=True)
    
    print(f"   Export directory: {h1_dir}")
    
except Exception as e:
    print(f"❌ Error during path check: {e}")
    import traceback
    traceback.print_exc()
