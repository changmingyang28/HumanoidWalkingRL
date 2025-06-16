import sys
sys.path.append('.')
import os
import mujoco as mj

print("=== Debugging H1 initialization with correct config ===")

# 1. 检查原始模型
print("1. Checking original scene.xml...")
scene_path = "models/mujoco_menagerie/unitree_h1/scene.xml"
if os.path.exists(scene_path):
    model = mj.MjModel.from_xml_path(scene_path)
    print(f"   ✓ Scene model has {model.nbody} bodies")
    for i in range(min(10, model.nbody)):
        print(f"     Body {i}: {model.body(i).name}")
else:
    print("   ❌ Scene model not found!")

# 2. 运行完整的builder过程
print("\n2. Running builder with correct config...")

try:
    from envs.h1.gen_xml import builder, LEG_JOINTS, WAIST_JOINTS, ARM_JOINTS
    
    export_dir = "/tmp/mjcf-export/h1"
    os.makedirs(export_dir, exist_ok=True)
    
    # 使用H1Env中实际使用的配置
    config = {
        'unused_joints': [WAIST_JOINTS, ARM_JOINTS],
        'rangefinder': False,
        'raisedplatform': False,
        'ctrllimited': True,
        'jointlimited': True,
        'minimal': False,
    }
    
    print(f"   Config: {config}")
    print(f"   WAIST_JOINTS: {WAIST_JOINTS}")
    print(f"   ARM_JOINTS: {ARM_JOINTS}")
    
    # 运行builder
    builder(export_dir, config)
    
    # 3. 检查导出的文件
    print("\n3. Checking exported XML...")
    exported_path = os.path.join(export_dir, "h1.xml")
    if os.path.exists(exported_path):
        model = mj.MjModel.from_xml_path(exported_path)
        print(f"   ✓ Exported model has {model.nbody} bodies")
        for i in range(min(10, model.nbody)):
            print(f"     Body {i}: {model.body(i).name}")
        
        # 检查是否有pelvis
        try:
            pelvis = model.body("pelvis")
            print(f"   ✓ Pelvis found in exported model, mass: {pelvis.mass}")
        except Exception as e:
            print(f"   ❌ Pelvis NOT found in exported model: {e}")
            
        # 检查是否有torso_link  
        try:
            torso = model.body("torso_link")
            print(f"   ✓ torso_link found in exported model, mass: {torso.mass}")
        except Exception as e:
            print(f"   ❌ torso_link NOT found in exported model: {e}")
    else:
        print("   ❌ Exported XML not found!")
        
except Exception as e:
    print(f"❌ Error during builder test: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Testing what happens when we load this XML...")
try:
    exported_path = "/tmp/mjcf-export/h1/h1.xml"
    if os.path.exists(exported_path):
        model = mj.MjModel.from_xml_path(exported_path)
        print(f"   Model loaded, {model.nbody} bodies")
        
        # 尝试pelvis操作
        model.body("pelvis").mass = 8.89
        print("   ✓ Successfully set pelvis mass")
        
        model.body("torso_link").mass = 21.289
        print("   ✓ Successfully set torso_link mass")
        
    else:
        print("   ❌ No exported file to test")
        
except Exception as e:
    print(f"   ❌ Error testing exported model: {e}")
    import traceback
    traceback.print_exc()
