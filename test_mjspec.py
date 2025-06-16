import sys
sys.path.append('.')
import mujoco
import os

print("=== Testing MjSpec vs MjModel ===")

xml_path = "/tmp/mjcf-export/h1/h1.xml"

if os.path.exists(xml_path):
    print(f"Testing file: {xml_path}")
    
    # 方法1: 直接加载XML (我们测试用的)
    print("\n1. Using MjModel.from_xml_path:")
    try:
        model1 = mujoco.MjModel.from_xml_path(xml_path)
        print(f"   ✓ Loaded {model1.nbody} bodies")
        for i in range(min(5, model1.nbody)):
            print(f"     Body {i}: {model1.body(i).name}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 方法2: 使用MjSpec (MujocoEnv用的)
    print("\n2. Using MjSpec:")
    try:
        spec = mujoco.MjSpec()
        spec.from_file(xml_path)
        model2 = spec.compile()
        print(f"   ✓ Loaded {model2.nbody} bodies")
        for i in range(min(5, model2.nbody)):
            print(f"     Body {i}: {model2.body(i).name}")
        
        # 测试pelvis访问
        try:
            pelvis = model2.body("pelvis")
            print(f"   ✓ Pelvis accessible via MjSpec")
        except Exception as e:
            print(f"   ❌ Pelvis NOT accessible via MjSpec: {e}")
            
    except Exception as e:
        print(f"   ❌ MjSpec error: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print(f"XML file not found: {xml_path}")
