import mujoco as mj
import os

# 检查导出的模型
model_path = "/tmp/mjcf-export/h1/h1.xml"
if os.path.exists(model_path):
    try:
        model = mj.MjModel.from_xml_path(model_path)
        print(f"Model loaded successfully!")
        print(f"Number of bodies: {model.nbody}")
        print("Body names:")
        for i in range(model.nbody):
            body_name = model.body(i).name
            body_mass = model.body(i).mass
            print(f"  Body {i}: '{body_name}' (mass: {body_mass:.3f})")
        
        print("\nLooking for pelvis-like names:")
        for i in range(model.nbody):
            body_name = model.body(i).name.lower()
            if 'pelvis' in body_name or 'torso' in body_name or 'base' in body_name:
                print(f"  Found candidate: '{model.body(i).name}'")
                
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found: {model_path}")

# 也检查原始模型
orig_model_path = "models/mujoco_menagerie/unitree_h1/h1.xml"
if os.path.exists(orig_model_path):
    print(f"\n--- Original H1 model ---")
    try:
        model = mj.MjModel.from_xml_path(orig_model_path)
        print(f"Original model bodies:")
        for i in range(min(10, model.nbody)):  # 只显示前10个
            body_name = model.body(i).name
            print(f"  Body {i}: '{body_name}'")
    except Exception as e:
        print(f"Error loading original model: {e}")
