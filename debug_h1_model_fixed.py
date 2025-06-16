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
            try:
                body_mass = float(model.body(i).mass)
                print(f"  Body {i}: '{body_name}' (mass: {body_mass:.3f})")
            except:
                print(f"  Body {i}: '{body_name}' (mass: error)")
        
        # 检查pelvis是否存在
        try:
            pelvis_body = model.body("pelvis")
            print(f"\n✓ Found pelvis body with mass: {float(pelvis_body.mass):.3f}")
        except Exception as e:
            print(f"\n❌ Error accessing pelvis: {e}")
                
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found: {model_path}")
