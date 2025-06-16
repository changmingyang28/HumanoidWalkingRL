import sys
sys.path.append('.')
import mujoco as mj

try:
    # 尝试直接加载导出的模型并设置pelvis质量
    model_path = "/tmp/mjcf-export/h1/h1.xml"
    model = mj.MjModel.from_xml_path(model_path)
    
    print("Trying to access pelvis body...")
    pelvis_body = model.body("pelvis")
    print(f"✓ Pelvis body found: {pelvis_body.name}")
    
    print("Trying to read pelvis mass...")
    current_mass = pelvis_body.mass
    print(f"✓ Current pelvis mass: {current_mass}")
    
    print("Trying to set pelvis mass...")
    model.body("pelvis").mass = 8.89
    print(f"✓ Successfully set pelvis mass to 8.89")
    
    print("✓ All pelvis operations successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
