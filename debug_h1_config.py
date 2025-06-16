import sys
sys.path.append('.')
import os

print("=== Debugging H1 configuration ===")

try:
    from envs.common import config_builder
    from envs.h1.gen_xml import LEG_JOINTS
    
    print(f"LEG_JOINTS: {LEG_JOINTS}")
    
    # 加载配置文件
    config_path = os.path.join('envs', 'h1', 'configs', 'base.yaml')
    print(f"Config path: {config_path}")
    
    if os.path.exists(config_path):
        print("✓ Config file exists")
        cfg = config_builder.load_yaml(config_path)
        print(f"Config loaded: {type(cfg)}")
        
        # 检查配置内容
        print(f"Config attributes: {dir(cfg)}")
        
        # 检查pdgains
        if hasattr(cfg, 'pdgains'):
            print(f"✓ pdgains found: {cfg.pdgains}")
            gains_dict = cfg.pdgains.to_dict()
            print(f"Gains dict: {gains_dict}")
            
            # 检查每个关节
            for joint in LEG_JOINTS:
                if joint in gains_dict:
                    print(f"  ✓ {joint}: {gains_dict[joint]}")
                else:
                    print(f"  ❌ {joint}: NOT FOUND")
        else:
            print("❌ pdgains not found in config")
            
    else:
        print("❌ Config file not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
