"""GradMotion Platform Logger Integration"""
import os
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import shutil

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Only tensorboard logging will be used.")


class GMLogger:
    """Enhanced logger for GradMotion platform with model uploading capabilities."""
    
    def __init__(self, log_dir, algorithm_name, run_name=None, use_wandb=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.algorithm_name = algorithm_name
        self.run_name = run_name or f"{algorithm_name}_{self.log_dir.name}"
        
        # Tensorboard writer (always available)
        self.tb_writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        # WandB integration for GM platform
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            self.init_wandb()
        
        # Model saving paths
        self.models_dir = self.log_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"GMLogger initialized for {algorithm_name}")
        print(f"Log directory: {self.log_dir}")
        print(f"WandB enabled: {self.use_wandb}")
    
    def init_wandb(self):
        """Initialize WandB for GM platform integration."""
        try:
            # Get WandB project from environment or use default
            project = os.getenv('WANDB_PROJECT', 'humanoid-walking-rl')
            entity = os.getenv('WANDB_ENTITY', None)  # Optional: set your GM username
            
            wandb.init(
                project=project,
                entity=entity,
                name=self.run_name,
                dir=str(self.log_dir),
                tags=[self.algorithm_name, 'humanoid', 'walking']
            )
            print(f"WandB initialized: project={project}, run={self.run_name}")
            
        except Exception as e:
            print(f"WandB initialization failed: {e}")
            self.use_wandb = False
    
    def log_scalar(self, tag, value, step):
        """Log scalar value to both tensorboard and wandb."""
        self.tb_writer.add_scalar(tag, value, step)
        
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
    
    def log_scalars(self, metrics, step):
        """Log multiple metrics at once."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)
    
    def save_model(self, model_state, step, model_name="model"):
        """Save model and optionally upload to GM platform."""
        # Save locally
        filename = f"{model_name}_{step}.pt"
        model_path = self.models_dir / filename
        
        torch.save(model_state, model_path)
        print(f"Model saved: {model_path}")
        
        # Upload to WandB (GM platform)
        if self.use_wandb:
            try:
                wandb.save(str(model_path), base_path=str(self.log_dir))
                print(f"Model uploaded to GM platform: {filename}")
            except Exception as e:
                print(f"Failed to upload model to GM platform: {e}")
        
        return model_path
    
    def save_checkpoint(self, checkpoint_data, step, checkpoint_name="checkpoint"):
        """Save full training checkpoint."""
        filename = f"{checkpoint_name}_{step}.pt"
        checkpoint_path = self.models_dir / filename
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        if self.use_wandb:
            try:
                wandb.save(str(checkpoint_path), base_path=str(self.log_dir))
                print(f"Checkpoint uploaded to GM platform: {filename}")
            except Exception as e:
                print(f"Failed to upload checkpoint to GM platform: {e}")
        
        return checkpoint_path
    
    def log_hyperparameters(self, hparams):
        """Log hyperparameters."""
        if self.use_wandb:
            wandb.config.update(hparams)
        
        # Also save to a text file
        hparams_file = self.log_dir / "hyperparameters.txt"
        with open(hparams_file, 'w') as f:
            for key, value in hparams.items():
                f.write(f"{key}: {value}\n")
    
    def close(self):
        """Close all logging resources."""
        self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()
        print("GMLogger closed")