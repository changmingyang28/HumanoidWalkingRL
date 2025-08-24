import torch
import time
from pathlib import Path

import mujoco
import mujoco.viewer

import imageio
from datetime import datetime

class EvaluateEnv:
    def __init__(self, env, policy, args):
        self.env = env
        self.policy = policy
        self.ep_len = args.ep_len
        self.override_params = None

        if args.out_dir is None:
            args.out_dir = Path(args.path.parent, "videos")

        video_outdir = Path(args.out_dir)
        try:
            Path.mkdir(video_outdir, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            video_fn = Path(video_outdir, args.path.stem + "-" + now + ".mp4")
            self.writer = imageio.get_writer(video_fn, fps=60)
        except Exception as e:
            print("Could not create video writer:", e)
            exit(-1)
    
    def _apply_overrides(self):
        """Apply parameter overrides after environment reset"""
        if self.override_params is None:
            return
            
        if self.override_params['speed'] is not None:
            self.env.task._goal_speed_ref = self.override_params['speed']
            
        if self.override_params['height'] is not None:
            self.env.task._goal_height_ref = self.override_params['height']
            
        if self.override_params['swing_duration'] is not None:
            self.env.task._swing_duration = self.override_params['swing_duration']
            
        if self.override_params['stance_duration'] is not None:
            self.env.task._stance_duration = self.override_params['stance_duration']
            
        if self.override_params['swing_duration'] is not None or self.override_params['stance_duration'] is not None:
            self.env.task._total_duration = self.env.task._swing_duration + self.env.task._stance_duration

    @torch.no_grad()
    def run(self):

        height = 480
        width = 640
        renderer = mujoco.Renderer(self.env.model, height, width)
        viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        frames = []

        # Make a camera.
        cam = viewer.cam
        mujoco.mjv_defaultCamera(cam)
        cam.elevation = -20
        cam.distance = 4

        reset_counter = 0
        observation = self.env.reset()
        self._apply_overrides()
        
        # Initialize LSTM hidden states if using LSTM policy
        if hasattr(self.policy, 'init_hidden_state'):
            self.policy.init_hidden_state()
        
        while self.env.data.time < self.ep_len:

            step_start = time.time()

            # forward pass and step
            raw = self.policy.forward(torch.tensor(observation, dtype=torch.float32), deterministic=True).detach().numpy()
            observation, reward, done, _ = self.env.step(raw.copy())

            # render scene
            cam.lookat = self.env.data.body(1).xpos.copy()
            renderer.update_scene(self.env.data, cam)
            pixels = renderer.render()
            frames.append(pixels)

            viewer.sync()

            if done and reset_counter < 3:
                observation = self.env.reset()
                self._apply_overrides()
                # Re-initialize LSTM hidden states after reset
                if hasattr(self.policy, 'init_hidden_state'):
                    self.policy.init_hidden_state()
                reset_counter += 1

            time_until_next_step = max(
                0, self.env.frame_skip*self.env.model.opt.timestep - (time.time() - step_start))
            time.sleep(time_until_next_step)

        for frame in frames:
            self.writer.append_data(frame)
        self.writer.close()
        self.env.close()
        viewer.close()
