"""Inference components extracted from the frozen detection_ft runtime."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from specialists import PedestrianPipeline
from specialists import StaticSpecialist
from models import calibrate_dual_lidar
from agenticdriving.planners.bridgedrive.carla_inputs import BRIDGEDRIVE_SENSOR_SPECS
class ClosedLoopDetection:
    def __init__(self, agent):
        root = Path(__file__).resolve().parents[1]
        self.device = agent.device
        self.config = agent.training_config
        self.frames = 0
        self.raw = None
        checkpoint = root / 'checkpoints/bridgedrive/model_BridgeDrive_m2_k60_0030.pth'
        expected = json.loads((root/'checkpoints/detection_ft/pedestrian/selection.json').read_text())['original_checkpoint_sha256']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            # Construction must not advance the original inference RNG streams.
            with torch.random.fork_rng(devices=[self.device] if self.device.type == 'cuda' else []):
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                self.backbone = copy.deepcopy(agent.closed_loop_inference.nets[0].backbone)
                self.backbone.load_state_dict({k[len('backbone.'):]: v for k,v in state.items() if k.startswith('backbone.')}, strict=True)
                self.backbone.eval().requires_grad_(False)
                del state
                self.human = PedestrianPipeline(root/'checkpoints/detection_ft/pedestrian', device=self.device)
                self.static = StaticSpecialist(root/'checkpoints/detection_ft/static', device=self.device)
                assert not self.human.selection.get('tracking', False)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        out = Path(os.environ.get('BRIDGE_AUDIT_OUT') or os.environ['AGENTICDRIVING_RUN_DIR'])
        out.mkdir(parents=True, exist_ok=True)
        self.stream = (out/'detection_changes.jsonl').open('w', buffering=1)
        (out/'detection_loaded.json').write_text(json.dumps({
            'backbone': str(checkpoint), 'backbone_sha256': expected,
            'human': str(root/'checkpoints/detection_ft/pedestrian'), 'static': str(root/'checkpoints/detection_ft/static'),
            'host_checkpoint': agent.config_closed_loop.__class__.__name__,
            'planner_changed': False, 'semantic_changed': False,
            'thresholds': {'1': .1, 'others': .35},
        }, indent=2))

    def observe(self, inputs):
        for key in ('lidar1', 'lidar2', 'rgb_1', 'rgb_2', 'rgb_3'):
            if key not in inputs:
                raise ValueError('Fine-tuned detection needs the original full rig: '+key)
        # SensorAgent.tick mutates its input mapping during preprocessing.
        self.raw = {k: (inputs[k][0], np.asarray(inputs[k][1]).copy())
                    for k in ('lidar1', 'lidar2')}
        self.frames += 1
        self.human.observe_motion(float(inputs['speed'][1]['speed']),
                                  float(np.asarray(inputs['imu'][1])[-1]), self.frames/20.)

    @torch.inference_mode()
    def apply(self, harvest, tensors):
        before = np.asarray(harvest['boxes_vehicle_system'], dtype=np.float32).reshape(-1,9)
        # Training used full mount z translation; the old LiDAR geometry
        # validator has a different convention and must not supply this cloud.
        cloud = calibrate_dual_lidar({k: v[1] for k,v in self.raw.items()}, BRIDGEDRIVE_SENSOR_SPECS)
        with torch.autocast(self.device.type, dtype=self.config.torch_float_type,
                            enabled=self.config.use_mixed_precision_training):
            bev, _ = self.backbone(tensors)
            feature = self.backbone.top_down(bev).detach().float()
        human = self.human.predict(before, feature, cloud)
        after = self.static.predict(human, feature, cloud)
        untouched = lambda boxes: boxes[(boxes[:,7] != 1) & (boxes[:,7] != 5)]
        np.testing.assert_array_equal(untouched(before), untouched(after))
        np.testing.assert_array_equal(human[human[:,7] == 1], after[after[:,7] == 1])
        self.stream.write(json.dumps({'tick': self.frames, 'sensor_frame': int(self.raw['lidar1'][0]),
            'before': before.tolist(), 'after': after.tolist(),
            'non_target_boxes_unchanged': True})+'\n')
        return dict(harvest, boxes_vehicle_system=after, score_threshold_by_class={1: .1})
