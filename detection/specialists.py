"""Frozen detection_ft specialists; inference only."""
import argparse
import numpy as np
import torch
from models import CHECKPOINT, build_model, decode
from models import rasterize
class PedestrianSpecialist:
    def __init__(self,checkpoint,device='cuda:0'):
        self.device=torch.device(device)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.config=argparse.Namespace(**saved['config'])
        original=torch.load(CHECKPOINT/'model_BridgeDrive_m2_k60_0030.pth',map_location='cpu',weights_only=True)
        self.model=build_model(self.config,original).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(saved['state_dict'],strict=True)
        self.threshold=saved['threshold']

    @torch.inference_mode()
    def predict_native(self,bev_feature,current_dual_lidar_ego=None,score_threshold=None):
        """Return native ego boxes [x,y,half_length,half_width,yaw,speed,brake,class,score].

        bev_feature: [1,64,80,96] post-top-down original features, or the
        original pre-top-down feature for an upstream trial.
        current_dual_lidar_ego: current scan only, both calibrated LiDARs
        concatenated in the current ego frame; not a temporal accumulation.
        """
        feature=bev_feature.to(self.device)
        if getattr(self.config,'lidar',False):
            if current_dual_lidar_ego is None:
                raise ValueError('This checkpoint needs calibrated current dual-LiDAR points')
            # Offline caches are float16; preserve the same quantization online.
            feature=dict(bev=feature,lidar=torch.from_numpy(rasterize(current_dual_lidar_ego).astype(np.float16))[None].to(self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32))
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            logits,reg=self.model(feature)
        boxes=decode(logits.float(),reg.float(),self.config.cell,floor=self.threshold if score_threshold is None else score_threshold,
                     dense=getattr(self.config,'dense_regression',False),include_velocity=True,
                     peak_kernel=getattr(self.config,'peak_kernel',3),box_nms=getattr(self.config,'box_nms',0.))[0]
        return np.asarray([[b[0],b[1],b[2]/2,b[3]/2,b[4],b[6],0.,1.,b[5]] for b in boxes],dtype=np.float32).reshape(-1,9)

    def merge_native(self,original_boxes,bev_feature,current_dual_lidar_ego=None):
        original=np.asarray(original_boxes).reshape(-1,9)
        # Replace WALKER only; BIKER and every non-human class remain intact.
        retained=original[original[:,7]!=1].copy()
        human=self.predict_native(bev_feature,current_dual_lidar_ego)
        return np.concatenate((retained,human),axis=0)


import hashlib
import json
import math
from pathlib import Path
from scipy.spatial import cKDTree
from models import model_class, patch
from models import describe
class PedestrianPipeline:
    def __init__(self,bundle,device='cuda:0'):
        self.bundle=Path(bundle);self.selection=json.loads((self.bundle/'selection.json').read_text())
        for name,expected in self.selection['bundle_hashes'].items():
            if hashlib.sha256((self.bundle/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Bundle checksum mismatch: {name}')
        if self.selection.get('tracking', False):
            raise ValueError('Release supports the frozen single-frame detection pipeline only')
        self.detector=PedestrianSpecialist(self.bundle/'detector.pt',device=device)
        self.device=torch.device(device);self.refiner=None;self.gate=None
        if (self.bundle/'refiner.pt').exists():
            state=torch.load(self.bundle/'refiner.pt',map_location='cpu',weights_only=True)
            assert state['architecture']=='silu_v2'
            self.refiner=model_class()().to(self.device).eval().requires_grad_(False)
            self.refiner.load_state_dict(state['state_dict'])
        if (self.bundle/'shape_gate.joblib').exists():
            import joblib
            self.gate=joblib.load(self.bundle/'shape_gate.joblib')
        self.reset()

    def reset(self):
        self.position=np.zeros(2);self.last_motion=None;self.pose=None
        self.tracking_state=None;self.frame=0;self.last_prediction_time=None

    def observe_motion(self,speed,compass,timestamp):
        yaw=float(compass)-math.pi/2;t=float(timestamp)
        if not np.isfinite([speed,yaw,t]).all():raise ValueError('Nonfinite motion sensor input')
        velocity=float(speed)*np.array([math.cos(yaw),math.sin(yaw)])
        if self.last_motion is not None:
            dt=t-self.last_motion[0]
            if dt<=0:raise ValueError('Motion timestamps must increase; reset between episodes')
            self.position+=dt*(velocity+self.last_motion[1])*.5
        self.last_motion=(t,velocity);self.pose=dict(x=float(self.position[0]),y=float(self.position[1]),yaw=yaw,t=t)

    @torch.inference_mode()
    def predict(self,original_native_boxes,bev_feature,current_dual_lidar_ego):
        if self.pose is None:raise ValueError('Call observe_motion on every sensor frame first')
        if self.last_prediction_time==self.pose['t']:raise ValueError('Duplicate prediction timestamp')
        self.last_prediction_time=self.pose['t'];self.frame+=1
        cloud=np.asarray(current_dual_lidar_ego)[:,:3]
        native=self.detector.predict_native(bev_feature,cloud,score_threshold=.1)
        cloud=cloud.astype(np.float32)
        boxes=[[float(b[0]),float(b[1]),float(2*b[2]),float(2*b[3]),float(b[4]),float(b[8]),float(b[5])] for b in native]
        if self.refiner is not None and boxes:
            centers=np.asarray([b[:2] for b in boxes]);direction=centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1)
            crops=np.stack([patch(cloud,b[:2]) for b in boxes]);offsets=[]
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                for start in range(0,len(boxes),128):
                    offsets.append(self.refiner(torch.from_numpy(crops[start:start+128]).to(self.device),torch.from_numpy(direction[start:start+128].astype(np.float32)).to(self.device)).float().cpu().numpy())
            for b,delta in zip(boxes,np.concatenate(offsets)):
                b[0]+=float(delta[0])*self.selection.get('refiner_strength',1.)
                b[1]+=float(delta[1])*self.selection.get('refiner_strength',1.)
        if self.gate is not None and boxes:
            tree=cKDTree(cloud[:,:2]);features=np.asarray([describe(cloud,tree,b[:2]) for b in boxes],np.float32)
            probabilities=self.gate.predict_proba(features)[:,1]
            for b,p in zip(boxes,probabilities):b[5]*=float(p)
        human=np.asarray([[b[0],b[1],b[2]/2,b[3]/2,b[4],b[6] if len(b)>6 else 0.,0.,1.,b[5]]
                          for b in boxes if b[5]>=self.selection['threshold']],dtype=np.float32).reshape(-1,9)
        original=np.asarray(original_native_boxes).reshape(-1,9)
        return np.concatenate((original[original[:,7]!=1].copy(),human),0)


from models import read
from models import merge_static
class StaticSpecialist:
    def __init__(self,bundle,device='cuda:0'):
        self.bundle=Path(bundle);self.selection=read(self.bundle/'selection.json');self.device=torch.device(device)
        for name,expected in self.selection['bundle_hashes'].items():
            if hashlib.sha256((self.bundle/name).read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint hash mismatch: '+name)
        saved=torch.load(self.bundle/'detector.pt',map_location='cpu',weights_only=True);self.config=argparse.Namespace(**saved['config'])
        original=torch.load(CHECKPOINT/'model_BridgeDrive_m2_k60_0030.pth',map_location='cpu',weights_only=True)
        self.model=build_model(self.config,original).to(self.device).eval().requires_grad_(False);del original
        self.model.load_state_dict(saved['state_dict'],strict=True);self.refiner=None
        if (self.bundle/'refiner.pt').exists():
            state=torch.load(self.bundle/'refiner.pt',map_location='cpu',weights_only=True)
            assert state['semantic']=='cone' and state['architecture']=='silu_v2'
            self.refiner=model_class()().to(self.device).eval().requires_grad_(False);self.refiner.load_state_dict(state['state_dict'])

    @torch.inference_mode()
    def candidates(self,bev_feature,current_dual_lidar_ego):
        cloud=np.asarray(current_dual_lidar_ego)[:,:3]
        feature=dict(bev=bev_feature.to(self.device),lidar=torch.from_numpy(rasterize(cloud).astype(np.float16))[None].to(self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32))
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            logits,reg=self.model(feature)
        boxes=decode(logits.float(),reg.float(),self.config.cell,dense=True,
                     peak_kernel=self.config.peak_kernel,box_nms=self.config.box_nms)[0]
        if boxes and self.refiner is not None:
            cloud=cloud.astype(np.float32);centers=np.asarray([b[:2] for b in boxes])
            directions=centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1.)
            crops=np.stack([patch(cloud,b[:2]) for b in boxes]);offsets=[]
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                for start in range(0,len(boxes),128):
                    offsets.append(self.refiner(torch.from_numpy(crops[start:start+128]).to(self.device),torch.from_numpy(directions[start:start+128].astype(np.float32)).to(self.device)).float().cpu().numpy())
            for b,d in zip(boxes,np.concatenate(offsets)):
                b[0]+=float(d[0])*self.selection['refiner_strength'];b[1]+=float(d[1])*self.selection['refiner_strength']
        return boxes

    def predict(self,original_native_boxes,bev_feature,current_dual_lidar_ego):
        original=np.asarray(original_native_boxes).reshape(-1,9)
        old=[[float(b[0]),float(b[1]),float(2*b[2]),float(2*b[3]),float(b[4]),float(b[8])]
             for b in original if b[7]==5 and -32<=b[0]<64 and -40<=b[1]<40]
        boxes=merge_static(old,self.candidates(bev_feature,current_dual_lidar_ego),self.selection['threshold'],self.selection.get('nms',.3),self.selection.get('merge_policy','score'))
        added=np.asarray([[b[0],b[1],b[2]/2,b[3]/2,b[4],0.,0.,5.,b[5]] for b in boxes],np.float32).reshape(-1,9)
        return np.concatenate((original[original[:,7]!=5].copy(),added),axis=0)
