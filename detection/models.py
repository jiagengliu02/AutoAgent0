"""Frozen detection_ft models; inference only."""
import json
from pathlib import Path
CHECKPOINT = Path(__file__).resolve().parents[1] / 'checkpoints/bridgedrive'

def read(path):
    return json.loads(Path(path).read_text())

def build_model(args, state):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Specialist(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell=args.cell
            self.upstream=args.upstream
            self.lidar=getattr(args,'lidar',False)
            self.no_context=getattr(args,'no_context',False)
            if self.lidar:
                self.lidar_branch=build_branch(getattr(args,'coordinate_input',False))
            if self.upstream:
                c5=state['backbone.c5_conv.weight']
                self.c5_conv=nn.Conv2d(c5.shape[1],64,1)
                self.up_conv5=nn.Conv2d(64,64,3,padding=1)
                self.up_conv4=nn.Conv2d(64,64,3,padding=1)
                for name in ('c5_conv','up_conv5','up_conv4'):
                    getattr(self,name).load_state_dict({k:state['backbone.'+name+'.'+k] for k in ('weight','bias')})
            self.trunk=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),
                                     nn.Conv2d(64,64,3,padding=1),nn.ReLU())
            self.heat=nn.Conv2d(64,1,1)
            self.reg=nn.Conv2d(64,7,1)
            self.trunk[0].load_state_dict({k:state['center_net_decoder.heatmap_head.0.'+k] for k in ('weight','bias')})
            nn.init.normal_(self.heat.weight,std=.001)
            nn.init.constant_(self.heat.bias,-4.)
            nn.init.zeros_(self.reg.weight)
            nn.init.zeros_(self.reg.bias)
            with torch.no_grad():
                self.reg.bias[5]=1.

        def forward(self,x):
            if self.lidar:
                if self.no_context:
                    x=dict(x,bev=torch.zeros_like(x['bev']))
                x=self.lidar_branch(x)
            if self.upstream:
                x=F.relu(self.c5_conv(x))
                x=F.relu(self.up_conv5(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)))
                x=F.relu(self.up_conv4(F.interpolate(x,size=(80,96),mode='bilinear',align_corners=False)))
            x=F.interpolate(x,size=(round(80/self.cell),round(96/self.cell)),mode='bilinear',align_corners=False)
            x=self.trunk(x)
            return self.heat(x)[:,0], self.reg(x)
    return Specialist()

def decode(logit,reg,cell,floor=.05,dense=False,include_velocity=False,peak_kernel=3,box_nms=0.):
    import torch
    import torch.nn.functional as F
    probability=logit.sigmoid()
    peak=probability*(probability==F.max_pool2d(probability[:,None],peak_kernel,1,peak_kernel//2)[:,0])
    scores,indices=peak.flatten(1).topk(400 if peak_kernel==1 else 100,dim=1)
    values=reg.flatten(2).gather(2,indices[:,None].expand(-1,7,-1)).transpose(1,2)
    offsets=values[:,:,:2]+.5 if dense else values[:,:,:2].sigmoid()
    width=logit.shape[2]
    x=(indices%width+offsets[:,:,0])*cell-32
    y=(indices//width+offsets[:,:,1])*cell-40
    sizes=values[:,:,2:4].clamp(-3,3).exp()*.375
    yaw=.5*torch.atan2(values[:,:,4],values[:,:,5])
    fields=(x[:,:,None],y[:,:,None],sizes,yaw[:,:,None],scores[:,:,None])
    if include_velocity:fields=fields+(values[:,:,6:7],)
    boxes=torch.cat(fields,2).detach().float().cpu().numpy()
    results=[b[b[:,5]>=floor].tolist() for b in boxes]
    if box_nms:
        for i,items in enumerate(results):
            kept=[]
            for box in items:
                reject=False
                for previous in kept:
                    bound=(box[2]+box[3]+previous[2]+previous[3])/2
                    if abs(box[0]-previous[0])>bound or abs(box[1]-previous[1])>bound:continue
                    if bev_iou(box[:5],previous[:5])>box_nms:reject=True;break
                if not reject:kept.append(box)
            results[i]=kept
    return results


import numpy as np
def rasterize(points):
    points=np.asarray(points)
    p=points[(points[:,0]>=-32)&(points[:,0]<64)&(points[:,1]>=-40)&(points[:,1]<40)&(points[:,2]>=-.5)&(points[:,2]<3.5)]
    ix=((p[:,0]+32)*4).astype(int);iy=((p[:,1]+40)*4).astype(int)
    flat=iy*384+ix
    result=np.zeros((8,320*384),np.float32)
    # Six vertical density slices, plus mean subcell point offsets. Offsets
    # retain localization information finer than the 25cm raster spacing.
    height=np.clip(((p[:,2]+.5)/4*6).astype(int),0,5)
    np.add.at(result,(height,flat),1)
    count=result[:6].sum(0)
    np.add.at(result[6],flat,(p[:,0]+32)*4-ix-.5)
    np.add.at(result[7],flat,(p[:,1]+40)*4-iy-.5)
    result[6:]/=np.maximum(count,1)[None]
    result[:6]=np.log1p(result[:6])/np.log(16)
    return result.reshape(8,320,384)

def calibrate_dual_lidar(raw,specs):
    """Combine current raw LiDAR scans using the recorded mount convention."""
    clouds=[]
    for spec in specs:
        if spec['id'] not in ('lidar1','lidar2'):continue
        assert spec['pitch']==0 and spec['roll']==0
        yaw=np.radians(spec['yaw']);c,s=np.cos(yaw),np.sin(yaw)
        matrix=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        clouds.append(np.asarray(raw[spec['id']])[:,:3]@matrix.T+np.array([spec[k] for k in ('x','y','z')]))
    if len(clouds)!=2:raise ValueError('Both current calibrated LiDAR scans are required')
    return np.concatenate(clouds)

def build_branch(coordinates=False):
    import torch
    from torch import nn
    import torch.nn.functional as F
    def block(cin,cout,stride=1):
        return nn.Sequential(nn.Conv2d(cin,cout,3,stride=stride,padding=1),nn.GroupNorm(4,cout),nn.ReLU(),
                             nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(4,cout),nn.ReLU())
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.fine=block(12 if coordinates else 8,16)
            self.coordinates=coordinates
            if coordinates:
                y,x=torch.meshgrid(torch.arange(320)*.25-40+.125,torch.arange(384)*.25-32+.125,indexing='ij')
                distance=torch.sqrt(x*x+y*y).clamp_min(.1)
                self.register_buffer('coordinates_grid',torch.stack((x/64,y/40,x/distance,y/distance))[None],persistent=False)
            self.mid=block(16,32,2)
            self.coarse=block(32,64,2)
            self.context=nn.Conv2d(64,16,1)
            self.fuse=block(80,64)
            self.upmid=block(96,32)
            self.upfine=block(48,64)
        def forward(self,inputs):
            lidar=inputs['lidar']
            if self.coordinates:
                lidar=torch.cat((lidar,self.coordinates_grid.to(lidar.dtype).expand(lidar.shape[0],-1,-1,-1)),1)
            fine=self.fine(lidar)
            mid=self.mid(fine)
            coarse=self.coarse(mid)
            context=self.context(inputs['bev'])
            x=self.fuse(torch.cat((coarse,context),1))
            x=self.upmid(torch.cat((F.interpolate(x,size=mid.shape[-2:],mode='bilinear',align_corners=False),mid),1))
            return self.upfine(torch.cat((F.interpolate(x,size=fine.shape[-2:],mode='bilinear',align_corners=False),fine),1))
    return Branch()


def patch(points,center):
    c=np.asarray(center)
    delta=points[:,:2]-c
    keep=(np.abs(delta[:,0])<1)&(np.abs(delta[:,1])<1)&(points[:,2]>=-.5)&(points[:,2]<3.5)
    p=points[keep];d=delta[keep]
    ix=((d[:,0]+1)*32).astype(int);iy=((d[:,1]+1)*32).astype(int)
    flat=iy*64+ix;height=np.clip(((p[:,2]+.5)/4*6).astype(int),0,5)
    result=np.zeros((8,64*64),np.float32)
    np.add.at(result,(height,flat),1)
    count=result[:6].sum(0)
    np.add.at(result[6],flat,(d[:,0]+1)*32-ix-.5)
    np.add.at(result[7],flat,(d[:,1]+1)*32-iy-.5)
    result[6:]/=np.maximum(count,1)[None]
    result[:6]=np.log1p(result[:6])/np.log(16)
    return result.reshape(8,64,64)

def model_class():
    import torch
    from torch import nn
    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            layers=[];cin=12
            for cout in [16,32,64,96]:
                layers.extend([nn.Conv2d(cin,cout,3,stride=2,padding=1),nn.GroupNorm(4,cout),nn.ReLU()]);cin=cout
            self.encoder=nn.Sequential(*layers)
            self.output=nn.Sequential(nn.Flatten(),nn.Linear(96*4*4,128),nn.SiLU(),nn.Linear(128,2))
            nn.init.normal_(self.output[-1].weight,std=.001);nn.init.zeros_(self.output[-1].bias)
            y,x=torch.meshgrid(torch.arange(64)/32-1+1/64,torch.arange(64)/32-1+1/64,indexing='ij')
            self.register_buffer('xy',torch.stack((x,y))[None],persistent=False)
        def forward(self,x,direction):
            xy=self.xy.to(x.dtype).expand(x.shape[0],-1,-1,-1)
            direction=direction[:,:,None,None].expand(-1,-1,64,64).to(x.dtype)
            return .5*self.output(self.encoder(torch.cat((x,xy,direction),1))).tanh()
    return Refiner


def describe(cloud,tree,center):
    ids=tree.query_ball_point(center,1.5)
    near=cloud[ids]
    ground=float(np.percentile(near[:,2],5)) if len(near) else 0.
    result=[float(np.log1p(np.linalg.norm(center)))]
    for radius in [.25,.5,1.]:
        p=near[np.linalg.norm(near[:,:2]-center,axis=1)<radius] if len(near) else near
        z=p[:,2]-ground
        result.append(float(np.log1p(len(p))))
        result.extend([float(np.quantile(z,q)) if len(z) else 0. for q in [.5,.9,1.]])
        result.extend(float(np.log1p((z>height).sum())) for height in [.5,1.,1.5])
        high=p[z>.5,:2]
        if len(high)>2:
            covariance=np.cov(high.T);eigen=np.linalg.eigvalsh(covariance)
            result.extend(eigen.tolist())
        else:result.extend([0.,0.])
    return result


def corners(box):
    x, y, length, width, yaw = box
    local = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=np.float64)
    local *= [length / 2, width / 2]
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c,s],[-s,c]]) + [x,y]

def bev_iou(a, b):
    import cv2
    if min(a[2:4]) <= 0 or min(b[2:4]) <= 0:
        return 0.
    area, _ = cv2.intersectConvexConvex(corners(a).astype('float32'),
                                     corners(b).astype('float32'))
    return float(max(0., min(1., area / max(a[2]*a[3]+b[2]*b[3]-area, 1e-12))))


def merge_static(original,extra,threshold,nms=.3,policy='score'):
    old=[b for b in original if b[5]>=.35];new=[b for b in extra if b[5]>=threshold]
    candidates=(sorted(new,key=lambda b:-b[5])+sorted(old,key=lambda b:-b[5])) if policy=='prefer_new' else sorted(old+new,key=lambda b:-b[5])
    result=[]
    for b in candidates:
        if not any(bev_iou(b[:5],old[:5])>nms for old in result):result.append(b)
    return result
