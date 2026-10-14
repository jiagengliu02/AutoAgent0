import numpy as np
import torch
import torch.nn.functional as F
import cv2
import time
from PIL import Image
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class DenseDepthProbLabelGenerator(object):
    def __init__(
        self,
        max_depth=10,
        min_depth=0.25,
        num_depth=64,
        origin_stride=4,
        strides=[4, 8, 16, 32],
        depth_mode=0,
        image_hw=None,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.num_depth = num_depth
        self.origin_stride = origin_stride
        self.strides = [stride // origin_stride for stride in strides]
        self.depth_mode = depth_mode
        self.image_hw = np.array(image_hw)

    def __call__(self, input_dict):
        aug_config = input_dict.get("aug_config")
        filename = input_dict["depth_filename"]
        depths = [cv2.imread(name, cv2.IMREAD_UNCHANGED) for name in filename]
        N = len(depths)
        new_depths = []
        for i in range(N):
            depth = self._img_transform(
                depths[i] , aug_config,
            )
            new_depths.append(depth)

        depth = np.stack(new_depths)
        if self.origin_stride != 1:
            H, W = depth.shape[-2:]
            depth = np.transpose(depth, (1, 2, 0))
            depth = mmcv.imresize(
                depth, (W//self.origin_stride, H//self.origin_stride),
                interpolation="nearest",
            )
            depth = np.transpose(depth, (2, 0, 1))
        input_dict["dense_depth"] = depth

        if self.depth_mode == 0:
            return self.get_depth_gt_prob(input_dict)
        elif self.depth_mode == 1:
            return self.get_depth_gt_onehot(input_dict)
        else:
            assert False

        # i=0
        # imgs = input_dict["img"]
        # for image_id in range(6):
        #     i+=1
        #     image = imgs[image_id]
        #     gt_depth_image = depth_map[image_id]
        #     gt_depth_image = np.expand_dims(gt_depth_image,2).repeat(3,2)

        #     #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
        #     im_color=cv2.applyColorMap(cv2.convertScaleAbs(gt_depth_image,alpha=15),cv2.COLORMAP_JET)
        #     #convert to mat png
        #     # image[gt_depth_image>0] = im_color[gt_depth_image>0]
        #     im=Image.fromarray(np.uint8(image))
        #     #save image
        #     im.save('vis/im_visualize_{}.png'.format(i))



        # import matplotlib.pyplot as plt
        # for i, depth in enumerate(new_depths):
        #     plt.imshow(depth)
        #     plt.colorbar()
        #     plt.savefig(f"vis/depth_hm_{i}.jpg")
        #     plt.close()
        # imgs = input_dict["img"]
        # image = np.concatenate(
        #     [
        #         np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
        #         np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
        #     ],
        #     axis=0,
        # )
        # cv2.imwrite(f"vis/img.jpg", image)

        # for i in range(3):
        #     imgs = new_depths[i]
        #     image = np.concatenate(
        #         [
        #             np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
        #             np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
        #         ],
        #         axis=0,
        #     )
        #     cv2.imwrite(f"vis/depth_{i}.jpg", image*255)

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        return img

    def get_depth_gt_prob(self, input_dict):
        depth = input_dict["dense_depth"]
        depth = depth[:, None]
        depth_valid_mask = depth > 0
        depth = np.clip(
            depth,
            a_min=self.min_depth,
            a_max=self.max_depth,
        )
        depth_anchor = np.linspace(
            self.min_depth, self.max_depth, self.num_depth)[:, None, None]
        distance = np.abs(depth - depth_anchor)
        mask = distance < (depth_anchor[1] - depth_anchor[0])
        depth_gt = np.where(mask, depth_anchor, 0)
        y = depth_gt.sum(axis=1, keepdims=True) - depth_gt
        depth_prob_gt = np.where(
            (depth_gt != 0) & depth_valid_mask,
            (depth - y) / (depth_gt - y),
            0,
        )
        views, _, H, W = depth.shape
        gt = []
        for s in self.strides:
            gt_tmp = np.reshape(
                depth_prob_gt, (views, self.num_depth, H//s, s, W//s, s))
            gt_tmp = gt_tmp.sum(axis=-1).sum(axis=3)
            mask_tmp = depth_valid_mask.reshape(views, 1, H//s, s, W//s, s)
            mask_tmp = mask_tmp.sum(axis=-1).sum(axis=3)
            gt_tmp /= np.clip(mask_tmp, a_min=1, a_max=None)
            gt_tmp = gt_tmp.reshape(views, self.num_depth, -1)
            gt_tmp = np.transpose(gt_tmp, (0, 2, 1))
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        input_dict["depth_prob_gt"] = gt
        return input_dict

    def get_depth_gt_onehot(self, input_dict):
        depth = input_dict["dense_depth"]
        depth = np.where(depth > 0, depth, 1e5)
        views, H, W = depth.shape
        depth_interval = (self.max_depth - self.min_depth) / (self.num_depth - 1)
        gt = []
        for s in self.strides:
            gt_tmp = np.reshape(
                depth, (views, H//s, s, W//s, s))
            gt_tmp = gt_tmp.transpose(0, 1, 3, 2, 4).reshape((views, H//s, W//s, -1))
            gt_tmp = gt_tmp.min(axis=-1)
            gt_tmp = (gt_tmp - self.min_depth) / depth_interval + 1
            gt_tmp = np.where(
                (gt_tmp >= 0) & (gt_tmp < self.num_depth + 1),
                gt_tmp,
                0,
            )
            gt_tmp = F.one_hot(
                torch.tensor(gt_tmp).long(),
                num_classes=self.num_depth + 1,
            )[..., 1:]
            gt_tmp = gt_tmp.reshape(views, -1, self.num_depth)
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        input_dict["depth_prob_gt"] = gt
        return input_dict

@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict

@PIPELINES.register_module()
class CustomPointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        import torch
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float64) - 1
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        # sort = (ranks + depth / 100.).argsort()
        sort = np.argsort(depth.numpy())
        sort = torch.tensor(sort.copy())
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        import torch
        points_lidar = torch.tensor(results['points']).to(torch.float64)
        imgs = np.stack(results['img'])
        # img_aug_matrix  = results['img_aug_matrix']
        # post_rots = [torch.tensor(single_aug_matrix[:3, :3]).to(torch.float) for single_aug_matrix in img_aug_matrix]
        # post_trans = torch.stack([torch.tensor(single_aug_matrix[:3, 3]).to(torch.float) for single_aug_matrix in img_aug_matrix])
        # import pdb;pdb.set_trace()
        intrins = results['cam_intrinsic']
        depth_map_list = []

        for cid in range(len(imgs)):
            # import pdb;pdb.set_trace()
            # lidar2lidarego = torch.tensor(results['lidar2ego']).to(torch.float32)
            # lidarego2global = torch.tensor(results['ego2global']).to(torch.float32)
            # cam2camego = torch.tensor(results['camera2ego'][cid])

            # camego2global = results['camego2global'][cid]

            # cam2img = torch.tensor(intrins[cid]).to(torch.float32)

            # lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
            #     lidarego2global.matmul(lidar2lidarego))
            # lidar2img = cam2img.matmul(lidar2cam)
            lidar2img = torch.tensor(results['lidar2img'][cid]).to(torch.float64)
            points_img = points_lidar[:, :3].matmul(
                lidar2img[:3, :3].T.to(torch.float64)) + lidar2img[:3, 3].to(torch.float64).unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            # points_img = points_img.matmul(
            #     post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[1],
                                             imgs.shape[2])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)

        ##################################################################
        i=0
        import cv2
        from PIL import Image
        for image_id in range(imgs.shape[0]):
            i+=1
            image = imgs[image_id]
            gt_depth_image = depth_map[image_id].numpy()

            gt_depth_image = np.expand_dims(gt_depth_image,2).repeat(3,2)

            #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
            im_color=cv2.applyColorMap(cv2.convertScaleAbs(gt_depth_image,alpha=15),cv2.COLORMAP_JET)
            #convert to mat png
            image[gt_depth_image>0] = im_color[gt_depth_image>0]
            im=Image.fromarray(np.uint8(image))
            #save image
            im.save('vis/visualize_{}.png'.format(i))
        #################################################################

        results['gt_depth_'] = depth_map
        depth_map_ = results["gt_depth"][0]
        depth_map = torch.tensor(depth_map)
        depth_map_ = torch.tensor(depth_map_)

        d1 = depth_map[0][depth_map[0]!=-1]
        d2 = depth_map_[0][depth_map_[0]!=-1]

        return results


@PIPELINES.register_module()
class DepthProbLabelGenerator(object):
    def __init__(
        self,
        max_depth=10,
        min_depth=0.25,
        num_depth=64,
        origin_stride=4,
        strides=[4, 8, 16, 32],
        depth_mode=0,
        image_hw=None,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.num_depth = num_depth
        self.origin_stride = origin_stride
        self.strides = [stride // origin_stride for stride in strides]
        self.depth_mode = depth_mode
        self.image_hw = np.array(image_hw)

    def points2depth(self, input_dict, invalid_value=-1):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = self.image_hw
            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, self.min_depth, self.max_depth)
            h, w = (int(H / self.origin_stride), int(W / self.origin_stride))
            u = np.floor(U / self.origin_stride).astype(np.int32)
            v = np.floor(V / self.origin_stride).astype(np.int32)
            depth_map = np.ones([h, w], dtype=np.float32) * invalid_value
            depth_map[v, u] = depths
            gt_depth.append(depth_map)

        return np.stack(gt_depth)

    def __call__(self, input_dict):
        if self.depth_mode == 0:
            return self.get_depth_gt_prob(input_dict)
        elif self.depth_mode == 1:
            return self.get_depth_gt_onehot(input_dict)
        else:
            assert False

    def get_depth_gt_prob(self, input_dict):
        depth = self.points2depth(input_dict, invalid_value=-1)
        depth = depth[:, None]
        depth_valid_mask = depth != -1
        depth = np.clip(
            depth,
            a_min=self.min_depth,
            a_max=self.max_depth,
        )
        depth_anchor = np.linspace(
            self.min_depth, self.max_depth, self.num_depth)[:, None, None]
        distance = np.abs(depth - depth_anchor)
        mask = distance < (depth_anchor[1] - depth_anchor[0])
        depth_gt = np.where(mask, depth_anchor, 0)
        y = depth_gt.sum(axis=1, keepdims=True) - depth_gt
        depth_prob_gt = np.where(
            (depth_gt != 0) & depth_valid_mask,
            (depth - y) / (depth_gt - y),
            0,
        )
        views, _, H, W = depth.shape
        gt = []
        for s in self.strides:
            gt_tmp = np.reshape(
                depth_prob_gt, (views, self.num_depth, H//s, s, W//s, s))
            gt_tmp = gt_tmp.sum(axis=-1).sum(axis=3)
            mask_tmp = depth_valid_mask.reshape(views, 1, H//s, s, W//s, s)
            mask_tmp = mask_tmp.sum(axis=-1).sum(axis=3)
            gt_tmp /= np.clip(mask_tmp, a_min=1, a_max=None)
            gt_tmp = gt_tmp.reshape(views, self.num_depth, -1)
            gt_tmp = np.transpose(gt_tmp, (0, 2, 1))
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        input_dict["depth_prob_gt"] = gt
        return input_dict

    def get_depth_gt_onehot(self, input_dict):
        depth = self.points2depth(input_dict, invalid_value=1e5)
        views, H, W = depth.shape
        depth_interval = (self.max_depth - self.min_depth) / (self.num_depth - 1)
        gt = []
        for s in self.strides:
            gt_tmp = np.reshape(
                depth, (views, H//s, s, W//s, s))
            gt_tmp = gt_tmp.transpose(0, 1, 3, 2, 4).reshape((views, H//s, W//s, -1))
            gt_tmp = gt_tmp.min(axis=-1)
            gt_tmp = (gt_tmp - self.min_depth) / depth_interval + 1
            gt_tmp = np.where(
                (gt_tmp >= 0) & (gt_tmp < self.num_depth + 1),
                gt_tmp,
                0,
            )
            gt_tmp = F.one_hot(
                torch.tensor(gt_tmp).long(),
                num_classes=self.num_depth + 1,
            )[..., 1:]
            gt_tmp = gt_tmp.reshape(views, -1, self.num_depth)
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        input_dict["depth_prob_gt"] = gt
        return input_dict

@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    def __init(self):
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict["lidar2img"])
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )

        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        for key in [
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False)

        # for key in [
        #     'gt_ego_fut_trajs',
        #     'gt_ego_fut_masks',
        #     'gt_ego_fut_cmd',
        #     'ego_status',
        # ]:
        #     if key not in input_dict:
        #         continue
        #     input_dict[key] = DC(to_tensor(input_dict[key]), stack=True, cpu_only=False, pad_dims=None)

        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][gt_bboxes_mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class ProcessRoute(object):
    def __init__(self, min_dis=4.0, max_dis=30.0, point_num=10):
        self.min_dis = min_dis
        self.max_dis = max_dis
        self.point_num = point_num

    def __call__(self, input_dict):
        route = input_dict["route"]
        route = self.process_route(route)
        input_dict["route"] = route
        # self.vis_route(input_dict)
        return input_dict

    def process_route(self, route):
        new_route = []
        last_point_in_range = route[0]
        for point in route:
            dis = np.linalg.norm(point)
            if dis > self.min_dis and dis < self.max_dis:
                new_route.append(point)
                last_point_in_range = point
        while len(new_route) < self.point_num:
            new_route.append(last_point_in_range)
        new_route = np.array(new_route)
        return new_route[:self.point_num]

    def vis_route(self, input):
        route = input["route"][:]
        token = input["token"]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.scatter(route[:, 0], route[:, 1], c='blue', s=120, )
        plt.xlim(-5, 5)
        plt.ylim(-3, 100)
        plt.savefig(f'vis/future_route_{token[-4:]}_after.png', dpi=150)


@PIPELINES.register_module()
class ProcessRoute2TwoTargetPoint(object):
    def __init__(self, min_dis=4.0, max_dis=20.0, point_num=2):
        self.min_dis = min_dis
        self.max_dis = max_dis
        self.point_num = point_num

    def __call__(self, input_dict):
        route = input_dict["route"]
        route = self.process_route(route)
        input_dict["route"] = route
        # self.vis_route(input_dict)
        return input_dict

    def process_route(self, route):
        new_route = []
        last_point_in_range = route[0]
        for point in route:
            dis = np.linalg.norm(point)
            if dis > self.min_dis and dis < self.max_dis:
                new_route.append(point)
                last_point_in_range = point
        while len(new_route) < self.point_num:
            new_route.append(last_point_in_range)
        tps = np.concatenate([new_route[0:1], new_route[-1:]], axis=0)
        return tps


import numpy as np
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners

# ---------------- 工具函数 ----------------
def get_yaw(traj):
    """
    用差分法求轨迹朝向角
    traj: (..., T, 2)  -> yaw: (..., T)
    """
    dxy = np.diff(traj, axis=-2)                      # (..., T-1, 2)
    yaw = np.arctan2(dxy[..., 1], dxy[..., 0])        # (..., T-1)
    yaw = np.concatenate([yaw[..., :1], yaw], axis=-1)  # 复制首帧
    return yaw

def get_yaw_ego(traj, start_yaw=np.pi/2):
    traj = torch.tensor(traj)
    start_yaw = torch.tensor(start_yaw)
    traj = torch.cat([torch.zeros_like(traj)[..., 0:1, :], traj], dim=-2)
    yaw = traj.new_zeros(traj.shape[:-1])
    yaw[..., 1:-1] = torch.atan2(
        traj[..., 2:, 1] - traj[..., :-2, 1],
        traj[..., 2:, 0] - traj[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj[..., -1, 1] - traj[..., -2, 1],
        traj[..., -1, 0] - traj[..., -2, 0],
    )
    yaw[..., 0] = start_yaw
    # for static object, estimated future yaw would be unstable
    start = traj[..., 0, :]
    end = traj[..., -1, :]
    dist = torch.linalg.norm(end - start, dim=-1)
    mask = dist < 0.2
    start_yaw = yaw[..., 0].unsqueeze(-1)
    yaw = torch.where(
        mask.unsqueeze(-1),
        start_yaw,
        yaw,
    )
    return yaw[..., 1:]


def get_yaw_motion(traj, box):
    traj = torch.tensor(traj)
    box = torch.tensor(box)
    start_yaw = box[..., 6]
    loc = box[..., :2]
    traj = torch.cat([loc[:,None], traj], dim=-2)

    yaw = traj.new_zeros(traj.shape[:-1])
    yaw[..., 1:-1] = torch.atan2(
        traj[..., 2:, 1] - traj[..., :-2, 1],
        traj[..., 2:, 0] - traj[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj[..., -1, 1] - traj[..., -2, 1],
        traj[..., -1, 0] - traj[..., -2, 0],
    )
    yaw[..., 0] = start_yaw
    # for static object, estimated future yaw would be unstable
    start = traj[..., 0, :]
    end = traj[..., -1, :]
    dist = torch.linalg.norm(end - start, dim=-1)
    mask = dist < 0.2
    start_yaw = yaw[..., 0].unsqueeze(-1)
    yaw = torch.where(
        mask.unsqueeze(-1),
        start_yaw,
        yaw,
    )
    return yaw[..., 1:]

def get_corners(box):
    """
    box: (..., 7)  (x, y, z, w, l, h, yaw)
    return: (..., 4, 2)  四个角点
    """
    x, y, _, w, l, _, yaw = [box[..., i] for i in range(7)]
    cos, sin = np.cos(yaw), np.sin(yaw)

    # 局部坐标系下的 4 个角
    dx = np.stack([ w/2,  w/2, -w/2, -w/2], axis=-1)
    dy = np.stack([ l/2, -l/2, -l/2,  l/2], axis=-1)

    # 旋转到全局
    rx = dx * cos[..., None] - dy * sin[..., None]
    ry = dx * sin[..., None] + dy * cos[..., None]

    return np.stack([x[..., None] + rx,
                     y[..., None] + ry], axis=-1)     # (..., 4, 2)

def sat_2d(corners_a, corners_b):
    """
    二维 SAT 碰撞检测（纯 NumPy 向量化）
    corners_a: (A, T, 4, 2)   自车所有 anchor 的角点
    corners_b: (B, T, 4, 2)   所有障碍物的角点
    return:    (A, B, T) bool  碰撞掩码
    """
    A, T = corners_a.shape[:2]
    B = corners_b.shape[0]

    # 1) 计算各自 4 条边
    edges_a = corners_a[:, :, [0,1,2,3], :] - corners_a[:, :, [1,2,3,0], :]  # (A, T, 4, 2)
    edges_b = corners_b[:, :, [0,1,2,3], :] - corners_b[:, :, [1,2,3,0], :]  # (B, T, 4, 2)

    # 2) 通过 tile 对齐到 (A, B, T, 4, 2)
    edges_a = np.tile(edges_a[:, None], (1, B, 1, 1, 1))   # (A, B, T, 4, 2)
    edges_b = np.tile(edges_b[None],   (A, 1, 1, 1, 1))    # (A, B, T, 4, 2)

    # 3) 拼接成 8 条边
    edges = np.concatenate([edges_a, edges_b], axis=-2)    # (A, B, T, 8, 2)

    # 4) 法向量
    axes = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)  # (A, B, T, 8, 2)
    axes = axes / (np.linalg.norm(axes, axis=-1, keepdims=True) + 1e-8)

    # 5) 投影：内积得到 (A, B, T, 4, 8)
    proj_a = np.einsum('abtpi,abtji->abtpj',
                       np.tile(corners_a[:, None], (1, B, 1, 1, 1)),
                       axes)
    proj_b = np.einsum('abtpi,abtji->abtpj',
                       np.tile(corners_b[None],   (A, 1, 1, 1, 1)),
                       axes)

    min_a, max_a = proj_a.min(axis=-2), proj_a.max(axis=-2)  # (A, B, T, 8)
    min_b, max_b = proj_b.min(axis=-2), proj_b.max(axis=-2)

    overlap = (max_a >= min_b) & (max_b >= min_a)
    return overlap.all(axis=-1)        # (A, B, T)

def sat_2d_cpu(corners_a: np.ndarray,
               corners_b: np.ndarray,
               aabb_thresh: float = 0.0):
    """
    2-D SAT 碰撞检测（NumPy + AABB 预剪枝）
    corners_a : (A, T, 4, 2)  float32/float64
    corners_b : (B, T, 4, 2)
    aabb_thresh : 外扩距离，默认 0
    return    : (A, B, T) bool  True=碰撞
    """
    A, T = corners_a.shape[:2]
    B = corners_b.shape[0]

    # ---------- 1. AABB 剪枝 ----------
    a_min = corners_a.min(axis=2)          # (A,T,2)
    a_max = corners_a.max(axis=2)
    b_min = corners_b.min(axis=2)          # (B,T,2)
    b_max = corners_b.max(axis=2)

    if aabb_thresh > 0:
        a_min -= aabb_thresh;  a_max += aabb_thresh
        b_min -= aabb_thresh;  b_max += aabb_thresh

    # 广播得 (A,B,T) 掩码
    no_overlap = (a_max[:, None, :, 0] < b_min[None, :, :, 0]) | \
                 (b_max[None, :, :, 0] < a_min[:, None, :, 0]) | \
                 (a_max[:, None, :, 1] < b_min[None, :, :, 1]) | \
                 (b_max[None, :, :, 1] < a_min[:, None, :, 1])
    valid_mask = ~no_overlap
    if valid_mask.sum() == 0:          # 全空提前返回
        return np.zeros((A, B, T), dtype=bool)

    # 稀疏三元组
    a_idx, b_idx, t_idx = np.where(valid_mask)      # (N,)
    corners_a_sparse = corners_a[a_idx, t_idx]      # (N,4,2)
    corners_b_sparse = corners_b[b_idx, t_idx]      # (N,4,2)

    # ---------- 2. 8 条边 ----------
    edges_a = corners_a_sparse[:, [0,1,2,3]] - corners_a_sparse[:, [1,2,3,0]]
    edges_b = corners_b_sparse[:, [0,1,2,3]] - corners_b_sparse[:, [1,2,3,0]]
    edges   = np.concatenate([edges_a, edges_b], axis=1)  # (N,8,2)

    # ---------- 3. 法向量 ----------
    axes = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)  # (N,8,2)
    axes /= (np.linalg.norm(axes, axis=-1, keepdims=True) + 1e-8)

    # ---------- 4. 投影 ----------
    proj_a = np.einsum('npi,nji->npj', corners_a_sparse, axes)  # (N,4,8)
    proj_b = np.einsum('npi,nji->npj', corners_b_sparse, axes)
    min_a, max_a = proj_a.min(axis=1), proj_a.max(axis=1)     # (N,8)
    min_b, max_b = proj_b.min(axis=1), proj_b.max(axis=1)

    # ---------- 5. 重叠 ----------
    overlap = (max_a >= min_b) & (max_b >= min_a)
    collide = overlap.all(axis=1)                             # (N,)

    # ---------- 6. 填回 ----------
    out = np.zeros((A, B, T), dtype=bool)
    out[a_idx, b_idx, t_idx] = collide
    return out

def sat_2d_cuda(corners_a: torch.Tensor,
                corners_b: torch.Tensor,
                aabb_thresh: float = 0.0):
    """
    2-D SAT 碰撞检测（纯 PyTorch + CUDA）
    corners_a : (A, T, 4, 2)  float32/float16
    corners_b : (B, T, 4, 2)
    aabb_thresh : 允许 AABB 外扩一点，默认 0
    return    : (A, B, T) bool  True=碰撞
    """
    corners_a = torch.tensor(corners_a).cuda()
    corners_b = torch.tensor(corners_b).cuda()
    device = corners_a.device
    dtype  = corners_a.dtype
    A, T, _, _ = corners_a.shape
    B = corners_b.shape[0]

    # ---------- 1. AABB 预剪枝 ----------
    # (A,T,2)  (minx/miny)
    a_min = corners_a.min(dim=2)[0]          # (A,T,2)
    a_max = corners_a.max(dim=2)[0]
    b_min = corners_b.min(dim=2)[0]          # (B,T,2)
    b_max = corners_b.max(dim=2)[0]

    # 外扩
    if aabb_thresh > 0:
        a_min -= aabb_thresh;  a_max += aabb_thresh
        b_min -= aabb_thresh;  b_max += aabb_thresh

    # 广播比较  (A,B,T)
    no_overlap = (a_max[:, None, :, 0] < b_min[None, :, :, 0]) | \
                 (b_max[None, :, :, 0] < a_min[:, None, :, 0]) | \
                 (a_max[:, None, :, 1] < b_min[None, :, :, 1]) | \
                 (b_max[None, :, :, 1] < a_min[:, None, :, 1])
    valid_mask = ~no_overlap                 # (A,B,T)
    # 如果全 False 可直接返回
    if valid_mask.sum() == 0:
        return torch.zeros((A, B, T), dtype=torch.bool, device=device)

    # 只保留需要 SAT 的 (a,b,t) 三元组 → 稀疏列表
    a_idx, b_idx, t_idx = torch.where(valid_mask)   # 1-D 长 ~N
    corners_a_sparse = corners_a[a_idx, t_idx]      # (N,4,2)
    corners_b_sparse = corners_b[b_idx, t_idx]      # (N,4,2)

    # ---------- 2. 计算 8 条边 ----------
    # (N,4,2)
    edges_a = corners_a_sparse[:, [0,1,2,3]] - corners_a_sparse[:, [1,2,3,0]]
    edges_b = corners_b_sparse[:, [0,1,2,3]] - corners_b_sparse[:, [1,2,3,0]]
    edges   = torch.cat([edges_a, edges_b], dim=1)  # (N,8,2)

    # ---------- 3. 法向量并归一化 ----------
    axes = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)  # (N,8,2)
    axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)

    # ---------- 4. 投影 ----------
    # (N,4,8)
    proj_a = torch.einsum('npi,nji->npj', corners_a_sparse, axes)
    proj_b = torch.einsum('npi,nji->npj', corners_b_sparse, axes)
    min_a, max_a = proj_a.min(dim=1)[0], proj_a.max(dim=1)[0]  # (N,8)
    min_b, max_b = proj_b.min(dim=1)[0], proj_b.max(dim=1)[0]

    # ---------- 5. 重叠判断 ----------
    overlap = (max_a >= min_b) & (max_b >= min_a)        # (N,8)
    collide = overlap.all(dim=1)                         # (N,)

    # ---------- 6. 填回稠密张量 ----------
    out = torch.zeros((A, B, T), dtype=torch.bool, device=device)
    out[a_idx, b_idx, t_idx] = collide
    return out

# ---------------- 主函数 ----------------
def anchor_collide_mask1(anchors,      # (num_anchor, 6, 2)
                        box_static,   # (num_box, 7)
                        traj,         # (num_box, 6, 2)
                        mask):        # (num_box, 6)  0 表示无效
    """
    返回 (num_anchor,) bool ndarray，True 表示该 anchor 与任意障碍物任意有效步碰撞
    """
    num_anchor, T = anchors.shape[:2]
    num_box = box_static.shape[0]

    # 1) 构造障碍物未来 box：(num_box, 6, 7)
    box_future = np.broadcast_to(box_static[:, None, :], (num_box, T, 7)).copy()
    box_future[..., :2] = traj
    box_future[..., 6] = get_yaw_motion(traj, box_static)
    invalid = mask == 0
    box_future[invalid] = 1e9          # 把无效步扔到远处

    # 2) 构造自车 anchor box：(num_anchor, 6, 7)
    ego_box = np.empty((num_anchor, T, 7))
    ego_box[..., :2] = anchors
    ego_box[..., [3, 4]] = 4.08, 1.73  # 固定尺寸 (w, l)
    ego_box[..., 6] = get_yaw_ego(anchors)
    # 3) 展开成 4 角点
    ego_corners = get_corners(ego_box)    # (num_anchor, 6, 4, 2)
    obs_corners = get_corners(box_future) # (num_box, 6, 4, 2)

    # 4) 碰撞检测
    collide = sat_2d_cpu(ego_corners, obs_corners)  # (num_anchor, num_box, 6)

    # 5) 汇总：只保留有效步
    collide &= mask.astype(bool)[None, :, :]
    return collide.any(axis=(1, 2))               # (num_anchor,)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def anchor_collide_mask(anchors,      # (num_anchor, 6, 2)
                        box_static,   # (num_box, 7)
                        traj,         # (num_box, 6, 2)
                        mask):        # (num_box, 6)  0 表示无效
    """
    返回 (num_anchor,) bool ndarray，True 表示该 anchor 与任意障碍物任意有效步碰撞
    """
    num_anchor, T = anchors.shape[:2]
    num_box = box_static.shape[0]
    # 1) 构造障碍物未来 box：(num_box, 6, 7)
    box_future = np.broadcast_to(box_static[:, None, :], (num_box, T, 7)).copy()
    box_future[..., :2] = traj
    box_future[..., 6] = get_yaw_motion(traj, box_static)
    invalid = mask == 0
    box_future[invalid] = 1e9          # 把无效步扔到远处
    # 2) 构造自车 anchor box：(num_anchor, 6, 7)
    ego_box = np.empty((num_anchor, T, 7))
    ego_box[..., :2] = anchors
    ego_box[..., [3, 4]] = 4.89, 1.83  # 固定尺寸 (w, l)
    # array([1.83671331, 4.89238167, 1.49027765])
    ego_box[..., 6] = get_yaw_ego(anchors)
    # 3) 展开成 4 角点
    ego_corners = get_corners(ego_box)    # (num_anchor, 6, 4, 2)
    obs_corners = get_corners(box_future) # (num_box, 6, 4, 2)

    # 4) 碰撞检测
    collide = sat_2d_cpu(ego_corners, obs_corners)  # (num_anchor, num_box, 6)
    # 5) 汇总：只保留有效步
    collide &= mask.astype(bool)[None, :, :]
    collide = collide.any(axis=(1, 2))
    # 可视化部分 - 选择一个示例进行绘制
    # visualize_corners_with_collision(ego_corners, obs_corners, mask, collide)

    return collide             # (num_anchor,)

def visualize_corners_with_collision(ego_corners, obs_corners, mask, collide):
    """
    可视化 ego_corners 的最后1024个和全部的 obs_corners，根据碰撞情况用不同颜色
    """
    # 获取 ego_corners 的最后1024个（如果不足1024个则取全部）
    num_ego = ego_corners.shape[0]
    start_idx = max(0, num_ego - 1024)
    ego_corners = ego_corners.reshape(1024, 45, -1, 4, 2)
    collide = collide.reshape(1024, 45)
    ego_corners_subset = ego_corners[:, -1]
    collide_subset = collide[:, -1]

    fig, ax = plt.subplots(figsize=(14, 12))

    # 统计碰撞情况
    num_collide = np.sum(collide_subset)
    num_no_collide = len(collide_subset) - num_collide

    # 绘制自车轨迹的最后1024个，根据碰撞情况用不同颜色
    for ego_idx in range(ego_corners_subset.shape[0]):
        collision_status = collide_subset[ego_idx]
        color = 'red' if collision_status else 'green'
        alpha = 0.6 if collision_status else 0.3
        label_collide = 'Ego (Collision)' if collision_status and ego_idx == 0 else ""
        label_no_collide = 'Ego (No Collision)' if not collision_status and ego_idx == 0 else ""

        for t in range(ego_corners_subset.shape[1]):
            # 绘制自车矩形
            ego_corner = ego_corners_subset[ego_idx, t]
            ego_rect = patches.Polygon(ego_corner, closed=True,
                                      fill=False, color=color, linewidth=1,
                                      alpha=alpha,
                                      label=label_collide if collision_status else label_no_collide)
            ax.add_patch(ego_rect)

            # 只在第一个和最后一个时间步标记中心点
            if t == 0 or t == ego_corners_subset.shape[1] - 1:
                center = ego_corner.mean(axis=0)
                marker = 'o' if t == 0 else 's'
                ax.plot(center[0], center[1], marker=marker, color=color,
                       markersize=3, alpha=0.8)

    # 绘制全部障碍物轨迹的有效时间步
    for obs_idx in range(obs_corners.shape[0]):
        for t in range(obs_corners.shape[1]):
            if mask[obs_idx, t] == 0:  # 跳过无效步
                continue

            # 绘制障碍物矩形
            obs_corner = obs_corners[obs_idx, t]
            obs_rect = patches.Polygon(obs_corner, closed=True,
                                      fill=False, color='blue', linewidth=1.5,
                                      linestyle='--', alpha=0.7,
                                      label='Obstacle' if obs_idx == 0 and t == 0 else "")
            ax.add_patch(obs_rect)

            # 在矩形中心添加障碍物和时间步标记
            center = obs_corner.mean(axis=0)
            ax.text(center[0], center[1], f'O{obs_idx}-T{t}', fontsize=6,
                    ha='center', va='center', color='blue', alpha=0.8)

    # 设置图形属性
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Ego Trajectories (last {ego_corners_subset.shape[0]}) vs All Obstacle Trajectories\n'
                f'Collision: {num_collide}, No Collision: {num_no_collide}')

    # 创建图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label=f'Ego Collision ({num_collide})'),
        Line2D([0], [0], color='green', linewidth=2, label=f'Ego No Collision ({num_no_collide})'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label=f'Obstacles ({obs_corners.shape[0]})')
    ]
    ax.legend(handles=legend_elements)

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 自动调整坐标轴范围
    all_ego_points = ego_corners_subset.reshape(-1, 2)
    all_obs_points = obs_corners.reshape(-1, 2)
    all_obs_points[all_obs_points>1e3] = 0
    all_points = np.vstack([all_ego_points, all_obs_points])

    x_min, y_min = all_points.min(axis=0) - 5
    x_max, y_max = all_points.max(axis=0) + 5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 保存图片
    import time
    plt.savefig(f'vis/collision_visualization_with_{time.time()}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"碰撞状态可视化图片已保存为 'collision_visualization_with_status.png'")
    print(f"自车轨迹: {ego_corners_subset.shape[0]} 个 (碰撞: {num_collide}, 无碰撞: {num_no_collide})")
    print(f"障碍物轨迹: {obs_corners.shape[0]} 个")

def visualize_corners1(ego_corners, obs_corners, mask):
    """
    可视化 ego_corners 的最后1024个和全部的 obs_corners 代表的矩形
    """
    # 获取 ego_corners 的最后1024个（如果不足1024个则取全部）
    num_ego = ego_corners.shape[0]
    start_idx = max(0, num_ego - 1024)
    ego_corners_subset = ego_corners[start_idx:]

    fig, ax = plt.subplots(figsize=(14, 12))

    # 绘制自车轨迹的最后1024个
    for ego_idx in range(ego_corners_subset.shape[0]):
        for t in range(ego_corners_subset.shape[1]):
            # 绘制自车矩形
            ego_corner = ego_corners_subset[ego_idx, t]
            ego_rect = patches.Polygon(ego_corner, closed=True,
                                      fill=False, color='blue', linewidth=1, 
                                      alpha=0.3, label='Ego' if ego_idx == 0 and t == 0 else "")
            ax.add_patch(ego_rect)

    # 绘制全部障碍物轨迹的有效时间步
    for obs_idx in range(obs_corners.shape[0]):
        for t in range(obs_corners.shape[1]):
            if mask[obs_idx, t] == 0:  # 跳过无效步
                continue

            # 绘制障碍物矩形
            obs_corner = obs_corners[obs_idx, t]
            obs_rect = patches.Polygon(obs_corner, closed=True,
                                      fill=False, color='red', linewidth=1.5,
                                      linestyle='--', alpha=0.7,
                                      label='Obstacle' if obs_idx == 0 and t == 0 else "")
            ax.add_patch(obs_rect)

            # 在矩形中心添加障碍物和时间步标记
            center = obs_corner.mean(axis=0)
            ax.text(center[0], center[1], f'O{obs_idx}-T{t}', fontsize=6,
                    ha='center', va='center', color='red', alpha=0.8)

    # 设置图形属性
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Ego Trajectories (last {ego_corners_subset.shape[0]}) vs All Obstacle Trajectories')

    # 创建图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label=f'Ego (last {ego_corners_subset.shape[0]})'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label=f'Obstacles ({obs_corners.shape[0]})')
    ]
    ax.legend(handles=legend_elements)

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 自动调整坐标轴范围
    all_ego_points = ego_corners_subset.reshape(-1, 2)
    all_obs_points = obs_corners.reshape(-1, 2)
    all_obs_points[all_obs_points>1e3] = 0
    all_points = np.vstack([all_ego_points, all_obs_points])

    x_min, y_min = all_points.min(axis=0) - 5
    x_max, y_max = all_points.max(axis=0) + 5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # 保存图片
    import time
    plt.savefig(f'vis/collision_visualization_last1024_{time.time()}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"可视化图片已保存为 'collision_visualization_last1024.png'")
    print(f"显示了 {ego_corners_subset.shape[0]} 个自车轨迹和 {obs_corners.shape[0]} 个障碍物轨迹")
    import ipdb; ipdb.set_trace()


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

def visualize_collision(anchors,
                        box_static,
                        traj,
                        mask,
                        collide_mask):
    """
    障碍物：每个 box 分配唯一颜色（循环）
    自车轨迹：
        未碰撞 -> 蓝色细线
        碰撞   -> 红色粗线
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(True)

    T = anchors.shape[1]
    num_box = box_static.shape[0]

    # 1. 给每个障碍物分配颜色
    colors = cm.tab10.colors           # 最多 10 种，循环使用
    box_colors = [colors[i % len(colors)] for i in range(num_box)]

    # 2. 画障碍物
    for b in range(num_box):
        for t in range(T):
            if mask[b, t] == 0:
                continue
            x, y = traj[b, t]
            w, l, yaw = box_static[b, 3], box_static[b, 4], box_static[b, 6]

            # 计算左下角坐标
            dx = -w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
            dy = -w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

            rect = Rectangle((x + dx, y + dy), w, l,
                             angle=np.degrees(yaw),
                             linewidth=1.5,
                             edgecolor=box_colors[b],
                             facecolor='none',
                             label=f'box_{b}' if t == 0 else None)
            ax.add_patch(rect)

    # 3. 画自车轨迹
    for a in range(anchors.shape[0]):
        xs, ys = anchors[a, :, 0], anchors[a, :, 1]
        if collide_mask[a]:
            ax.plot(xs, ys, color='red', linewidth=2.5, zorder=10)
        else:
            ax.plot(xs, ys, color='blue', linewidth=0.8, alpha=0.6)

    # 4. 图例：去重
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Collision Visualization (obstacles colored)')
    plt.tight_layout()
    import time
    plt.savefig(f"vis/col/{time.time()}.png")

@PIPELINES.register_module()
class GetColLabel(object):
    def __init__(self, anchor):
        self.anchor = np.load(anchor)

    def __call__(self, input_dict):
        col = anchor_collide_mask(
            self.anchor,
            input_dict["gt_bboxes_3d"][..., :7], 
            input_dict["gt_agent_fut_trajs"].cumsum(axis=1)+input_dict["gt_bboxes_3d"][..., None, :2],
            input_dict["gt_agent_fut_masks"],
        )
        input_dict["collision"] = col
        return input_dict
        # num_box = input_dict["gt_bboxes_3d"].shape[0]
        # col1 = rescore(
        #     None,
        #     self.anchor[None],
        #     np.ones((1, num_box, 1)),
        #     input_dict["gt_agent_fut_trajs"][None,:,None],
        #     input_dict["gt_bboxes_3d"][None],
        #     np.ones((1, num_box)),
        #     input_dict["gt_agent_fut_masks"][None],
        # )[0].numpy()
        # print((col!=col1).sum(), col.sum())
        # visualize_collision(
        #     self.anchor,
        #     input_dict["gt_bboxes_3d"][..., :7],
        #     input_dict["gt_agent_fut_trajs"].cumsum(axis=1)+input_dict["gt_bboxes_3d"][..., None, :2],
        #     input_dict["gt_agent_fut_masks"],
        #     col,
        # )

# 修正版 interp_paths_speeds（去掉 concat，用严格 1 m 刻度）
def interp_paths_speeds_fix(paths, speeds, T):
    num_mode, num_point, _ = paths.shape
    s_path = np.arange(num_point, dtype=float)      # 0,1,2,…,N-1
    s_query = (speeds.reshape(1,-1) * T.reshape(-1,1)).ravel()
    s_query = np.clip(s_query, 0, num_point-1)
    out = np.empty((num_mode, speeds.size, T.size, 2))
    for m in range(num_mode):
        x = np.interp(s_query, s_path, paths[m,:,0])
        y = np.interp(s_query, s_path, paths[m,:,1])
        out[m] = np.stack([x, y], -1).reshape(speeds.size, T.size, 2)
    return out

def interp_paths_speeds(paths: np.ndarray,
                            speeds: np.ndarray,
                            T: np.ndarray) -> np.ndarray:
    """
    绝对里程版，风格保持与 equal_spacing_route 一致。
    返回 (num_mode, num_speeds, num_time, 2) 的绝对坐标。
    """
    # paths = np.concatenate((np.zeros_like(paths[:, :1]),  paths), axis=1) # Add 0 to front
    num_mode, num_point, _ = paths.shape
    num_speeds = speeds.size
    num_time = T.size

    # 1. 准备查询里程：s = v * t，拉成一维便于后续插值
    s_query = np.clip(np.outer(speeds, T), 0, None)          # (num_time, num_speeds)
    s_query_flat = s_query.ravel()                           # (T*S,)
    # 2. 对每条路径计算累积弧长
    result = np.empty((num_mode, num_speeds, num_time, 2))
    for m in range(num_mode):
        route = paths[m]                                     # (P, 2)
        route = np.concatenate((np.zeros_like(route[:1]),  route), axis=0) # Add 0 to front

        # 2.1 构造 shift 并计算段长
        shift = np.roll(route, 1, axis=0)
        shift[0] = shift[1]                                  # 把绕回值改掉
        seg_len = np.linalg.norm(route - shift, axis=1)      # (P,)
        # 2.2 累积里程并加微小量保证严格递增
        cum_len = np.cumsum(seg_len)
        cum_len += np.arange(len(cum_len)) * 1e-4
        # 2.3 分别插值 x、y
        x_interp = np.interp(s_query_flat, cum_len, route[:, 0])
        y_interp = np.interp(s_query_flat, cum_len, route[:, 1])
        # 2.4 reshape 回 (S, T, 2) 并写入结果
        coords_m = np.stack([x_interp, y_interp], axis=-1).reshape(num_speeds, num_time, 2)
        result[m] = coords_m

    return result                                           # (M, S, T, 2)


def generate_trajectories_correct(paths, speeds, T):
    """
    正确处理等距曲线路径的轨迹生成
    
    Args:
        paths: (1024, 15, 2) 等距路径点，点间距为1米
        speeds: (45,) 候选速度 (m/s)
        T: (3,) 时间点 [t1, t2, t3] (秒)
    
    Returns:
        trajectories: (1024, 45, 3, 2) 轨迹点
    """
    batch_size, num_points, _ = paths.shape
    paths = torch.cat([torch.zeros(batch_size, 1, 2), paths], dim=1)
    batch_size, num_points, _ = paths.shape
    num_speeds = speeds.shape[0]
    num_times = T.shape[0]

    # 计算目标行驶距离（米）和对应的浮点索引
    target_distances = speeds.view(1, num_speeds, 1) * T.view(1, 1, num_times)  # (1, 45, 3)
    point_indices = torch.clamp(target_distances, 0, num_points - 1 - 1e-6)

    # 分离整数和小数部分
    lower_indices = point_indices.floor().long()  # (1, 45, 3)
    ratios = point_indices - lower_indices.float()  # (1, 45, 3)

    # 扩展维度用于广播
    # 创建合适的索引张量
    batch_indices = torch.arange(batch_size).view(batch_size, 1, 1, 1)  # (1024, 1, 1, 1)

    # 使用 gather 方法正确收集点
    # 扩展 lower_indices 以匹配 paths 的形状
    lower_indices_expanded = lower_indices.expand(batch_size, num_speeds, num_times)  # (1024, 45, 3)

    # 获取线段起点
    start_points = torch.gather(
        paths.unsqueeze(1).unsqueeze(2).expand(batch_size, num_speeds, num_times, num_points, 2),
        dim=3,
        index=lower_indices_expanded.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_speeds, num_times, 1, 2)
    ).squeeze(3)  # (1024, 45, 3, 2)

    # 获取线段终点（下一个点）
    end_indices = torch.clamp(lower_indices_expanded + 1, 0, num_points - 1)
    end_points = torch.gather(
        paths.unsqueeze(1).unsqueeze(2).expand(batch_size, num_speeds, num_times, num_points, 2),
        dim=3,
        index=end_indices.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_speeds, num_times, 1, 2)
    ).squeeze(3)  # (1024, 45, 3, 2)

    # 线性插值计算最终位置
    trajectories = start_points + ratios.unsqueeze(-1) * (end_points - start_points)

    return trajectories

def generate_trajectories_simple_fixed(paths, speeds, T):
    """
    简单且正确的实现方案
    """
    batch_size, num_points, _ = paths.shape
    paths = torch.cat([torch.zeros(batch_size, 1, 2), paths], dim=1)
    batch_size, num_points, _ = paths.shape
    num_speeds = speeds.shape[0]
    num_times = T.shape[0]

    # 计算目标距离和对应的浮点索引
    target_distances = speeds.view(1, num_speeds, 1) * T.view(1, 1, num_times)
    point_indices = torch.clamp(target_distances, 0, num_points - 1 - 1e-6)

    # 分离整数和小数部分
    lower_indices = point_indices.floor().long()  # (1, 45, 3)
    ratios = point_indices - lower_indices.float()  # (1, 45, 3)

    # 使用简单的循环（虽然慢一些，但确保正确）
    trajectories = torch.zeros(batch_size, num_speeds, num_times, 2, device=paths.device)

    for i in range(batch_size):
        for j in range(num_speeds):
            for k in range(num_times):
                idx = lower_indices[0, j, k].item()
                ratio = ratios[0, j, k].item()

                # 获取线段起点和终点
                start_point = paths[i, idx]
                end_point = paths[i, min(idx + 1, num_points - 1)]

                # 线性插值
                trajectories[i, j, k] = start_point + ratio * (end_point - start_point)

    return trajectories


@PIPELINES.register_module()
class GetSpatialColLabel(object):
    def __init__(self, plan_config, speed_intervals, T):
        self.plan_config = plan_config
        self.speed_intervals = torch.tensor(speed_intervals)
        self.T = torch.tensor(T)

        self.speed_intervals = np.array(speed_intervals)
        self.T = np.array(T)

    def __call__(self, input_dict):
        for key, value in self.plan_config.items():
            if "col" not in key:
                continue
            if not hasattr(self, key + "_anchor"):
                setattr(self, key + "_anchor", np.load(value["anchor"]))
            anchor = getattr(self, key + "_anchor")
            trajs = interp_paths_speeds(anchor, self.speed_intervals, self.T)
            num_mode, num_speeds, num_times = trajs.shape[:3]
            trajs_ = trajs.reshape(-1, num_times, 2)
            col = anchor_collide_mask(
                trajs_,
                input_dict["gt_bboxes_3d"][..., :7], 
                (input_dict["gt_agent_fut_trajs"].cumsum(axis=1)+input_dict["gt_bboxes_3d"][..., None, :2])[:, 0:2],
                input_dict["gt_agent_fut_masks"][:, 0:2],
            )
            col = col.reshape(num_mode, num_speeds)
            input_dict[f"gt_{key}"] = col
            # visualize_collision(
            #     trajs[:, -1],
            #     input_dict["gt_bboxes_3d"][..., :7],
            #     (input_dict["gt_agent_fut_trajs"].cumsum(axis=1)+input_dict["gt_bboxes_3d"][..., None, :2])[:, 0:2],
            #     input_dict["gt_agent_fut_masks"][:, 0:2],
            #     col[:, -1],
            # )
        return input_dict
        # num_box = input_dict["gt_bboxes_3d"].shape[0]
        # col1 = rescore(
        #     None,
        #     self.anchor[None],
        #     np.ones((1, num_box, 1)),
        #     input_dict["gt_agent_fut_trajs"][None,:,None],
        #     input_dict["gt_bboxes_3d"][None],
        #     np.ones((1, num_box)),
        #     input_dict["gt_agent_fut_masks"][None],
        # )[0].numpy()
        # print((col!=col1).sum(), col.sum())
        # visualize_collision(
        #     self.anchor,
        #     input_dict["gt_bboxes_3d"][..., :7],
        #     input_dict["gt_agent_fut_trajs"].cumsum(axis=1)+input_dict["gt_bboxes_3d"][..., None, :2],
        #     input_dict["gt_agent_fut_masks"],
        #     col,
        # )


def rescore(
    plan_cls,
    plan_reg,
    motion_cls,
    motion_reg,
    det_anchors,
    det_confidence,
    mask,
    score_thresh=0.5,
    static_dis_thresh=0.5,
    dim_scale=1.,
    num_motion_mode=1,
    offset=0.,
):
    X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
    CNS, YNS = 0, 1  # centerness and yawness indices in quality
    YAW = 6  # decoded
    det_anchors = torch.tensor(det_anchors, dtype=torch.float32)
    motion_cls = torch.tensor(motion_cls, dtype=torch.float32)
    det_confidence = torch.tensor(det_confidence, dtype=torch.float32)
    motion_reg = torch.tensor(motion_reg, dtype=torch.float32)

    def cat_with_zero(traj):
        traj = torch.tensor(traj, dtype=torch.float32)
        zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
        traj_cat = torch.cat([zeros, traj], dim=-2)
        return traj_cat

    def get_yaw(traj, start_yaw=np.pi/2):
        yaw = traj.new_zeros(traj.shape[:-1])
        yaw[..., 1:-1] = torch.atan2(
            traj[..., 2:, 1] - traj[..., :-2, 1],
            traj[..., 2:, 0] - traj[..., :-2, 0],
        )
        yaw[..., -1] = torch.atan2(
            traj[..., -1, 1] - traj[..., -2, 1],
            traj[..., -1, 0] - traj[..., -2, 0],
        )
        yaw[..., 0] = start_yaw
        # for static object, estimated future yaw would be unstable
        start = traj[..., 0, :]
        end = traj[..., -1, :]
        dist = torch.linalg.norm(end - start, dim=-1)
        mask = dist < static_dis_thresh
        start_yaw = yaw[..., 0].unsqueeze(-1)
        yaw = torch.where(
            mask.unsqueeze(-1),
            start_yaw,
            yaw,
        )
        return yaw.unsqueeze(-1)

    ## ego
    bs = plan_reg.shape[0]
    plan_reg_cat = cat_with_zero(plan_reg)
    ego_box = det_anchors.new_zeros(bs, 1024, 6 + 1, 7)
    ego_box[..., [X, Y]] = plan_reg_cat
    ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
    ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

    ## motion
    motion_reg = motion_reg[..., :6, :].cumsum(-2)
    motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
    _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
    motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, 6 + 1, 2)
    motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

    motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
    motion_box[..., [X, Y]] = motion_reg
    motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]]
    box_yaw = torch.atan2(
        det_anchors[..., SIN_YAW],
        det_anchors[..., COS_YAW],
    )
    motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

    filter_mask = det_confidence < score_thresh
    motion_box[filter_mask] = 1e6

    ego_box = ego_box[..., 1:, :]
    motion_box = motion_box[..., 1:, :]

    bs, num_ego_mode, ts, _ = ego_box.shape
    bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
    ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
    motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

    # ego_box[0] += offset * torch.cos(ego_box[6])
    # ego_box[1] += offset * torch.sin(ego_box[6])
    ego_box[..., 0] += offset * torch.cos(ego_box[..., 6])
    ego_box[..., 1] += offset * torch.sin(ego_box[..., 6])
    col = check_collision(ego_box, motion_box)
    col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
    col = col * mask[:, None, :, None]
    col = col.flatten(2, -1).any(dim=-1)

    return col


def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection: 
            check if any corner point of boxes1 is inside boxes2 and vice versa.
        
        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision

def corners_in_box(boxes1, boxes2):
    if  boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    boxes1_yaw = boxes1[:, 6].clone()
    boxes1_loc = boxes1[:, :3].clone()
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)
    rot_mat_T = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ]
    )
    # translate and rotate boxes
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision

@PIPELINES.register_module(force=True)
class Collect:
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            if key not in results:
                # print(f"Collect {key} not exist")
                continue
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
