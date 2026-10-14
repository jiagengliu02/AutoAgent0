import torch

import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from PIL import Image


@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img, mat = self._img_transform(
                np.uint8(imgs[i]), aug_config,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            if "cam_intrinsic" in results:
            #     results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]
                results["cam_intrinsic"][i][:3, :3] = (
                    mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]
                )

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

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

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix


@PIPELINES.register_module()
class BBoxRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                results["lidar2img"][view] @ rot_mat_inv
            )
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(
                results["gt_bboxes_3d"], angle
            )
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d

@PIPELINES.register_module()
class BBoxMapRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                results["lidar2img"][view] @ rot_mat_inv
            )
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(
                results["gt_bboxes_3d"], angle
            )

        if "map_geoms" in results:
            map_geoms_rotate = {}
            for label, geom_list in results["map_geoms"].items():
                map_geoms_rotate[label] = []
                for geom in geom_list:
                    geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
                    map_geoms_rotate[label].append(geom_rotate)
            results["map_geoms"] = map_geoms_rotate

        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d

# @PIPELINES.register_module()
# class BBoxMapTrajRotation(object):
#     def __call__(self, results):
#         angle = results["aug_config"]["rotate_3d"]
#         rot_cos = np.cos(angle)
#         rot_sin = np.sin(angle)

#         rot_mat = np.array(
#             [
#                 [rot_cos, -rot_sin, 0, 0],
#                 [rot_sin, rot_cos, 0, 0],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1],
#             ]
#         )
#         rot_mat_inv = np.linalg.inv(rot_mat)

#         num_view = len(results["lidar2img"])
#         for view in range(num_view):
#             results["lidar2img"][view] = (
#                 results["lidar2img"][view] @ rot_mat_inv
#             )
#         if "lidar2global" in results:
#             results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
#         if "gt_bboxes_3d" in results:
#             gt_box_clone = np.copy(results["gt_bboxes_3d"])
#             results["gt_bboxes_3d"] = self.box_rotate(
#                 results["gt_bboxes_3d"], angle
#             )

#         if "map_geoms" in results:
#             map_geoms_rotate = {}
#             for label, geom_list in results["map_geoms"].items():
#                 map_geoms_rotate[label] = []
#                 for geom in geom_list:
#                     geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
#                     map_geoms_rotate[label].append(geom_rotate)
#             results["map_geoms"] = map_geoms_rotate

#         ## from now on, traj rotation:
#         gt_agent_fut_trajs = results["gt_agent_fut_trajs"].cumsum(axis=-2) + gt_box_clone[:, None, :2] ## num_box, timestep, 2
#         gt_agent_fut_masks = results["gt_agent_fut_masks"]
#         gt_ego_fut_trajs = results["gt_ego_fut_trajs"].cumsum(axis=-2) ## timestep, 2
#         gt_ego_fut_masks = results["gt_ego_fut_masks"]

#         return results

#     @staticmethod
#     def box_rotate(bbox_3d, angle):
#         rot_cos = np.cos(angle)
#         rot_sin = np.sin(angle)
#         rot_mat_T = np.array(
#             [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
#         )
#         bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
#         bbox_3d[:, 6] += angle
#         if bbox_3d.shape[-1] > 7:
#             vel_dims = bbox_3d[:, 7:].shape[-1]
#             bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
#         return bbox_3d


import numpy as np
import shapely
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import time

@PIPELINES.register_module()
class BBoxMapTrajRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = results["lidar2img"][view] @ rot_mat_inv

        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv

        # Save original boxes for visualization
        gt_boxes_orig = None
        if "gt_bboxes_3d" in results:
            gt_boxes_orig = np.copy(results["gt_bboxes_3d"])
            results["gt_bboxes_3d"] = self.box_rotate(results["gt_bboxes_3d"], angle)

        # Save original map for visualization
        map_geoms_orig = None
        if "map_geoms" in results:
            map_geoms_orig = results["map_geoms"].copy()
            map_geoms_rotate = {}
            for label, geom_list in results["map_geoms"].items():
                map_geoms_rotate[label] = []
                for geom in geom_list:
                    geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
                    map_geoms_rotate[label].append(geom_rotate)
            results["map_geoms"] = map_geoms_rotate

        # ====== Trajectory Rotation ======
        # Save original trajectories for visualization
        gt_agent_fut_trajs_orig = None
        gt_ego_fut_trajs_orig = None

        if "gt_agent_fut_trajs" in results:
            rot_mat_T_2d = rot_mat[:2, :2].T

            num_box = gt_boxes_orig.shape[0]
            gt_agent_fut_trajs_orig = np.concatenate((np.zeros((num_box, 1, 2)), results["gt_agent_fut_trajs"]), axis=1)
            gt_agent_fut_trajs_orig = gt_agent_fut_trajs_orig.cumsum(axis=-2) + gt_boxes_orig[:, None, :2] ## num_box, timestep, 2
            gt_agent_fut_masks = results["gt_agent_fut_masks"]

            gt_ego_fut_trajs_orig = np.concatenate((np.zeros((1, 2)), results["gt_ego_fut_trajs"]), axis=0)
            gt_ego_fut_trajs_orig = gt_ego_fut_trajs_orig.cumsum(axis=-2) ## timestep, 2
            gt_ego_fut_masks = results["gt_ego_fut_masks"]

            results["gt_agent_fut_trajs"] = (results["gt_agent_fut_trajs"] @ rot_mat_T_2d).astype(np.float32)
            results["gt_ego_fut_trajs"] = (results["gt_ego_fut_trajs"] @ rot_mat_T_2d).astype(np.float32)

            gt_agent_fut_trajs_rot = np.concatenate((np.zeros((num_box, 1, 2)), results["gt_agent_fut_trajs"]), axis=1)
            gt_agent_fut_trajs_rot = gt_agent_fut_trajs_rot.cumsum(axis=-2) + results["gt_bboxes_3d"][:, None, :2] ## num_box, timestep, 2
            gt_ego_fut_trajs_rot = np.concatenate((np.zeros((1, 2)), results["gt_ego_fut_trajs"]), axis=0)
            gt_ego_fut_trajs_rot = gt_ego_fut_trajs_rot.cumsum(axis=-2) ## timestep, 2

            results["gt_trajs_orig"] = {
                "agent": gt_agent_fut_trajs_orig,
                "ego": gt_ego_fut_trajs_orig
            }
            results["gt_trajs_rot"] = {
                "agent": gt_agent_fut_trajs_rot,
                "ego": gt_ego_fut_trajs_rot
            }

        # ====== Visualization ======
        # self.visualize_rotation(
        #     results,
        #     gt_boxes_orig,
        #     map_geoms_orig,
        #     gt_agent_fut_trajs_orig,
        #     gt_ego_fut_trajs_orig
        # )
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d

    def visualize_rotation(self, results, gt_boxes_orig, map_geoms_orig,
                           gt_agent_trajs_orig, gt_ego_trajs_orig):
        """Visualize ground truths before and after rotation"""
        try:
            if "gt_trajs_rot" not in results:
                return

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle(f"Rotation: {results['aug_config']['rotate_3d']:.2f} rad", fontsize=16)

            # ===== Before Rotation =====
            ax1.set_title("Before Rotation")
            self._plot_scene(
                ax1,
                map_geoms_orig,
                gt_boxes_orig,
                gt_agent_trajs_orig,
                gt_ego_trajs_orig
            )

            # ===== After Rotation =====
            ax2.set_title("After Rotation")
            self._plot_scene(
                ax2,
                results.get("map_geoms", None),
                results.get("gt_bboxes_3d", None),
                results["gt_trajs_rot"]["agent"],
                results["gt_trajs_rot"]["ego"]
            )

            # Save figure
            os.makedirs("viz_rotations", exist_ok=True)
            timestamp = int(time.time() * 1000)
            plt.savefig(f"viz_rotations/{timestamp}.png")
            plt.close(fig)
        except Exception as e:
            print(f"Visualization failed: {str(e)}")

    def _plot_scene(self, ax, map_geoms, gt_boxes, agent_trajs, ego_trajs):
        """Plot single scene on given axis"""
        # Plot map geometries
        if map_geoms is not None:
            for label, geom_list in map_geoms.items():
                for geom in geom_list:
                    if geom.geom_type == "LineString":
                        x, y = geom.xy
                        ax.plot(x, y, 'b-', linewidth=1, alpha=0.5)
                    elif geom.geom_type == "Polygon":
                        x, y = geom.exterior.xy
                        ax.plot(x, y, 'g-', linewidth=1, alpha=0.3)

        # Plot agent trajectories
        if agent_trajs is not None:
            for i, traj in enumerate(agent_trajs):
                ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1.5, alpha=0.7)
                ax.scatter(traj[0, 0], traj[0, 1], c='r', marker='o', s=30)

        # Plot ego trajectory
        if ego_trajs is not None:
            ax.plot(ego_trajs[:, 0], ego_trajs[:, 1], 'g-', linewidth=2.5)
            ax.scatter(ego_trajs[0, 0], ego_trajs[0, 1], c='g', marker='s', s=80)
            ax.scatter(ego_trajs[-1, 0], ego_trajs[-1, 1], c='purple', marker='*', s=100)

        # Plot bounding boxes
        if gt_boxes is not None and len(gt_boxes) > 0:
            for box in gt_boxes:
                center_x, center_y = box[:2]
                length, width = box[3], box[4]
                angle = box[6]

                # Calculate corner points
                dx = length / 2
                dy = width / 2
                corners = np.array([
                    [-dx, -dy],
                    [-dx, dy],
                    [dx, dy],
                    [dx, -dy],
                    [-dx, -dy]
                ])

                # Rotation matrix
                rot_mat = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])

                # Rotate and translate corners
                rotated_corners = corners @ rot_mat.T + np.array([center_x, center_y])

                # Plot bounding box
                ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'm-', linewidth=1.5)

        # Set plot properties
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.axis("equal")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)


@PIPELINES.register_module()
class MapRotation(object):
    def __init__(self, rot_range=[-0.3925, 0.3925]):
        self.rot_range = rot_range

    def __call__(self, results):
        angle = np.random.uniform(*self.rot_range)
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        map_geoms_rotate = {}
        for label, geom_list in results["map_geoms"].items():
            map_geoms_rotate[label] = []
            for geom in geom_list:
                geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
                map_geoms_rotate[label].append(geom_rotate)
        results["map_geoms"] = map_geoms_rotate

        return results


import os
import numpy as np
import matplotlib.pyplot as plt
import shapely
import shapely.affinity
from shapely.geometry import LineString, Polygon
from mmcv.utils import mkdir_or_exist

@PIPELINES.register_module()
class MapRotationVis(object):
    def __init__(self, rot_range=[-0.3925, 0.3925], vis_dir=None):
        self.rot_range = rot_range
        self.vis_dir = './vis'
        if vis_dir is not None:
            mkdir_or_exist(vis_dir)

    def plot_geoms(self, geoms_dict, filepath):
        """绘制并保存几何图形"""
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        for label, geom_list in geoms_dict.items():
            for geom in geom_list:
                if isinstance(geom, LineString):
                    x, y = geom.xy
                    ax.plot(x, y, label=label)
                elif isinstance(geom, Polygon):
                    x, y = geom.exterior.xy
                    ax.fill(x, y, alpha=0.5, label=label)

        ax.axis('equal')
        ax.grid(True)
        ax.legend()
        plt.savefig(filepath)
        plt.close()

    def __call__(self, results):
        angle = np.random.uniform(*self.rot_range)
        angle = 90/180*np.pi
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        # 可视化旋转前的几何
        if self.vis_dir is not None:
            sample_token = results.get('sample_token', 'unknown')
            before_path = os.path.join(self.vis_dir, f'{sample_token}_before_rot.png')
            self.plot_geoms(results["map_geoms"], before_path)

        # 执行旋转
        map_geoms_rotate = {}
        for label, geom_list in results["map_geoms"].items():
            map_geoms_rotate[label] = []
            for geom in geom_list:
                geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
                map_geoms_rotate[label].append(geom_rotate)
        results["map_geoms"] = map_geoms_rotate

        # 可视化旋转后的几何
        if self.vis_dir is not None:
            after_path = os.path.join(self.vis_dir, f'{sample_token}_after_rot_{np.rad2deg(angle):.1f}deg.png')
            self.plot_geoms(results["map_geoms"], after_path)
        return results

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mmcv.utils import mkdir_or_exist

@PIPELINES.register_module()
class BBoxRotationVis(object):
    def __init__(self, vis_dir=None):
        self.vis_dir = "vis/"
        if vis_dir is not None:
            mkdir_or_exist(vis_dir)

    def plot_boxes(self, boxes, filepath, title=""):
        """绘制3D边界框的俯视图(xy平面)"""
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        for box in boxes:
            # 提取中心坐标、长宽和朝向角
            x, y, z = box[:3]
            length, width = box[3], box[4]
            yaw = box[6]

            # 计算四个角点
            half_l, half_w = length/2, width/2
            corners = np.array([
                [-half_l, -half_w],
                [ half_l, -half_w],
                [ half_l,  half_w],
                [-half_l,  half_w]
            ])

            # 旋转角点
            rot_mat = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])
            rotated_corners = corners @ rot_mat.T + np.array([x, y])

            # 绘制边界框
            rect = plt.Polygon(rotated_corners, closed=True,
                              fill=False, linewidth=2, edgecolor='r')
            ax.add_patch(rect)

            # 绘制朝向箭头
            front = rotated_corners[1] - rotated_corners[0]
            front = front / np.linalg.norm(front)
            ax.arrow(x, y, front[0]*2, front[1]*2,
                    head_width=0.5, head_length=0.7, fc='b', ec='b')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title(title)
        ax.grid(True)
        ax.axis('equal')

        # 设置合理的坐标范围
        if len(boxes) > 0:
            centers = boxes[:, :2]
            max_extent = np.max(boxes[:, [3,4]]) * 1.5
            x_min, y_min = np.min(centers, axis=0) - max_extent
            x_max, y_max = np.max(centers, axis=0) + max_extent
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        plt.savefig(filepath)
        plt.close()

    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        angle = 90/180*np.pi
        # 可视化旋转前的边界框
        if self.vis_dir is not None and "gt_bboxes_3d" in results:
            sample_token = results.get('sample_token', 'unknown')
            before_path = os.path.join(self.vis_dir, f'{sample_token}_bbox_before_rot.png')
            self.plot_boxes(results["gt_bboxes_3d"], before_path, 
                          f"Before Rotation (Angle: {np.rad2deg(angle):.1f}°)")

        # 执行旋转变换
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        # 变换相机参数
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = results["lidar2img"][view] @ rot_mat_inv

        # 变换全局坐标
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv

        # 变换3D边界框
        if "gt_bboxes_3d" in results:
            original_boxes = results["gt_bboxes_3d"].copy()
            results["gt_bboxes_3d"] = self.box_rotate(results["gt_bboxes_3d"], angle)

            # 可视化旋转后的边界框
            if self.vis_dir is not None:
                after_path = os.path.join(self.vis_dir, f'{sample_token}_bbox_after_rot.png')
                self.plot_boxes(results["gt_bboxes_3d"], after_path,
                              f"After Rotation (Angle: {np.rad2deg(angle):.1f}°)")

                # 可选：在同一图中绘制前后对比
                compare_path = os.path.join(self.vis_dir, f'{sample_token}_bbox_compare.png')
                plt.figure(figsize=(10, 10))
                ax = plt.gca()

                # 绘制原始框(红色)
                for box in original_boxes:
                    self._draw_single_box(ax, box, 'r', 'Original')

                # 绘制旋转后框(蓝色)
                for box in results["gt_bboxes_3d"]:
                    self._draw_single_box(ax, box, 'b', 'Rotated')

                ax.legend()
                ax.set_title(f"Rotation Comparison ({np.rad2deg(angle):.1f}°)")
                ax.grid(True)
                ax.axis('equal')
                plt.savefig(compare_path)
                plt.close()
        import ipdb; ipdb.set_trace()
        return results

    def _draw_single_box(self, ax, box, color, label):
        """辅助函数：绘制单个边界框"""
        x, y = box[:2]
        length, width = box[3], box[4]
        yaw = box[6]

        half_l, half_w = length/2, width/2
        corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ])

        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        rotated_corners = corners @ rot_mat.T + np.array([x, y])

        rect = plt.Polygon(rotated_corners, closed=True,
                          fill=False, linewidth=2, edgecolor=color, label=label)
        ax.add_patch(rect)

        front = rotated_corners[1] - rotated_corners[0]
        front = front / np.linalg.norm(front)
        ax.arrow(x, y, front[0]*2, front[1]*2,
                head_width=0.5, head_length=0.7, fc=color, ec=color)

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str



@PIPELINES.register_module()
class BBoxMapPathRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = results["lidar2img"][view] @ rot_mat_inv

        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv

        # Save original boxes for visualization
        gt_boxes_orig = None
        if "gt_bboxes_3d" in results:
            gt_boxes_orig = np.copy(results["gt_bboxes_3d"])
            results["gt_bboxes_3d"] = self.box_rotate(results["gt_bboxes_3d"], angle)

        # Save original map for visualization
        map_geoms_orig = None
        if "map_geoms" in results:
            map_geoms_orig = results["map_geoms"].copy()
            map_geoms_rotate = {}
            for label, geom_list in results["map_geoms"].items():
                map_geoms_rotate[label] = []
                for geom in geom_list:
                    geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
                    map_geoms_rotate[label].append(geom_rotate)
            results["map_geoms"] = map_geoms_rotate

        # ====== Trajectory Rotation ======
        # Save original trajectories for visualization
        gt_agent_fut_trajs_orig = None
        gt_ego_fut_trajs_orig = None

        if "gt_agent_fut_trajs" in results:
            rot_mat_T_2d = rot_mat[:2, :2].T

            num_box = gt_boxes_orig.shape[0]
            gt_agent_fut_trajs_orig = np.concatenate((np.zeros((num_box, 1, 2)), results["gt_agent_fut_trajs"]), axis=1)
            gt_agent_fut_trajs_orig = gt_agent_fut_trajs_orig.cumsum(axis=-2) + gt_boxes_orig[:, None, :2] ## num_box, timestep, 2
            gt_agent_fut_masks = results["gt_agent_fut_masks"]

            gt_ego_fut_trajs_orig = np.concatenate((np.zeros((1, 2)), results["gt_spatial"]), axis=0)
            gt_ego_fut_masks = results["gt_spatial_mask"]
            tp_orig = results["tp_near"]

            results["gt_agent_fut_trajs"] = (results["gt_agent_fut_trajs"] @ rot_mat_T_2d).astype(np.float32)
            results["gt_spatial"] = (results["gt_spatial"] @ rot_mat_T_2d).astype(np.float32)

            gt_agent_fut_trajs_rot = np.concatenate((np.zeros((num_box, 1, 2)), results["gt_agent_fut_trajs"]), axis=1)
            gt_agent_fut_trajs_rot = gt_agent_fut_trajs_rot.cumsum(axis=-2) + results["gt_bboxes_3d"][:, None, :2] ## num_box, timestep, 2
            gt_ego_fut_trajs_rot = np.concatenate((np.zeros((1, 2)), results["gt_spatial"]), axis=0)

            results["gt_trajs_orig"] = {
                "agent": gt_agent_fut_trajs_orig,
                "ego": gt_ego_fut_trajs_orig,
                "tp": tp_orig,
            }
            results["gt_trajs_rot"] = {
                "agent": gt_agent_fut_trajs_rot,
                "ego": gt_ego_fut_trajs_rot,
                "tp": results["tp_near"],
            }

        # ====== Visualization ======
        # self.visualize_rotation(
        #     results,
        #     gt_boxes_orig,
        #     map_geoms_orig,
        #     gt_agent_fut_trajs_orig,
        #     gt_ego_fut_trajs_orig,
        #     tp_orig,
        # )
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d

    def visualize_rotation(self, results, gt_boxes_orig, map_geoms_orig,
                           gt_agent_trajs_orig, gt_ego_trajs_orig, tp_orig):
        """Visualize ground truths before and after rotation"""
        try:
            if "gt_trajs_rot" not in results:
                return

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle(f"Rotation: {results['aug_config']['rotate_3d']:.2f} rad", fontsize=16)

            # ===== Before Rotation =====
            ax1.set_title("Before Rotation")
            self._plot_scene(
                ax1,
                map_geoms_orig,
                gt_boxes_orig,
                gt_agent_trajs_orig,
                gt_ego_trajs_orig,
                tp_orig,
            )

            # ===== After Rotation =====
            ax2.set_title("After Rotation")
            self._plot_scene(
                ax2,
                results.get("map_geoms", None),
                results.get("gt_bboxes_3d", None),
                results["gt_trajs_rot"]["agent"],
                results["gt_trajs_rot"]["ego"],
                results["gt_trajs_rot"]["tp"],
            )

            # Save figure
            os.makedirs("viz_rotations", exist_ok=True)
            timestamp = int(time.time() * 1000)
            plt.savefig(f"viz_rotations/{timestamp}.png")
            plt.close(fig)
        except Exception as e:
            print(f"Visualization failed: {str(e)}")

    def _plot_scene(self, ax, map_geoms, gt_boxes, agent_trajs, ego_trajs, tp):
        """Plot single scene on given axis"""
        # Plot map geometries
        if map_geoms is not None:
            for label, geom_list in map_geoms.items():
                for geom in geom_list:
                    if geom.geom_type == "LineString":
                        x, y = geom.xy
                        ax.plot(x, y, 'b-', linewidth=1, alpha=0.5)
                    elif geom.geom_type == "Polygon":
                        x, y = geom.exterior.xy
                        ax.plot(x, y, 'g-', linewidth=1, alpha=0.3)

        # Plot agent trajectories
        if agent_trajs is not None:
            for i, traj in enumerate(agent_trajs):
                ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1.5, alpha=0.7)
                ax.scatter(traj[0, 0], traj[0, 1], c='r', marker='o', s=30)

        # Plot ego trajectory
        if ego_trajs is not None:
            ax.plot(ego_trajs[:, 0], ego_trajs[:, 1], 'g-', linewidth=2.5)
            ax.scatter(ego_trajs[0, 0], ego_trajs[0, 1], c='g', marker='s', s=80)
            ax.scatter(ego_trajs[-1, 0], ego_trajs[-1, 1], c='purple', marker='*', s=100)

        # Plot bounding boxes
        if gt_boxes is not None and len(gt_boxes) > 0:
            for box in gt_boxes:
                center_x, center_y = box[:2]
                length, width = box[3], box[4]
                angle = box[6]

                # Calculate corner points
                dx = length / 2
                dy = width / 2
                corners = np.array([
                    [-dx, -dy],
                    [-dx, dy],
                    [dx, dy],
                    [dx, -dy],
                    [-dx, -dy]
                ])

                # Rotation matrix
                rot_mat = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])

                # Rotate and translate corners
                rotated_corners = corners @ rot_mat.T + np.array([center_x, center_y])

                # Plot bounding box
                ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'm-', linewidth=1.5)

        # plot tp
        ax.scatter(tp[0], tp[1], c='b', marker='*', s=500)

        # Set plot properties
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.axis("equal")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

@PIPELINES.register_module()
class BBoxMapPathTargetPointRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = results["lidar2img"][view] @ rot_mat_inv

        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv

        # Save original boxes for visualization
        gt_boxes_orig = None
        if "gt_bboxes_3d" in results:
            gt_boxes_orig = np.copy(results["gt_bboxes_3d"])
            results["gt_bboxes_3d"] = self.box_rotate(results["gt_bboxes_3d"], angle)

        # Save original map for visualization
        map_geoms_orig = None
        if "map_geoms" in results:
            map_geoms_orig = results["map_geoms"].copy()
            map_geoms_rotate = {}
            for label, geom_list in results["map_geoms"].items():
                map_geoms_rotate[label] = []
                for geom in geom_list:
                    geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
                    map_geoms_rotate[label].append(geom_rotate)
            results["map_geoms"] = map_geoms_rotate

        # ====== Trajectory Rotation ======
        # Save original trajectories for visualization
        gt_agent_fut_trajs_orig = None
        gt_ego_fut_trajs_orig = None

        if "gt_agent_fut_trajs" in results:
            rot_mat_T_2d = rot_mat[:2, :2].T

            num_box = gt_boxes_orig.shape[0]
            gt_agent_fut_trajs_orig = np.concatenate((np.zeros((num_box, 1, 2)), results["gt_agent_fut_trajs"]), axis=1)
            gt_agent_fut_trajs_orig = gt_agent_fut_trajs_orig.cumsum(axis=-2) + gt_boxes_orig[:, None, :2] ## num_box, timestep, 2
            gt_agent_fut_masks = results["gt_agent_fut_masks"]

            gt_ego_fut_trajs_orig = np.concatenate((np.zeros((1, 2)), results["gt_spatial"]), axis=0)
            gt_ego_fut_masks = results["gt_spatial_mask"]
            tp_orig = results["tp_near"]

            results["gt_agent_fut_trajs"] = (results["gt_agent_fut_trajs"] @ rot_mat_T_2d).astype(np.float32)
            results["gt_spatial"] = (results["gt_spatial"] @ rot_mat_T_2d).astype(np.float32)
            results["tp_near"] = (results["tp_near"] @ rot_mat_T_2d).astype(np.float32)

            gt_agent_fut_trajs_rot = np.concatenate((np.zeros((num_box, 1, 2)), results["gt_agent_fut_trajs"]), axis=1)
            gt_agent_fut_trajs_rot = gt_agent_fut_trajs_rot.cumsum(axis=-2) + results["gt_bboxes_3d"][:, None, :2] ## num_box, timestep, 2
            gt_ego_fut_trajs_rot = np.concatenate((np.zeros((1, 2)), results["gt_spatial"]), axis=0)

            results["gt_trajs_orig"] = {
                "agent": gt_agent_fut_trajs_orig,
                "ego": gt_ego_fut_trajs_orig,
                "tp": tp_orig,
            }
            results["gt_trajs_rot"] = {
                "agent": gt_agent_fut_trajs_rot,
                "ego": gt_ego_fut_trajs_rot,
                "tp": results["tp_near"],
            }

        # ====== Visualization ======
        # self.visualize_rotation(
        #     results,
        #     gt_boxes_orig,
        #     map_geoms_orig,
        #     gt_agent_fut_trajs_orig,
        #     gt_ego_fut_trajs_orig,
        #     tp_orig,
        # )
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d

    def visualize_rotation(self, results, gt_boxes_orig, map_geoms_orig,
                           gt_agent_trajs_orig, gt_ego_trajs_orig, tp_orig):
        """Visualize ground truths before and after rotation"""
        try:
            if "gt_trajs_rot" not in results:
                return

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle(f"Rotation: {results['aug_config']['rotate_3d']:.2f} rad", fontsize=16)

            # ===== Before Rotation =====
            ax1.set_title("Before Rotation")
            self._plot_scene(
                ax1,
                map_geoms_orig,
                gt_boxes_orig,
                gt_agent_trajs_orig,
                gt_ego_trajs_orig,
                tp_orig,
            )

            # ===== After Rotation =====
            ax2.set_title("After Rotation")
            self._plot_scene(
                ax2,
                results.get("map_geoms", None),
                results.get("gt_bboxes_3d", None),
                results["gt_trajs_rot"]["agent"],
                results["gt_trajs_rot"]["ego"],
                results["gt_trajs_rot"]["tp"],
            )

            # Save figure
            os.makedirs("viz_rotations", exist_ok=True)
            timestamp = int(time.time() * 1000)
            plt.savefig(f"viz_rotations/{timestamp}.png")
            plt.close(fig)
        except Exception as e:
            print(f"Visualization failed: {str(e)}")

    def _plot_scene(self, ax, map_geoms, gt_boxes, agent_trajs, ego_trajs, tp):
        """Plot single scene on given axis"""
        # Plot map geometries
        if map_geoms is not None:
            for label, geom_list in map_geoms.items():
                for geom in geom_list:
                    if geom.geom_type == "LineString":
                        x, y = geom.xy
                        ax.plot(x, y, 'b-', linewidth=1, alpha=0.5)
                    elif geom.geom_type == "Polygon":
                        x, y = geom.exterior.xy
                        ax.plot(x, y, 'g-', linewidth=1, alpha=0.3)

        # Plot agent trajectories
        if agent_trajs is not None:
            for i, traj in enumerate(agent_trajs):
                ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1.5, alpha=0.7)
                ax.scatter(traj[0, 0], traj[0, 1], c='r', marker='o', s=30)

        # Plot ego trajectory
        if ego_trajs is not None:
            ax.plot(ego_trajs[:, 0], ego_trajs[:, 1], 'g-', linewidth=2.5)
            ax.scatter(ego_trajs[0, 0], ego_trajs[0, 1], c='g', marker='s', s=80)
            ax.scatter(ego_trajs[-1, 0], ego_trajs[-1, 1], c='purple', marker='*', s=100)

        # Plot bounding boxes
        if gt_boxes is not None and len(gt_boxes) > 0:
            for box in gt_boxes:
                center_x, center_y = box[:2]
                length, width = box[3], box[4]
                angle = box[6]

                # Calculate corner points
                dx = length / 2
                dy = width / 2
                corners = np.array([
                    [-dx, -dy],
                    [-dx, dy],
                    [dx, dy],
                    [dx, -dy],
                    [-dx, -dy]
                ])

                # Rotation matrix
                rot_mat = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])

                # Rotate and translate corners
                rotated_corners = corners @ rot_mat.T + np.array([center_x, center_y])

                # Plot bounding box
                ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'm-', linewidth=1.5)

        # plot tp
        ax.scatter(tp[0], tp[1], c='b', marker='*', s=500)

        # Set plot properties
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.axis("equal")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
