import json
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor
from random import shuffle
import numpy as np
import os
import cv2

class FrameDataset(VisionDataset):
    def __init__(self, base, grid_reduce=4, img_reduce=4, train_ratio = 0.9, train = True):
        super().__init__(base.root, transform=ToTensor(), target_transform=ToTensor())
        self.grid_reduce = grid_reduce
        self.img_reduce = img_reduce
        self.base = base
        self.root, self.num_frame = base.root, base.num_frame
        self.all_idx = base.all_indexs
        shuffle(self.all_idx)
        self.all_idx = np.array(self.all_idx)

        if train:
            frame_range = self.all_idx[list(range(0, int(self.num_frame * train_ratio)))]
        else:
            frame_range = self.all_idx[list(range(int(self.num_frame * train_ratio), self.num_frame))]

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.anno_fpaths = self.base.get_anno_fpaths(frame_range)

        if self.base.__name__ == "sentry":
            self.num_cam = base.num_cam
            self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
            self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))
            self.extrinsic_matrix = base.extrinsic_matrices
            self.intrinsic_matrix = base.intrinsic_matrices
            # self.img_fpaths = self.base.get_image_fpaths(frame_range)
            self.upsample_shape = list(map(lambda x: int(x / self.img_reduce), self.img_shape))
            img_reduce_local = np.array(self.img_shape) / np.array(self.upsample_shape)
            imgcoord2worldgrid_matrices = get_imgcoord2worldgrid_matrices(base.intrinsic_matrices,
                                                                          base.extrinsic_matrices,
                                                                          base.worldgrid2worldcoord_mat)
            img_zoom_mat = np.diag(np.append(img_reduce_local, [1]))
            map_zoom_mat = np.diag(np.append(np.ones([2]) / self.grid_reduce, [1]))

            self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                              for cam in range(2)]

            # create angle bins
            bins = base.bins
            overlap = base.overlap_ratio
            self.bins = bins
            self.angle_bins = np.zeros(bins)
            self.interval = 2 * np.pi / bins
            for i in range(1, bins):
                self.angle_bins[i] = i * self.interval
            self.angle_bins += self.interval / 2  # center of the bin

            self.overlap = overlap
            # ranges for confidence
            # [(min angle in bin, max angle in bin), ... ]
            self.bin_ranges = []
            for i in range(0, bins):
                self.bin_ranges.append(((i * self.interval - overlap) % (2 * np.pi), \
                                        (i * self.interval + self.interval + overlap) % (2 * np.pi)))

            self.bev_bboxes = {}
            self.left_bboxes = {}
            self.right_bboxes = {}
            self.left_dir = {}
            self.right_dir = {}
            self.left_angle = {}
            self.right_angle = {}
            self.left_orientation = {}
            self.left_conf = {}
            self.right_orientation = {}
            self.right_conf = {}
            self.world_xy = {}
            self.bev_angle = {}
            self.mark = {}

            self.sentry_prepare_bbox(frame_range)
            self.sentry_prepare_dir(frame_range)
            self.sentry_prepare_bins(frame_range)
        if self.base.__name__ == "rgbd":
            self.depth = {}
            self.rgbd_prepare_depth(frame_range)
        if self.base.__name__ == "rgb":
            self.boxes = {}
            self.classes = {}
            self.rgb_prepare_bbox(frame_range)

    def sentry_prepare_dir(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame_left_dir = []
            frame_right_dir = []
            frame_left_ang = []
            frame_right_ang = []
            frame_wxy = []
            frame_bev_angle = []
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    cars = [json.load(json_file)][0]
                for i, car in enumerate(cars):
                    wx = int(car["wx"]) // 10
                    wy = int(car["wy"]) // 10
                    mk = int(car["mark"])
                    bev_angle = float(car["angle"])

                    frame_wxy.append([wx, wy])

                    frame_left_dir.append(0)
                    frame_right_dir.append(0)

                    # 0~360
                    if bev_angle < 0:
                        bev_angle += 2 * np.pi
                    # 左角度标签
                    alpha = np.arctan((self.base.worldgrid_shape[0] - wy) / wx)
                    left_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)
                    # if frame in range(500, 600) and i == 2:
                        # print(wx, wy)
                        # print(np.rad2deg(bev_angle))
                        # print(np.rad2deg(alpha))
                        # print(np.rad2deg(left_target))
                        # print(np.arctan(np.sin(left_target) / np.cos(left_target)))
                    frame_left_ang.append([np.sin(left_target), np.cos(left_target)]) # 方案1, 回归sin cos

                    # 右角度标签, 颠倒一下正方向
                    bev_angle -= np.pi
                    if bev_angle < 0:
                        bev_angle += 2 * np.pi
                    frame_bev_angle.append(bev_angle)
                    alpha = np.arctan(wy / (self.base.worldgrid_shape[1] - wx))
                    right_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)
                    frame_right_ang.append([np.sin(right_target), np.cos(right_target)]) # 方案1, 回归sin cos

                self.world_xy[frame] = frame_wxy
                self.left_dir[frame] = frame_left_dir
                self.right_dir[frame] = frame_right_dir
                self.bev_angle[frame] = frame_bev_angle
                self.left_angle[frame] = frame_left_ang
                self.right_angle[frame] = frame_right_ang
                self.mark[frame] = mk

    def sentry_prepare_bbox(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame_bev_box = []
            frame_left_box = []
            frame_right_box = []
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    cars = [json.load(json_file)][0]
                for i, car in enumerate(cars):
                    ymin_od = int(car["ymin_od"])
                    xmin_od = int(car["xmin_od"])
                    ymax_od = int(car["ymax_od"])
                    xmax_od = int(car["xmax_od"])
                    frame_bev_box.append([ymin_od, xmin_od, ymax_od, xmax_od])

                    for j in range(self.num_cam):
                        ymin = car["views"][j]["ymin"]
                        xmin = car["views"][j]["xmin"]
                        ymax = car["views"][j]["ymax"]
                        xmax = car["views"][j]["xmax"]
                        if j == 0:
                            frame_left_box.append([ymin, xmin, ymax, xmax])
                        else:
                            frame_right_box.append([ymin, xmin, ymax, xmax])


                self.bev_bboxes[frame] = frame_bev_box
                self.left_bboxes[frame] = frame_left_box
                self.right_bboxes[frame] = frame_right_box

    def sentry_prepare_bins(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame_left_dir = []
            frame_right_dir = []
            frame_left_ang = []
            frame_right_ang = []
            frame_wxy = []
            frame_bev_angle = []
            frame_left_orientation = []
            frame_left_conf = []
            frame_right_orientation = []
            frame_right_conf = []

            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    cars = [json.load(json_file)][0]
                for i, car in enumerate(cars):
                    wx = int(car["wx"]) // 10
                    wy = int(car["wy"]) // 10
                    mk = int(car["mark"])
                    left_dir = 0
                    right_dir = 0
                    bev_angle = float(car["angle"])

                    frame_wxy.append([wx, wy])
                    frame_left_dir.append(0)
                    frame_right_dir.append(0)

                    # 0~360
                    if bev_angle < 0:
                        bev_angle += 2 * np.pi
                    # 左角度标签
                    alpha = np.arctan((self.base.worldgrid_shape[0] - wy) / wx)
                    left_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)
                    left_orientation = np.zeros((self.bins, 2))
                    left_confidence = np.zeros(self.bins)
                    left_bin_idxs = self.sentry_get_bin(left_target)
                    for bin_idx in left_bin_idxs:
                        angle_diff = left_target - self.angle_bins[bin_idx]
                        left_orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
                        left_confidence[bin_idx] = 1
                    frame_left_orientation.append(left_orientation)
                    frame_left_conf.append(left_confidence)


                    # 右角度标签, 颠倒一下正方向
                    bev_angle -= np.pi
                    if bev_angle < 0:
                        bev_angle += 2 * np.pi
                    frame_bev_angle.append(bev_angle)
                    alpha = np.arctan(wy / (self.base.worldgrid_shape[1] - wx))
                    right_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)

                    right_orientation = np.zeros((self.bins, 2))
                    right_confidence = np.zeros(self.bins)
                    right_bin_idxs = self.sentry_get_bin(right_target)
                    for bin_idx in right_bin_idxs:
                        angle_diff = right_target - self.angle_bins[bin_idx]
                        right_orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
                        right_confidence[bin_idx] = 1
                    frame_right_orientation.append(right_orientation)
                    frame_right_conf.append(right_confidence)


                self.left_orientation[frame] = frame_left_orientation
                self.left_conf[frame] = frame_left_conf
                self.right_orientation[frame] = frame_right_orientation
                self.right_conf[frame] = frame_right_conf

    def sentry_get_bin(self, angle):
        bin_idxs = []
        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def rgb_prepare_bbox(self, frame_range):
        for key in frame_range:
            frame_data = np.loadtxt(self.anno_fpaths[key]).reshape(-1, 5)
            # print(frame_data.shape)
            classes = frame_data[:, 0].reshape(-1)
            boxes = np.concatenate(([frame_data[:, 1].reshape(-1, 1) * 640, frame_data[:, 2].reshape(-1, 1) * 480,
                                     frame_data[:, 3].reshape(-1, 1) * 640, frame_data[:, 4].reshape(-1, 1) * 480]), axis=1)
            self.boxes[key] = boxes
            self.classes[key] = classes

    def rgbd_prepare_depth(self, frame_range):
        for key in frame_range:
            dpt = cv2.imread(self.anno_fpaths[key], cv2.IMREAD_ANYDEPTH)
            self.depth[key] = dpt
            # print(self.depth[key])

    def __getitem__(self, item):
        if self.base.__name__ == "sentry":
            frame = list(self.bev_bboxes.keys())[item]
            imgs = []
            for cam in range(self.num_cam):
                fpath = self.img_fpaths[cam][frame]
                img = Image.open(fpath).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)
            imgs = torch.stack(imgs)
            bev_bboxes = torch.tensor(self.bev_bboxes[frame])
            left_bboxes = torch.tensor(self.left_bboxes[frame])
            right_bboxes = torch.tensor(self.right_bboxes[frame])
            left_dirs = torch.tensor(self.left_dir[frame])
            right_dirs = torch.tensor(self.right_dir[frame])
            left_angles = torch.tensor(self.left_angle[frame])
            right_angles = torch.tensor(self.right_angle[frame])
            bev_xy = torch.tensor(self.world_xy[frame])
            bev_angle = torch.tensor(self.bev_angle[frame])
            mark = self.mark[frame]

            left_orientation = torch.tensor(self.left_orientation[frame])
            left_conf = torch.tensor(self.left_conf[frame])
            right_orientation = torch.tensor(self.right_orientation[frame])
            right_conf = torch.tensor(self.right_conf[frame])

            return imgs, bev_xy, bev_angle, bev_bboxes, \
                   left_bboxes, right_bboxes, \
                   left_dirs, right_dirs, \
                   left_angles, right_angles, \
                   left_orientation, right_orientation, \
                   left_conf, right_conf, \
                   frame, \
                   self.extrinsic_matrix, self.intrinsic_matrix, \
                   mark

        if self.base.__name__ == "rgb":
            frame = list(self.boxes.keys())[item]
            fpath = self.img_fpaths[frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            boxes = torch.tensor(self.boxes[frame])
            classes = torch.tensor(self.boxes[frame])

            return img, boxes, classes

        if self.base.__name__ == "rgbd":
            frame = list(self.depth.keys())[item]
            fpath = self.img_fpaths[frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            depth = torch.tensor(self.depth[frame] / 1000.)
            return img, depth


    def __len__(self):
        if self.base.__name__ == "sentry":
            return len(self.bev_bboxes.keys())
        if self.base.__name__ == "rgb":
            return len(self.boxes.keys())
        if self.base.__name__ == "rgbd":
            return len(self.depth.keys())

def get_imgcoord2worldgrid_matrices(intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
    projection_matrices = {}
    for cam in range(2):
        worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
        worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
        imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
        permutation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
    return projection_matrices

