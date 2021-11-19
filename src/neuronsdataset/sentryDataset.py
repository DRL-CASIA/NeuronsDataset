import os
import cv2
from torchvision.datasets import VisionDataset
import numpy as np
# from EX_CONST import Const

intrinsic_camera_matrix_filenames = ["intri_left.xml", "intri_right.xml"]
extrinsic_camera_matrix_filenames = ["extri_left.xml", "extri_right.xml"]

class sentryDataset(VisionDataset):
    def __init__(self, root, worldgrid_shape = [449, 800]):
        super().__init__(root)
        self.__name__ = 'sentry'
        self.img_shape = [480, 640]
        self.worldgrid_shape = worldgrid_shape
        self.bins = 2
        self.overlap_ratio = 0.1
        self.num_cam = 2
        self.worldgrid2worldcoord_mat = np.array([[1,0,0], [0,1,0], [0,0,1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
        camera_folder = os.listdir(os.path.join(self.root, 'img'))[0]
        path = os.path.join(self.root, 'img', camera_folder)
        self.num_frame = len(os.listdir(path))

        i = 0
        self.all_indexs = []
        while i < self.num_frame:
            file = os.listdir(path)[i]
            self.all_indexs.append(int(file.split("/")[-1].split(".")[0]))
            i += 1

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'img'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'img', camera_folder))):
                frame = int(fname.split('.')[0][-4:])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'img', camera_folder, fname)
        return img_fpaths

    def get_anno_fpaths(self, frame_range):
        anno_fpaths = {}
        anno_folder = os.path.join(self.root, "annotations")
        for fname in sorted(os.listdir(anno_folder)):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                anno_fpaths[frame] = os.path.join(anno_folder, fname)
        return anno_fpaths

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibration', 'intrinsic')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                             flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('intri_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_camera_path = os.path.join(self.root, 'calibration', 'extrinsic')
        extrinsic_params_file = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                             extrinsic_camera_matrix_filenames[camera_i]),
                                                             flags=cv2.FILE_STORAGE_READ)
        extrinsic_matrix = extrinsic_params_file.getNode('extri_matrix').mat()
        extrinsic_params_file.release()
        for i in range(3):
            extrinsic_matrix[i,3] /= 10
        return intrinsic_matrix, extrinsic_matrix

    def get_worldgrid_from_worldcoord(self, world_coord):
        coord_x, coord_y = world_coord
        grid_x = coord_x
        grid_y = coord_y
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        coord_x = grid_x
        coord_y = grid_y
        return np.array([coord_x, coord_y])

    def get_visualize_info(self, idx: int):
        imgs = self.get_image_fpaths([idx])
        annos = self.get_anno_fpaths([idx])

        return imgs, annos, self.extrinsic_matrices, self.intrinsic_matrices

