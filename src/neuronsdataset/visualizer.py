import json
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .sentryDataset import sentryDataset
from .rgbDataset import rgbDataset
from .rgbdDataset import rgbdDataset
import os

class Visualizer():
    def __init__(self, dataroot, dataset_name="rgb"):
        self.root = dataroot
        self.dataset_name = dataset_name
        assert self.dataset_name in ["sentry", "rgb", "rgbd"], "Invalid dataset name!"
        if self.dataset_name == "sentry":
            self.dataset = sentryDataset(os.path.join(dataroot, self.dataset_name))

        elif self.dataset_name == "rgb":
            self.dataset = rgbDataset(os.path.join(dataroot, self.dataset_name))

        elif self.dataset_name == "rgbd":
            self.dataset = rgbdDataset(os.path.join(dataroot, self.dataset_name))

    def visualize(self, idx: int):
        imgs, anno, extrinsics, intrinsics = self.dataset.get_visualize_info(idx)
        colors = [(255, 255, 0), (136, 11, 235), (90, 200, 2), (255, 255, 0), (126, 56, 190)]
        if self.dataset_name == "sentry":
            annopath = anno[idx]
            frame_bev_box, frame_left_box, frame_right_box = self.prepare_bbox(annopath)
            frame_type, frame_wxy, frame_left_dir, frame_right_dir, frame_bev_angle, frame_left_ang, frame_right_ang = self.prepare_dir(
                annopath)
            left_front_image = cv2.imread(imgs[0][idx])
            right_front_image = cv2.imread(imgs[1][idx])
            left_zeros = np.zeros((left_front_image.shape), dtype=np.uint8)
            right_zeros = np.zeros((left_front_image.shape), dtype=np.uint8)
            boxes_3d = []

            num_cars = len(frame_wxy)
            for i in range(num_cars):
                ymin, xmin, ymax, xmax = frame_left_box[i]
                points = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
                if ymin != -1:
                    cv2.rectangle(left_front_image, (xmin, ymin - 20), (xmin + 30, ymin), color=colors[i], thickness=-1)
                    cv2.putText(left_front_image, str(frame_type[i]), (xmin + 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                color=(255, 255, 255), fontScale=0.8, thickness=2)
                    cv2.rectangle(left_front_image, (xmin, ymin), (xmax, ymax), color=colors[i], thickness=2)
                    mask = cv2.fillPoly(left_zeros, np.array([points]), color=colors[i])
                    left_front_image = 0.2 * mask + left_front_image

                ymin, xmin, ymax, xmax = frame_right_box[i]
                points = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
                if ymin != -1:
                    cv2.rectangle(right_front_image, (xmin, ymin - 20), (xmin + 30, ymin), color=colors[i], thickness=-1)
                    cv2.putText(right_front_image, str(frame_type[i]), (xmin + 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                color=(255, 255, 255), fontScale=0.8, thickness=2)
                    cv2.rectangle(right_front_image, (xmin, ymin), (xmax, ymax), color=colors[i], thickness=2)
                    mask = cv2.fillPoly(right_zeros, np.array([points]), color=colors[i])
                    right_front_image = 0.2 * mask + right_front_image

            left_image = cv2.imread(imgs[0][idx])
            right_image = cv2.imread(imgs[1][idx])

            for i in range(num_cars):
                ymin, xmin, ymax, xmax = frame_bev_box[i]
                theta = frame_bev_angle[i]
                # theta =torch.tensor(0)
                center_x, center_y = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
                w = 60
                h = 50
                xmin = center_x - w // 2
                xmax = center_x + w // 2
                ymin = center_y - h // 2
                ymax = center_y + h // 2
                x1_ori, x2_ori, x3_ori, x4_ori, x_mid = center_x - w // 2, xmin, xmax, xmax, (xmin + xmax) / 2 - 40
                y1_ori, y2_ori, y3_ori, y4_ori, y_mid = self.dataset.worldgrid_shape[0] - ymin, \
                                                        self.dataset.worldgrid_shape[0] - ymax, \
                                                        self.dataset.worldgrid_shape[0] - ymax, \
                                                        self.dataset.worldgrid_shape[0] - ymin, (
                                                                self.dataset.worldgrid_shape[0] - ymax +
                                                                self.dataset.worldgrid_shape[0] - ymin) / 2

                x1_rot, x2_rot, x3_rot, x4_rot, xmid_rot = \
                    int(math.cos(theta) * (x1_ori - center_x) - math.sin(theta) * (
                            y1_ori - (self.dataset.worldgrid_shape[0] - center_y)) + center_x), \
                    int(math.cos(theta) * (x2_ori - center_x) - math.sin(theta) * (
                            y2_ori - (self.dataset.worldgrid_shape[0] - center_y)) + center_x), \
                    int(math.cos(theta) * (x3_ori - center_x) - math.sin(theta) * (
                            y3_ori - (self.dataset.worldgrid_shape[0] - center_y)) + center_x), \
                    int(math.cos(theta) * (x4_ori - center_x) - math.sin(theta) * (
                            y4_ori - (self.dataset.worldgrid_shape[0] - center_y)) + center_x), \
                    int(math.cos(theta) * (x_mid - center_x) - math.sin(theta) * (
                            y_mid - (self.dataset.worldgrid_shape[0] - center_y)) + center_x)

                y1_rot, y2_rot, y3_rot, y4_rot, ymid_rot = \
                    int(math.sin(theta) * (x1_ori - center_x) + math.cos(theta) * (
                            y1_ori - (self.dataset.worldgrid_shape[0] - center_y)) + (
                                self.dataset.worldgrid_shape[0] - center_y)), \
                    int(math.sin(theta) * (x2_ori - center_x) + math.cos(theta) * (
                            y2_ori - (self.dataset.worldgrid_shape[0] - center_y)) + (
                                self.dataset.worldgrid_shape[0] - center_y)), \
                    int(math.sin(theta) * (x3_ori - center_x) + math.cos(theta) * (
                            y3_ori - (self.dataset.worldgrid_shape[0] - center_y)) + (
                                self.dataset.worldgrid_shape[0] - center_y)), \
                    int(math.sin(theta) * (x4_ori - center_x) + math.cos(theta) * (
                            y4_ori - (self.dataset.worldgrid_shape[0] - center_y)) + (
                                self.dataset.worldgrid_shape[0] - center_y)), \
                    int(math.sin(theta) * (x_mid - center_x) + math.cos(theta) * (
                            y_mid - (self.dataset.worldgrid_shape[0] - center_y)) + (
                                self.dataset.worldgrid_shape[0] - center_y))

                pt0 = [x1_rot, y1_rot, 0]
                pt1 = [x2_rot, y2_rot, 0]
                pt2 = [x3_rot, y3_rot, 0]
                pt3 = [x4_rot, y4_rot, 0]
                pt_h_0 = [x1_rot, y1_rot, 40]
                pt_h_1 = [x2_rot, y2_rot, 40]
                pt_h_2 = [x3_rot, y3_rot, 40]
                pt_h_3 = [x4_rot, y4_rot, 40]
                pt_extra = [xmid_rot, ymid_rot, 0]

                boxes_3d.append([pt0, pt1, pt2, pt3, pt_h_0, pt_h_1, pt_h_2, pt_h_3, pt_extra])

            gt_ori = np.array(boxes_3d).reshape((num_cars, 9, 3))
            left_projected_2d = getprojected_3dbox(gt_ori, extrinsics, intrinsics, isleft=True)
            right_projected_2d = getprojected_3dbox(gt_ori, extrinsics, intrinsics, isleft=False)

            for k in range(num_cars):
                cv2.line(left_image, (left_projected_2d[k][0][0], left_projected_2d[k][0][1]),
                         (left_projected_2d[k][1][0], left_projected_2d[k][1][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][0][0], left_projected_2d[k][0][1]),
                         (left_projected_2d[k][3][0], left_projected_2d[k][3][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][0][0], left_projected_2d[k][0][1]),
                         (left_projected_2d[k][4][0], left_projected_2d[k][4][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][1][0], left_projected_2d[k][1][1]),
                         (left_projected_2d[k][5][0], left_projected_2d[k][5][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][1][0], left_projected_2d[k][1][1]),
                         (left_projected_2d[k][2][0], left_projected_2d[k][2][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][2][0], left_projected_2d[k][2][1]),
                         (left_projected_2d[k][3][0], left_projected_2d[k][3][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][2][0], left_projected_2d[k][2][1]),
                         (left_projected_2d[k][6][0], left_projected_2d[k][6][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][3][0], left_projected_2d[k][3][1]),
                         (left_projected_2d[k][7][0], left_projected_2d[k][7][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][4][0], left_projected_2d[k][4][1]),
                         (left_projected_2d[k][5][0], left_projected_2d[k][5][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][5][0], left_projected_2d[k][5][1]),
                         (left_projected_2d[k][6][0], left_projected_2d[k][6][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][6][0], left_projected_2d[k][6][1]),
                         (left_projected_2d[k][7][0], left_projected_2d[k][7][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][7][0], left_projected_2d[k][7][1]),
                         (left_projected_2d[k][4][0], left_projected_2d[k][4][1]), color=colors[k], thickness=2)
                cv2.line(left_image, (left_projected_2d[k][7][0], left_projected_2d[k][7][1]),
                         (left_projected_2d[k][4][0], left_projected_2d[k][4][1]), color=colors[k], thickness=2)
                cv2.arrowedLine(left_image, (int((left_projected_2d[k][0][0] + left_projected_2d[k][2][0]) / 2),
                                             int((left_projected_2d[k][0][1] + left_projected_2d[k][2][1]) / 2)),
                                (left_projected_2d[k][8][0], left_projected_2d[k][8][1]), color=(255, 60, 199),
                                thickness=2)

                cv2.line(right_image, (right_projected_2d[k][0][0], right_projected_2d[k][0][1]),
                         (right_projected_2d[k][1][0], right_projected_2d[k][1][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][0][0], right_projected_2d[k][0][1]),
                         (right_projected_2d[k][3][0], right_projected_2d[k][3][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][0][0], right_projected_2d[k][0][1]),
                         (right_projected_2d[k][4][0], right_projected_2d[k][4][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][1][0], right_projected_2d[k][1][1]),
                         (right_projected_2d[k][5][0], right_projected_2d[k][5][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][1][0], right_projected_2d[k][1][1]),
                         (right_projected_2d[k][2][0], right_projected_2d[k][2][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][2][0], right_projected_2d[k][2][1]),
                         (right_projected_2d[k][3][0], right_projected_2d[k][3][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][2][0], right_projected_2d[k][2][1]),
                         (right_projected_2d[k][6][0], right_projected_2d[k][6][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][3][0], right_projected_2d[k][3][1]),
                         (right_projected_2d[k][7][0], right_projected_2d[k][7][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][4][0], right_projected_2d[k][4][1]),
                         (right_projected_2d[k][5][0], right_projected_2d[k][5][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][5][0], right_projected_2d[k][5][1]),
                         (right_projected_2d[k][6][0], right_projected_2d[k][6][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][6][0], right_projected_2d[k][6][1]),
                         (right_projected_2d[k][7][0], right_projected_2d[k][7][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][7][0], right_projected_2d[k][7][1]),
                         (right_projected_2d[k][4][0], right_projected_2d[k][4][1]), color=colors[k], thickness=2)
                cv2.line(right_image, (right_projected_2d[k][7][0], right_projected_2d[k][7][1]),
                         (right_projected_2d[k][4][0], right_projected_2d[k][4][1]), color=colors[k], thickness=2)

                cv2.arrowedLine(right_image, (int((right_projected_2d[k][0][0] + right_projected_2d[k][2][0]) / 2),
                                              int((right_projected_2d[k][0][1] + right_projected_2d[k][2][1]) / 2)),
                                (right_projected_2d[k][8][0], right_projected_2d[k][8][1]), color=(255, 60, 199),
                                thickness=2)

            plt.figure(figsize=(16, 12))
            for plt_index in range(1, 5):
                plt.subplot(2, 2, plt_index)
                if plt_index == 1:
                    plt.imshow(left_front_image.astype(np.uint8))
                elif plt_index == 2:
                    plt.imshow(right_front_image.astype(np.uint8))
                elif plt_index == 3:
                    plt.imshow(left_image.astype(np.uint8))
                elif plt_index == 4:
                    plt.imshow(right_image.astype(np.uint8))

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.xticks([])
            plt.yticks([])
            plt.savefig("sentry.jpg")
            plt.show()

        if self.dataset_name == "rgb":
            annopath = anno[idx]
            imgpath = imgs[idx]

            frame_data = np.loadtxt(annopath).reshape(-1, 5)
            classes = frame_data[:, 0].reshape(-1)
            boxes = np.concatenate(([frame_data[:, 1].reshape(-1, 1) * 640, frame_data[:, 2].reshape(-1, 1) * 480,
                                     frame_data[:, 3].reshape(-1, 1) * 640, frame_data[:, 4].reshape(-1, 1) * 480]),
                                   axis=1)
            img = cv2.imread(imgpath)
            zeros = np.zeros((img.shape), dtype=np.uint8)

            for i, box in enumerate(boxes):
                x, y, w, h = box
                ymin, xmin, ymax, xmax = int(y - h / 2), int(x - w / 2), int(y + h / 2), int(x + w / 2)
                points = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
                cv2.rectangle(img, (xmin, ymin - 20), (xmin + 30, ymin), color=colors[int(classes[i])], thickness=-1)
                cv2.putText(img, str(int(classes[i])), (xmin + 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            color=(255, 255, 255), fontScale=0.8, thickness=2)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=colors[int(classes[i])], thickness=2)
                mask = cv2.fillPoly(zeros, np.array([points]), color=colors[int(classes[i])])
                image = 0.2 * mask + 0.8 * img
            b, g, r = cv2.split(image)
            image = cv2.merge([r, g, b])

            plt.imshow(image.astype(np.uint8))
            plt.imsave("rgb.jpg", image.astype(np.uint8))
            plt.xticks([])
            plt.yticks([])
            plt.show()

        if self.dataset_name == "rgbd":
            annopath = anno[idx]
            imgpath = imgs[idx]

            img = cv2.imread(imgpath)
            dpt_img = cv2.imread(annopath, -1)

            plt.figure(figsize=(20, 10))
            for plt_index in range(1,3):
                plt.subplot(1, 2, plt_index)
                if plt_index == 1:
                    plt.imshow(img)
                else:
                    plt.imshow(dpt_img)

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.xticks([])
            plt.yticks([])
            plt.savefig("rgbd.jpg")
            plt.show()

    def prepare_bbox(self, annopath):
        frame_bev_box = []
        frame_left_box = []
        frame_right_box = []
        with open(annopath) as json_file:
            cars = [json.load(json_file)][0]

        for i, car in enumerate(cars):
            ymin_od = int(car["ymin_od"])
            xmin_od = int(car["xmin_od"])
            ymax_od = int(car["ymax_od"])
            xmax_od = int(car["xmax_od"])
            frame_bev_box.append([ymin_od, xmin_od, ymax_od, xmax_od])

            car_percam_box = []
            for j in range(self.dataset.num_cam):
                ymin = car["views"][j]["ymin"]
                xmin = car["views"][j]["xmin"]
                ymax = car["views"][j]["ymax"]
                xmax = car["views"][j]["xmax"]
                if j == 0:
                    frame_left_box.append([ymin, xmin, ymax, xmax])
                else:
                    frame_right_box.append([ymin, xmin, ymax, xmax])
        return frame_bev_box, frame_left_box, frame_right_box

    def prepare_dir(self, annopath):
        frame_left_dir = []
        frame_right_dir = []
        frame_left_ang = []
        frame_right_ang = []
        frame_wxy = []
        frame_bev_angle = []
        frame_type = []

        with open(annopath) as json_file:
            cars = [json.load(json_file)][0]
        for i, car in enumerate(cars):
            wx = int(car["wx"]) // 10
            wy = int(car["wy"]) // 10
            type = car["type"]
            bev_angle = float(car["angle"])

            frame_wxy.append([wx, wy])
            frame_type.append(type)
            frame_left_dir.append(0)
            frame_right_dir.append(0)
            # 0~360
            if bev_angle < 0:
                bev_angle += 2 * np.pi
            # 左角度标签
            alpha = np.arctan((self.dataset.worldgrid_shape[0] - wy) / wx)
            left_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)
            frame_left_ang.append([np.sin(left_target), np.cos(left_target)])  # 方案1, 回归sin cos

            # 右角度标签, 颠倒一下正方向
            bev_angle -= np.pi
            if bev_angle < 0:
                bev_angle += 2 * np.pi
            frame_bev_angle.append(bev_angle)
            alpha = np.arctan(wy / (self.dataset.worldgrid_shape[1] - wx))
            right_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)
            frame_right_ang.append([np.sin(right_target), np.cos(right_target)])  # 方案1, 回归sin cos

        return frame_type, frame_wxy, frame_left_dir, frame_right_dir, frame_bev_angle, frame_left_ang, frame_right_ang


def getprojected_3dbox(points3ds, extrin, intrin, isleft=True):
    if isleft:
        extrin_ = extrin[0].reshape(1, 3, 4)
        intrin_ = intrin[0].reshape(1, 3, 3)
    else:
        extrin_ = extrin[1].reshape(1, 3, 4)
        intrin_ = intrin[1].reshape(1, 3, 3)

    # print(extrin_.shape, intrin_.shape)
    extrin_big = extrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)
    intrin_big = intrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)

    points3ds_big = points3ds.reshape(points3ds.shape[0], points3ds.shape[1], 3, 1)
    homog = np.ones((points3ds.shape[0], points3ds.shape[1], 1, 1))
    homo3dpts = np.concatenate((points3ds_big, homog), 2).reshape(points3ds.shape[0] * points3ds.shape[1], 4, 1)
    res = np.matmul(extrin_big, homo3dpts)
    Zc = res[:, -1]
    # print(intrin_big.shape, res.shape)
    res2 = np.matmul(intrin_big, res)
    imagepoints = (res2.reshape(-1, 3) / Zc).reshape((points3ds.shape[0], points3ds.shape[1], 3))[:, :, :2].astype(int)

    return imagepoints

