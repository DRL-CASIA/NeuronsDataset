from torchvision.datasets import VisionDataset
import os

class rgbDataset(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = "rgb"
        self.root = root
        camera_folder = os.path.join(self.root, "JPEGImages")
        self.num_frame = len(os.listdir(camera_folder))
        i = 0
        self.all_indexs = []
        while i < self.num_frame:
            file = os.listdir(camera_folder)[i]
            self.all_indexs.append(int(file.split("/")[-1].split(".")[0]))
            i +=1

    def get_image_fpaths(self, frame_range):
        img_fpaths = {}
        camera_folder = os.path.join(self.root, "JPEGImages")
        # print(camera_folder)
        for fname in sorted(os.listdir(camera_folder)):
            # print(fname)
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                img_fpaths[frame] = os.path.join(camera_folder, fname)
        return img_fpaths

    def get_anno_fpaths(self, frame_range):
        anno_fpaths = {}
        anno_folder = os.path.join(self.root, "labels")
        for fname in sorted(os.listdir(anno_folder)):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                anno_fpaths[frame] = os.path.join(anno_folder, fname)
        return anno_fpaths

    def get_visualize_info(self, idx: int):
        imgs = self.get_image_fpaths([idx])
        annos = self.get_anno_fpaths([idx])

        return imgs, annos, None, None
