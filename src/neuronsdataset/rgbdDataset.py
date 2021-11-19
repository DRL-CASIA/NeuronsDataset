from torchvision.datasets import VisionDataset
import os

class rgbdDataset(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = "rgbd"
        self.root = root
        i = 0
        camera_folder = os.path.join(self.root, "img")
        self.num_frame = len(os.listdir(camera_folder))
        self.all_indexs = []
        while i < self.num_frame:
            file = os.listdir(camera_folder)[i]
            self.all_indexs.append(int(file.split("/")[-1].split(".")[0]))
            i +=1


    def get_image_fpaths(self, frame_range):
        img_fpaths = {}
        camera_folder = os.path.join(self.root, "img")
        for fname in sorted(os.listdir(camera_folder)):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                img_fpaths[frame] = os.path.join(camera_folder, fname)
        return img_fpaths


    def get_anno_fpaths(self, frame_range):
        anno_fpaths = {}
        anno_folder = os.path.join(self.root, "dep")
        for fname in sorted(os.listdir(anno_folder)):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                anno_fpaths[frame] = os.path.join(anno_folder, fname)
        return anno_fpaths

    def get_visualize_info(self, idx: int):
        imgs = self.get_image_fpaths([idx])
        annos = self.get_anno_fpaths([idx])

        return imgs, annos, None, None