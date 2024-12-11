from torch.utils.data import Dataset
import os
from PIL import Image

from .geo_utils import *

import tifffile

def get_max_min_depth_img(image_path):
    image = tifffile.imread(image_path).astype(np.float32)
    image_t = (
        np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    )
    image = image_t[:, :, 2]
    zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
    im_max = np.max(image)
    im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
    return im_min, im_max


class MVTecADRGBDDataset(Dataset):
    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0):
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)
        self.depth_transform = depth_transform

        self.global_max, self.global_min = 1, 0
        if not test:
            for data_info_i in self.data_info:
                rgb_path, depth_path, gt, ad_label, ad_type = data_info_i
                im_min, im_max = get_max_min_depth_img(depth_path)
                self.global_min = min(self.global_min, im_min)
                self.global_max = max(self.global_max, im_max)
            self.global_min = self.global_min * 0.9
            self.global_max = self.global_max * 1.1
        else:
            self.global_max = test_max
            self.global_min = test_min

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, gt, ad_label, ad_type = self.data_info[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        depth_img, plane_mask = self.get_depth_image(depth_path, rgb_img.size()[-2])

        if self.depth_transform is not None:
            depth_img = torch.cat([depth_img, depth_img, depth_img], dim=0)
            depth_img = self.depth_transform(depth_img)

        if gt == 0:
            gt = torch.zeros([1, rgb_img.size()[-2], rgb_img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)

        assert rgb_img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return rgb_img, depth_img, gt, ad_label, ad_type

    def get_data_info(self, data_dir):
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            # print(dirs)
            for sub_dir in dirs:
                rgb_names = os.listdir(os.path.join(root, sub_dir, 'rgb'))
                rgb_names = list(filter(lambda x: x.endswith('.png'), rgb_names))
                for rgb_name in rgb_names:
                    rgb_path = os.path.join(root, sub_dir, 'rgb', rgb_name)
                    depth_name = rgb_name.replace(".png", ".tiff")
                    depth_path = os.path.join(root, sub_dir, 'xyz', depth_name)
                    if sub_dir == 'good':
                        data_info.append((rgb_path, depth_path, 0, 0, sub_dir))
                    else:
                        gt_name = rgb_name
                        gt_path = os.path.join(root, sub_dir, 'gt', gt_name)
                        data_info.append((rgb_path, depth_path, gt_path, 1, sub_dir))

            break

        np.random.shuffle(data_info)

        return data_info

    def get_depth_image(self, file, size=None):
        depth_img = tifffile.imread(file).astype(np.float32)
        size = self.image_size if size is None else size
        depth_img = cv2.resize(
            depth_img, (size, size), 0, 0, interpolation=cv2.INTER_NEAREST
        )
        depth_img = np.array(depth_img)

        image = depth_img
        image_t = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], 3))
            .astype(np.float32)
        )
        image = image_t[:, :, 2]

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0 is background, 1 is foreground
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)

        image = image * plane_mask[:, :, 0]
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        image = image * 0.8 + 0.1
        image = image * (1.0 - zero_mask)  # 0 are missing pixels, the rest are in [0.1,0.9]
        image = fill_depth_map(image)  # fill missing pixels with mean of local valid values

        image = np.expand_dims(image, 2)

        depth_img = image.transpose((2, 0, 1))

        return torch.FloatTensor(depth_img), torch.FloatTensor(
            np.squeeze(plane_mask)
        )


class MVTecADRGBDDataset_test(Dataset):
    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0):
        # gt_dir == None --> train
        # gt_dir != None -->test
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)
        self.depth_transform = depth_transform

        self.global_max, self.global_min = 1, 0
        if not test:
            for data_info_i in self.data_info:
                rgb_path, depth_path, gt, ad_label, ad_type = data_info_i
                im_min, im_max = get_max_min_depth_img(depth_path)
                self.global_min = min(self.global_min, im_min)
                self.global_max = max(self.global_max, im_max)
            self.global_min = self.global_min * 0.9
            self.global_max = self.global_max * 1.1
        else:
            self.global_max = test_max
            self.global_min = test_min

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, gt, ad_label, ad_type = self.data_info[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        depth_img, plane_mask, depth_img_process = self.get_depth_image(depth_path, rgb_img.size()[-2])
        print(depth_img_process)

        if self.depth_transform is not None:
            depth_img = torch.cat([depth_img, depth_img, depth_img], dim=0)
            depth_img = self.depth_transform(depth_img)

        if gt == 0:
            gt = torch.zeros([1, rgb_img.size()[-2], rgb_img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)
                # print(gt[0][0])

        assert rgb_img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return rgb_img, depth_img, gt, ad_label, ad_type, rgb_path, depth_path, depth_img_process

    def get_data_info(self, data_dir):
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            # print(dirs)
            for sub_dir in dirs:
                rgb_names = os.listdir(os.path.join(root, sub_dir, 'rgb'))
                rgb_names = list(filter(lambda x: x.endswith('.png'), rgb_names))
                for rgb_name in rgb_names:
                    rgb_path = os.path.join(root, sub_dir, 'rgb', rgb_name)
                    depth_name = rgb_name.replace(".png", ".tiff")
                    depth_path = os.path.join(root, sub_dir, 'xyz', depth_name)
                    if sub_dir == 'good':
                        data_info.append((rgb_path, depth_path, 0, 0, sub_dir))
                    else:
                        # gt_name = rgb_path.replace(".png", "_mask.png")
                        gt_name = rgb_name
                        gt_path = os.path.join(root, sub_dir, 'gt', gt_name)
                        data_info.append((rgb_path, depth_path, gt_path, 1, sub_dir))

            break

        np.random.shuffle(data_info)

        return data_info

    def get_depth_image(self, file, size=None):
        depth_img = tifffile.imread(file).astype(np.float32)
        size = self.image_size if size is None else size
        depth_img = cv2.resize(
            depth_img, (size, size), 0, 0, interpolation=cv2.INTER_NEAREST
        )
        depth_img = np.array(depth_img)

        image = depth_img
        image_t = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], 3))
            .astype(np.float32)
        )
        image = image_t[:, :, 2]

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0 is background, 1 is foreground
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)

        # image = fill_depth_map(image) # fill missing pixels with mean of local valid values
        # zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        # image = (image - self.global_max) / (self.global_max - self.global_min)
        # image = image * (1.0 - zero_mask)

        image = image * plane_mask[:, :, 0]
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        image = image * 0.8 + 0.1
        image = image * (1.0 - zero_mask)  # 0 are missing pixels, the rest are in [0.1,0.9]
        image = fill_depth_map(image)  # fill missing pixels with mean of local valid values

        img_process = image * 100
        img_process = img_process.astype(np.uint8)

        image = np.expand_dims(image, 2)

        depth_img = image.transpose((2, 0, 1))

        # print(depth_img[0][100])

        return torch.FloatTensor(depth_img), torch.FloatTensor(
            np.squeeze(plane_mask)
        ), img_process
        # return depth_img.transpose((2, 0, 1)), np.squeeze(plane_mask)


