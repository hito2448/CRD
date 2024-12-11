from torch.utils.data import Dataset
import os
from PIL import Image


from .geo_utils import *

class EyeRGBDDataset(Dataset):
    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0):
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, normals_path, gt, ad_label, ad_type = self.data_info[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        normals_img = Image.open(normals_path).convert('RGB')
        if self.transform is not None:
            normals_img = self.transform(normals_img)
        depth_img = normals_img

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
                    depth_path = os.path.join(root, sub_dir, 'depth', rgb_name)
                    normals_path = os.path.join(root, sub_dir, 'normals', rgb_name)
                    if sub_dir == 'good':
                        data_info.append((rgb_path, depth_path, normals_path, 0, 0, sub_dir))
                    else:
                        gt_name = rgb_name
                        gt_path = os.path.join(root, sub_dir, 'gt', gt_name)
                        data_info.append((rgb_path, depth_path, normals_path, gt_path, 1, sub_dir))

            break

        np.random.shuffle(data_info)

        return data_info


class EyeRGBDDataset_test(Dataset):
    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0):
        # gt_dir == None --> train
        # gt_dir != None -->test
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, normals_path, gt, ad_label, ad_type = self.data_info[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        # print(np.array(rgb_img))
        # print(img_path)
        # print(np.array(Image.open('./data/mvtec_anomaly_detection/bottle/train/good/189.png')))
        # exit(0)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        normals_img = Image.open(normals_path).convert('RGB')
        # normals_img = np.array(normals_img)
        # print(np.array(normals_img))
        if self.transform is not None:
            normals_img = self.transform(normals_img)
            # print(normals_img)
        depth_img = normals_img

        if gt == 0:
            gt = torch.zeros([1, rgb_img.size()[-2], rgb_img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)
                # print(gt[0][0])

        assert rgb_img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        # print(torch.max(rgb_img))
        # print(torch.min(rgb_img))

        return rgb_img, depth_img, gt, ad_label, ad_type, rgb_path

    def get_data_info(self, data_dir):
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            # print(dirs)
            for sub_dir in dirs:
                rgb_names = os.listdir(os.path.join(root, sub_dir, 'rgb'))
                rgb_names = list(filter(lambda x: x.endswith('.png'), rgb_names))
                for rgb_name in rgb_names:
                    rgb_path = os.path.join(root, sub_dir, 'rgb', rgb_name)
                    # depth_name = rgb_name.replace(".png", ".tiff")
                    # depth_path = os.path.join(root, sub_dir, 'xyz', depth_name)
                    depth_path = os.path.join(root, sub_dir, 'depth', rgb_name)
                    normals_path = os.path.join(root, sub_dir, 'normals', rgb_name)
                    if sub_dir == 'good':
                        data_info.append((rgb_path, depth_path, normals_path, 0, 0, sub_dir))
                    else:
                        # gt_name = rgb_path.replace(".png", "_mask.png")
                        gt_name = rgb_name
                        gt_path = os.path.join(root, sub_dir, 'gt', gt_name)
                        data_info.append((rgb_path, depth_path, normals_path, gt_path, 1, sub_dir))

            break

        np.random.shuffle(data_info)

        return data_info

