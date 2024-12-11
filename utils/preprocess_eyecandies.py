import os
from shutil import copyfile
import cv2
import numpy as np


if __name__ == '__main__':

    dataset_path = '../data/Eyecandies/'
    target_dir = '../data/Eyecandies_preprocessed/'
    os.mkdir(target_dir)

    categories_list = os.listdir(dataset_path)

    print(categories_list)

    for category_dir in categories_list:
        category_root_path = os.path.join(dataset_path, category_dir)

        print(type(category_root_path))

        category_train_path = os.path.join(category_root_path, 'train/data')
        category_test_path = os.path.join(category_root_path, 'test_public/data')

        print(category_train_path)
        print(category_test_path)

        category_val_path = os.path.join(category_root_path, 'val/data')

        category_target_path = os.path.join(target_dir, category_dir)
        os.mkdir(category_target_path)

        os.mkdir(os.path.join(category_target_path, 'train'))
        category_target_train_good_path = os.path.join(category_target_path, 'train/good')
        category_target_train_good_rgb_path = os.path.join(category_target_train_good_path, 'rgb')
        category_target_train_good_normal_path = os.path.join(category_target_train_good_path, 'normals')
        category_target_train_good_depth_path = os.path.join(category_target_train_good_path, 'depth')
        os.mkdir(category_target_train_good_path)
        os.mkdir(category_target_train_good_rgb_path)
        os.mkdir(category_target_train_good_normal_path)
        os.mkdir(category_target_train_good_depth_path)

        os.mkdir(os.path.join(category_target_path, 'test'))
        category_target_test_good_path = os.path.join(category_target_path, 'test/good')
        category_target_test_good_rgb_path = os.path.join(category_target_test_good_path, 'rgb')
        category_target_test_good_normal_path = os.path.join(category_target_test_good_path, 'normals')
        category_target_test_good_depth_path = os.path.join(category_target_test_good_path, 'depth')
        category_target_test_good_gt_path = os.path.join(category_target_test_good_path, 'gt')
        os.mkdir(category_target_test_good_path)
        os.mkdir(category_target_test_good_rgb_path)
        os.mkdir(category_target_test_good_normal_path)
        os.mkdir(category_target_test_good_depth_path)
        os.mkdir(category_target_test_good_gt_path)
        category_target_test_bad_path = os.path.join(category_target_path, 'test/bad')
        category_target_test_bad_rgb_path = os.path.join(category_target_test_bad_path, 'rgb')
        category_target_test_bad_normal_path = os.path.join(category_target_test_bad_path, 'normals')
        category_target_test_bad_depth_path = os.path.join(category_target_test_bad_path, 'depth')
        category_target_test_bad_gt_path = os.path.join(category_target_test_bad_path, 'gt')
        os.mkdir(category_target_test_bad_path)
        os.mkdir(category_target_test_bad_rgb_path)
        os.mkdir(category_target_test_bad_normal_path)
        os.mkdir(category_target_test_bad_depth_path)
        os.mkdir(category_target_test_bad_gt_path)

        category_train_files = os.listdir(category_train_path)
        num_train_files = len(category_train_files)//17
        for i in range(0, num_train_files):
            copyfile(os.path.join(category_train_path, str(i).zfill(3)+'_image_4.png'),os.path.join(category_target_train_good_rgb_path, str(i).zfill(3)+'.png'))
            copyfile(os.path.join(category_train_path, str(i).zfill(3) + '_depth.png'),
                     os.path.join(category_target_train_good_depth_path, str(i).zfill(3) + '.png'))
            copyfile(os.path.join(category_train_path, str(i).zfill(3) + '_normals.png'),
                     os.path.join(category_target_train_good_normal_path, str(i).zfill(3) + '.png'))


        category_test_files = os.listdir(category_test_path)
        num_test_files = len(category_test_files)//17
        for i in range(0, num_test_files):
            mask = cv2.imread(os.path.join(category_test_path,str(i).zfill(2)+'_mask.png'))
            if np.any(mask):
                copyfile(os.path.join(category_test_path,str(i).zfill(2)+'_image_4.png'),os.path.join(category_target_test_bad_rgb_path, str(i).zfill(3)+'.png'))
                copyfile(os.path.join(category_test_path, str(i).zfill(2) + '_depth.png'),
                         os.path.join(category_target_test_bad_depth_path, str(i).zfill(3) + '.png'))
                copyfile(os.path.join(category_test_path, str(i).zfill(2) + '_normals.png'),
                         os.path.join(category_target_test_bad_normal_path, str(i).zfill(3) + '.png'))
            else:
                cv2.imwrite(os.path.join(category_target_test_good_gt_path, str(i).zfill(3)+'.png'), mask)
                copyfile(os.path.join(category_test_path,str(i).zfill(2)+'_image_4.png'),os.path.join(category_target_test_good_rgb_path, str(i).zfill(3)+'.png'))
                copyfile(os.path.join(category_test_path, str(i).zfill(2) + '_normals.png'),
                         os.path.join(category_target_test_good_normal_path, str(i).zfill(3) + '.png'))
                copyfile(os.path.join(category_test_path, str(i).zfill(2) + '_depth.png'),
                         os.path.join(category_target_test_good_depth_path, str(i).zfill(3) + '.png'))


        os.mkdir(os.path.join(category_target_path, 'validation'))
        category_target_val_good_path = os.path.join(category_target_path, 'validation/good')
        category_target_val_good_rgb_path = os.path.join(category_target_val_good_path, 'rgb')
        category_target_val_good_normal_path = os.path.join(category_target_val_good_path, 'normals')
        category_target_val_good_depth_path = os.path.join(category_target_val_good_path, 'depth')
        os.mkdir(category_target_val_good_path)
        os.mkdir(category_target_val_good_rgb_path)
        os.mkdir(category_target_val_good_normal_path)
        os.mkdir(category_target_val_good_depth_path)

        category_val_files = os.listdir(category_val_path)
        num_val_files = len(category_val_files) // 17
        for i in range(0, num_val_files):
            copyfile(os.path.join(category_val_path, str(i).zfill(2) + '_image_4.png'),
                     os.path.join(category_target_val_good_rgb_path, str(i).zfill(3) + '.png'))
            copyfile(os.path.join(category_val_path, str(i).zfill(2) + '_depth.png'), os.path.join(category_target_val_good_depth_path, str(i).zfill(3) + '.png'))
            copyfile(os.path.join(category_val_path, str(i).zfill(2) + '_normals.png'),
                     os.path.join(category_target_val_good_normal_path, str(i).zfill(3) + '.png'))
