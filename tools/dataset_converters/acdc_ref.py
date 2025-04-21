# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ACDC dataset to mmsegmentation format')
    parser.add_argument('raw_data', help='the path of raw data')
    parser.add_argument(
        '-o', '--out_dir', help='output path', default='./data/acdc')
    parser.add_argument(
        '--split',
        choices=['fog', 'night', 'rain', 'snow'],
        default='night')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Making directories...')
    mkdir_or_exist(args.out_dir)
    for subdir in ['gt/test', 'rgb_anno/test', 'gt/train', 'rgb_anno/train']:
        mkdir_or_exist(osp.join(args.out_dir, subdir))

    print('Moving images and annotations...')

    anno_str_train = f'rgb_anon/{args.split}/train_ref/'
    gt_str_train = f'gt/{args.split}/train_ref/'

    anno_str_test = f'rgb_anon/{args.split}/val_ref/'
    gt_str_test = f'gt/{args.split}/val_ref/'

    train_count = 0
    test_count = 0

    valid_train_count = 0
    valid_test_count = 0

    missing_gt = []
    
    for rgb_dataset, gt_dataset, split in [(anno_str_train, gt_str_train, 'train'), (anno_str_test, gt_str_test, 'test')]:
        for location_dir in os.listdir(os.path.join(args.raw_data, rgb_dataset)):
            for rgb_file in os.listdir(os.path.join(args.raw_data, rgb_dataset, location_dir)):
                if rgb_file.endswith('_rgb_ref_anon.png'):
                    if split == 'train':
                        train_count += 1
                    else:
                        test_count += 1

                    # Check if the corresponding GT file exists
                    gt_file = rgb_file.replace('_rgb_ref_anon.png', '_gt_ref_labelTrainIds.png')
                    if not os.path.exists(os.path.join(args.raw_data, gt_dataset, location_dir, gt_file)):
                        missing_gt.append(os.path.join(gt_dataset, location_dir, gt_file))
                        continue

                    # Copy the rgb file
                    new_path = os.path.join(args.out_dir, f'rgb_anno/{split}', rgb_file)
                    new_path = new_path.replace('_rgb_ref_anon.png', '_rgb_anon.png')
                    shutil.copy(os.path.join(args.raw_data, rgb_dataset, location_dir, rgb_file), new_path)

                    # Copy the gt file
                    new_path = os.path.join(args.out_dir, f'gt/{split}', gt_file)
                    new_path = new_path.replace('_gt_ref_labelTrainIds.png', '_gt_labelTrainIds.png')
                    shutil.copy(os.path.join(args.raw_data, gt_dataset, location_dir, gt_file), new_path)

                    if split == 'train':
                        valid_train_count += 1
                    else:
                        valid_test_count += 1

    print(f"Train: {train_count} RGB images and {valid_train_count} GT images")
    print(f"Test: {test_count} RGB images and {valid_test_count} GT images")

    # Output the missing pairs to a file
    with open(os.path.join(args.out_dir, f'{args.split}_missing_gt.txt'), 'w') as f:
        for item in missing_gt:
            f.write("%s\n" % item)

    print('Done!')


if __name__ == '__main__':
    main()