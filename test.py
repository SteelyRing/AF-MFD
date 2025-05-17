#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
#from scipy.misc import imsave
import imageio
imsave=imageio.imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
from networks.deeplabv3 import *
from networks import unet
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='/data/AF-MFD/checkpoint_290.pth.tar',
                        help='Model path')
    parser.add_argument(
        '--dataset', type=str, default='refuge', help='test folder id contain images ROIs to test'
    )
    parser.add_argument('-g', '--gpu', type=int, default=3)

    parser.add_argument(
        '--data-dir',
        default='/data/AF-MFD/fundus',
        help='data root path'
    )
    parser.add_argument(
        '--out-stride',
        type=int,
        default=16,
        help='out-stride of deeplabv3+',
    )
    parser.add_argument(
        '--save-root-ent',
        type=str,
        default='./results/ent/',
        help='path to save ent',
    )
    parser.add_argument(
        '--save-root-mask',
        type=str,
        default='./results/mask/',
        help='path to save mask',
    )
    parser.add_argument(
        '--sync-bn',
        type=bool,
        default=True,
        help='sync-bn in deeplabv3+',
    )
    parser.add_argument(
        '--freeze-bn',
        type=bool,
        default=False,
        help='freeze batch normalization of deeplabv3+',
    )
    parser.add_argument('--test-prediction-save-path', type=str,
                        default='./results/baseline/',
                        help='Path root for test image and mask')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test',
                                    transform=composed_transforms_test)

    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # 2. model
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                    sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()
    #model = unet.UNet(in_chns=3, class_num=3).cuda()
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model_gen.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model_gen.load_state_dict(model_dict)

    except Exception:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('==> Evaluating with %s' % (args.dataset))

    val_cup_dice = 0.0
    val_disc_dice = 0.0
    val_cup_iou = 0.0
    val_disc_iou = 0.0
    val_cup_bd = 0.0
    val_disc_bd = 0.0
    val_cup_bdsd = 0.0
    val_disc_bdsd = 0.0
    val_cup_asd = 0.0
    val_disc_asd = 0.0
    val_cup_acc = 0.0
    val_disc_acc = 0.0
    timestamp_start = \
        datetime.now(pytz.timezone('Asia/Hong_Kong'))

    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),
                                         total=len(test_loader),
                                         ncols=80, leave=False):
        data = sample['image']
        target = sample['map']
        img_name = sample['img_name']
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        prediction = model(data)
        prediction = torch.nn.functional.interpolate(prediction, size=(target.size()[2], target.size()[3]),
                                                     mode="bilinear")
        data = torch.nn.functional.interpolate(data, size=(target.size()[2], target.size()[3]), mode="bilinear")
        prediction = torch.sigmoid(prediction)
        draw_ent(prediction.data.cpu()[0].numpy(), os.path.join(args.save_root_ent, args.dataset), img_name[0])
        draw_mask(prediction.data.cpu()[0].numpy(), os.path.join(args.save_root_mask, args.dataset), img_name[0])
        prediction = postprocessing(prediction.data.cpu()[0].numpy(), dataset=args.dataset)
        target_numpy = target.data.cpu().numpy()
        cup_dice = dice_coefficient_numpy(prediction[0, ...], target_numpy[0, 0, ...])
        disc_dice = dice_coefficient_numpy(prediction[1, ...], target_numpy[0, 1, ...])
        cup_iou = iou(prediction[0, ...], target_numpy[0, 0, ...])
        disc_iou = iou(prediction[1, ...], target_numpy[0, 1, ...])
        cup_bd, cup_bdsd = calculate_bd(prediction[0, ...], target_numpy[0, 0, ...])
        disc_bd, disc_bdsd = calculate_bd(prediction[1, ...], target_numpy[0, 1, ...])

        cup_asd = compute_average_surface_distance(prediction[0, ...], target_numpy[0, 0, ...])
        disc_asd = compute_average_surface_distance(prediction[1, ...], target_numpy[0, 1, ...])
        cup_acc = calculate_mean_accuracy(prediction[0, ...], target_numpy[0, 0, ...])
        disc_acc = calculate_mean_accuracy(prediction[1, ...], target_numpy[0, 1, ...])


        val_cup_dice += cup_dice
        val_disc_dice += disc_dice
        val_cup_iou += cup_iou
        val_disc_iou += disc_iou
        val_cup_bd += cup_bd
        val_disc_bd += disc_bd
        val_cup_bdsd += cup_bdsd
        val_disc_bdsd += disc_bdsd
        val_cup_asd += cup_asd
        val_disc_asd += disc_asd
        val_cup_acc += cup_acc
        val_disc_acc += disc_acc

        imgs = data.data.cpu()

        for img, lt, lp in zip(imgs, target_numpy, [prediction]):
            img, lt = untransform(img, lt)
            save_per_img(img.numpy().transpose(1, 2, 0), os.path.join(args.test_prediction_save_path, args.dataset),
                         img_name[0],
                         lp, mask_path=None, ext="bmp")

    val_cup_dice /= len(test_loader)
    val_disc_dice /= len(test_loader)
    val_cup_iou /= len(test_loader)
    val_disc_iou /= len(test_loader)
    val_cup_bd /= len(test_loader)
    val_disc_bd /= len(test_loader)
    val_cup_bdsd /= len(test_loader)
    val_disc_bdsd /= len(test_loader)
    val_cup_asd /= len(test_loader)
    val_disc_asd /= len(test_loader)
    val_cup_acc /= len(test_loader)
    val_disc_acc /= len(test_loader)

    print('''\n==>val_cup_dice : {0}'''.format(val_cup_dice))
    print('''\n==>val_disc_dice : {0}'''.format(val_disc_dice))
    print('''\n==>val_cup_iou : {0}'''.format(val_cup_iou))
    print('''\n==>val_disc_iou : {0}'''.format(val_disc_iou))

    print('''\n==>val_cup_bd : {0}'''.format(val_cup_bd))
    print('''\n==>val_disc_bd : {0}'''.format(val_disc_bd))
    print('''\n==>val_cup_bdsd : {0}'''.format(val_cup_bdsd))
    print('''\n==>val_disc_bdsd : {0}'''.format(val_disc_bdsd))

    print('''\n==>val_cup_asd : {0}'''.format(val_cup_asd))
    print('''\n==>val_disc_asd : {0}'''.format(val_disc_asd))
    print('''\n==>val_cup_acc : {0}'''.format(val_cup_acc))
    print('''\n==>val_disc_acc : {0}'''.format(val_disc_acc))
    with open(osp.join(args.test_prediction_save_path, 'test_log.csv'), 'a') as f:
        elapsed_time = (
                datetime.now(pytz.timezone('Asia/Hong_Kong')) -
                timestamp_start).total_seconds()
        log = [[args.model_file] + ['cup dice coefficence: '] + \
               [val_cup_dice] + ['disc dice coefficence: '] + \
               [val_disc_dice] + ['cup iou: '] + \
               [val_cup_iou] + ['disc iou: '] + \
               [val_disc_iou] + [elapsed_time]]
        log = map(str, log)
        f.write(','.join(log) + '\n')



if __name__ == '__main__':
    main()
