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
    parser.add_argument('--model-file', type=str, default='/data/checkpoint_230.pth.tar',
                        help='Model path')
    parser.add_argument(
        '--dataset', type=str, default='road', help='test folder id contain images ROIs to test'
    )
    parser.add_argument('-g', '--gpu', type=int, nargs='+', default=1)
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
    model = unet.UNet(in_channels=3, num_classes=3, base_c=64).cuda()

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
    val_cldice = 0.0
    val_iou0 = 0.0
    val_acc = 0.0
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
        iou0 = iou(prediction[0, ...], target_numpy[0, 0, ...])
        acc = calculate_mean_accuracy(prediction[0, ...], target_numpy[0, 0, ...])
        cldice = clDice(prediction[0, ...], target_numpy[0, 0, ...])

        val_cup_dice += cup_dice
        val_cldice += cldice
        val_iou0 += iou0
        val_acc += acc


        imgs = data.data.cpu()

        for img, lt, lp in zip(imgs, target_numpy, [prediction]):
            img, lt = untransform(img, lt)
            save_per_img(img.numpy().transpose(1, 2, 0), os.path.join(args.test_prediction_save_path, args.dataset),
                         img_name[0],
                         lp, mask_path=None, ext="bmp")

    val_cup_dice /= len(test_loader)
    val_cldice /= len(test_loader)
    val_iou0 /= len(test_loader)
    val_acc /= len(test_loader)

    print('''\n==>val_cup_dice : {0}'''.format(val_cup_dice))
    print('''\n==>val_cldice : {0}'''.format(val_cldice))
    print('''\n==>val_iou0 : {0}'''.format(val_iou0))
    print('''\n==>val_acc : {0}'''.format(val_acc))

    with open(osp.join(args.test_prediction_save_path, 'test_log.csv'), 'a') as f:
        elapsed_time = (
                datetime.now(pytz.timezone('Asia/Hong_Kong')) -
                timestamp_start).total_seconds()
        log = [[args.model_file] + ['cup dice coefficence: '] + \
               [val_cup_dice] + ['disc dice coefficence: '] + \
               [val_disc_dice] + [elapsed_time]]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    main()
