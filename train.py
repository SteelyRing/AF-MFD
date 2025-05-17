from datetime import datetime
import os
import os.path as osp
import imageio
imsave=imageio.imsave

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer

# Custom includes
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks import unet
from networks.dual import WDual, UDual


here = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-g', '--gpu', type=int, default=3, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument(
        '--datasetL', type=str, default='refuge1', help='test folder id contain images ROIs to test'
    )
    parser.add_argument(
        '--datasetU', type=str, default='refuge2', help='refuge'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8, help='batch size for training the model'
    )
    parser.add_argument(
        '--group-num', type=int, default=1, help='group number for group normalization'
    )
    parser.add_argument(
        '--max-epoch', type=int, default=300, help='max epoch'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=300, help='stop epoch'
    )
    parser.add_argument(
        '--warmup-epoch', type=int, default=-1, help='warmup epoch begin train GAN'
    )

    parser.add_argument(
        '--interval-validate', type=int, default=10, help='interval epoch number to valide the model'
    )
    parser.add_argument(
        '--lr-gen', type=float, default=1e-4, help='learning rate',
    )
    parser.add_argument(
        '--lr-d', type=float, default=1e-7, help='learning rate',
    )
    parser.add_argument(
        '--lr-decrease-rate', type=float, default=0.1, help='ratio multiplied to initial lr',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--data-dir',
        default='/data/fundus',
        help='data root path'
    )
    parser.add_argument(
        '--pretrained-model',
        default='../../../models/pytorch/fcn16s_from_caffe.pth',
        help='pretrained model of FCN16s',
    )
    parser.add_argument(
        '--out-stride',
        type=int,
        default=16,
        help='out-stride of deeplabv3+',
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

    args = parser.parse_args()

    args.model = 'FCN8s'

    now = datetime.now()
    args.out = osp.join(here, 'logs', args.datasetT, now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)


    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(512),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomCrop(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetL, split='train',
                                                         transform=composed_transforms_tr)
    domain_loaderL = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    domain_U = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetU, split='train',
                                                             transform=composed_transforms_tr)
    domain_loaderU = DataLoader(domain_U, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetU, split='train',
                                       transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 2. model
    model_gen = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride,
                        sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()
    #model_gen = unet.UNet(in_chns = 3, class_num = 3).cuda()
    model_d = WDual().cuda()
    model_d2 = UDual().cuda()
    
    start_epoch = 0
    start_iteration = 0

    # 3. optimizer

    optim_gen = torch.optim.RMSprop(
        model_SEG.parameters(),
        lr=args.lr_gen,
    )

    optim_d = torch.optim.RMSprop(
        model_d.parameters(),
        lr=args.lr_d,
    )

    optim_d2 = torch.optim.RMSprop(
        model_d2.parameters(),
        lr=args.lr_d,
    )

    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model_gen.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model_gen.load_state_dict(model_dict)

        pretrained_dict = checkpoint['model_d_state_dict']
        model_dict = model_d.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_d.load_state_dict(model_dict)

        pretrained_dict = checkpoint['model_d2_state_dict']
        model_dict = model_d2.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_d2.load_state_dict(model_dict)


        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration'] + 1
        optim_gen.load_state_dict(checkpoint['optim_state_dict'])
        optim_d.load_state_dict(checkpoint['optim_d_state_dict'])
        optim_d2.load_state_dict(checkpoint['optim_d2_state_dict'])

    trainer = Trainer.Trainer(
        cuda=cuda,
        model_gen=model_gen,
        model_w_d=model_d,
        model_u_d=model_d2,
        optimizer_gen=optim_gen,
        optimizer_w_d=optim_d,
        optimizer_u_d=optim_d2,
        lr_gen=args.lr_gen,
        lr_d=args.lr_d,
        lr_decrease_rate=args.lr_decrease_rate,
        val_loader=domain_loader_val,
        domain_loaderL=domain_loaderL,
        domain_loaderU=domain_loaderU,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
        warmup_epoch=args.warmup_epoch,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()
