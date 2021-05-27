import argparse
import os
import time
import json

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from seco_util import clip_transforms
from seco_util.util import ClipGaussianBlur, AverageMeter
from seco_util.lr_scheduler import get_scheduler
from seco_util.logger import setup_logger
from dataset.video_dataset import VideoRGBTrainDataset

from model.model_factory import get_model_by_name, transfer_weights, remove_fc

from torch.cuda.amp import GradScaler


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    parser.add_argument('--dataset-class', type=str, default='video_dataset',
                        choices=["video_dataset"], help="class of dataset")
    # for video_dataset
    parser.add_argument('--list-file', type=str, default='', help='path of list file')
    parser.add_argument('--root-path', type=str, default='', help='path of root folder')
    parser.add_argument('--format', type=str, default='LMDB',
                        choices=["LMDB"], help="video format")
    # other parameters
    parser.add_argument('--time-dim', type=str, default='C',
                        choices=["T", "C"], help="dimension for time")
    parser.add_argument('--crop-size', type=int, default=224, help='crop_size')
    parser.add_argument('--num-classes', type=int, default=101, required=True, help='num of predict classes')
    parser.add_argument('--batch-size', type=int, default=16, help='batch_size')
    parser.add_argument('--iter-size', type=int, default=4, help='iter_size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--num-segments', type=int, default=7, help='num of segments')
    parser.add_argument('--clip-length', type=int, default=1, help='num of clip length')
    parser.add_argument('--num-steps', type=int, default=1, help='num of sampling steps')
    parser.add_argument('--horizontal-flip', type=str2bool, default='true', help='if horizontal flip the data')

    # network
    parser.add_argument('--net-name', type=str, default='resnet50', help='name of network architecture')
    parser.add_argument('--pooling-name', type=str, default='PoolingAverage', help='name of pooling architecture')
    parser.add_argument('--dropout-ratio', type=float, default=0.9, help='dropout ratio')
    parser.add_argument('--frozen-bn', type=str2bool, default='True', help='if frozen batch_norm layers')
    parser.add_argument('--transfer-weights', type=str2bool, default='true', help='if transfer weights from 2D network')
    parser.add_argument('--remove-fc', type=str2bool, default='true', help='if initialized weights for fc layer')

    # optimization
    parser.add_argument('--base-learning-rate', '--base-lr', type=float, default=0.001)
    parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[50, 100], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
    parser.add_argument('--clip-gradient', type=float, default=40, help='norm to clip gradient')
    parser.add_argument('--loss-weight', type=float, default=1, help='loss weight')
    parser.add_argument('--label-smooth', type=str2bool, default='false', help='if apply label smooth')

    # io
    parser.add_argument('--pretrained-model', default='', type=str, metavar='PATH',
                        help='path to pretrained weights like imagenet (default: none)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    args = parser.parse_args()
    return args


def get_loader(args):
    train_transform = transforms.Compose([
        clip_transforms.ClipRandomResizedCrop(args.crop_size, scale=(0.2, 1.), ratio=(0.75, 1.3333333333333333)),
        transforms.RandomApply([
            clip_transforms.ClipColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        clip_transforms.ClipRandomGrayscale(p=0.2),
        transforms.RandomApply([ClipGaussianBlur([.1, 2.])], p=0.5),
        clip_transforms.ClipRandomHorizontalFlip(p=0.5 if args.horizontal_flip else 0),
        clip_transforms.ToClipTensor(),
        clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(
            lambda clip: torch.cat(clip, dim=0))
    ])

    if args.dataset_class == 'video_dataset':
        assert (args.list_file != '' and args.root_path != '')
        train_dataset = VideoRGBTrainDataset(list_file=args.list_file, root_path=args.root_path,
                                             transform=train_transform, clip_length=args.clip_length,
                                             num_steps=args.num_steps, num_segments=args.num_segments,
                                             format=args.format)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    return train_loader


def build_model(args):
    model = get_model_by_name(net_name=args.net_name, num_classes=args.num_classes, dropout_ratio=args.dropout_ratio).cuda()
    if args.pretrained_model:
        load_pretrained(args, model)
    return model


def load_pretrained(args, model):
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    if 'model' in ckpt:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    else:
        state_dict = ckpt

    # convert initial weights
    if args.transfer_weights:
        state_dict = transfer_weights(args.net_name, state_dict)
    if args.remove_fc:
        state_dict = remove_fc(args.net_name, state_dict)

    [misskeys, unexpkeys] = model.load_state_dict(state_dict, strict=False)
    logger.info('Missing keys: {}'.format(misskeys))
    logger.info('Unexpect keys: {}'.format(unexpkeys))
    logger.info("==> loaded checkpoint '{}'".format(args.pretrained_model))


def save_checkpoint(args, epoch, model, optimizer, scheduler):
    logger.info('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(args.output_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.output_dir, 'ckpt_epoch_{}.pth'.format(epoch)))


def main(args):
    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    logger.info("length of training dataset: {}".format(n_data))

    model = build_model(args)
    # print network architecture
    if dist.get_rank() == 0:
        logger.info(model)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.base_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=True, find_unused_parameters=True)

    # routine
    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        tic = time.time()
        loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, args)
        logger.info('epoch {}, total time {:.2f}, loss is {:4f}'.format(epoch, time.time() - tic, loss))
        if dist.get_rank() == 0:
            # save model
            save_checkpoint(args, epoch, model, scheduler, optimizer)


def frozen_bn(model):
    first_bn = True
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if first_bn:
                first_bn = False
                print('Skip frozen first bn layer: ' + name)
                continue
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def train(epoch, train_loader, model, criterion, optimizer, scheduler, args):
    model.train()
    if args.frozen_bn:
        frozen_bn(model)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    end = time.time()

    optimizer.zero_grad()
    scaler = GradScaler()
    bnorm = 0

    for idx, train_data in enumerate(train_loader):
        x = train_data[0]
        label = train_data[1]

        bsz = x.size(0)

        # forward
        x = x.cuda(non_blocking=True)  # clip
        label = label.cuda(non_blocking=True)  # label

        # with torch.cuda.amp.autocast():
        # forward and get the predict score
        score = model(x)
        # get crossentropy loss
        if isinstance(score, list):
            loss = criterion(score[0], label) + criterion(score[1], label)
        else:
            loss = criterion(score, label)

        # backward
        scaler.scale(loss / args.iter_size * args.loss_weight).backward()

        if (idx + 1) % args.iter_size == 0:
            scaler.unscale_(optimizer)
            bnorm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                                   args.clip_gradient)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        # update meters
        loss_meter.update(loss.item(), bsz)
        norm_meter.update(bnorm, bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0:
            logger.info(
                'Train: [{:>3d}]/[{:>4d}/{:>4d}] BT={:>0.3f}/{:>0.3f} LR={:>0.3f} Loss={:>0.3f}/{:>0.3f} GradNorm={:>0.3f}/{:>0.3f}'.format(
                    epoch, idx, len(train_loader),
                    batch_time.val, batch_time.avg,
                    next(iter(optimizer.param_groups))['lr'],
                    loss.item(), loss_meter.avg,
                    bnorm, norm_meter.avg
                ))

    return loss_meter.avg


if __name__ == '__main__':
    opt = parse_option()
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="seco-finetune")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "seco_finetune.config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    main(opt)
