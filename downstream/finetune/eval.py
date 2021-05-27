import argparse
import os
import json

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from seco_util import clip_transforms
from seco_util.logger import setup_logger
from dataset.video_dataset import VideoRGBTestDataset

from model.model_factory import get_model_by_name
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    parser.add_argument('--list-file', type=str, required=True, help='list of dataset')
    parser.add_argument('--root-path', type=str, required=True, help='root path of dataset')
    parser.add_argument('--format', type=str, default='LMDB',
                        choices=["LMDB"], help="video format")
    # other parameters
    parser.add_argument('--time-dim', type=str, default='C',
                        choices=["T", "C"], help="dimension for time")
    parser.add_argument('--crop-size', type=int, default=256, help='crop_size')
    parser.add_argument('--num-classes', type=int, required=True, help='num of predict classes')
    parser.add_argument('--batch-size', type=int, default=16, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--clip-length', type=int, default=16, help='num of clip length')
    parser.add_argument('--num-steps', type=int, default=1, help='num of sampling steps')
    parser.add_argument('--num-segments', type=int, default=1, help='num of segments')
    parser.add_argument('--num-clips', type=int, default=20, help='num of sampled clips')
    parser.add_argument('--num-gpu', type=int, default=4, help='num of gpu')

    # network
    parser.add_argument('--net-name', type=str, default='resnet50', help='name of network architecture')
    parser.add_argument('--pooling-name', type=str, default='PoolingAverage', help='name of pooling architecture')
    parser.add_argument('--dropout-ratio', type=float, default=0.5, help='dropout ratio')

    # io
    parser.add_argument('--pretrained-model', default='', type=str, metavar='PATH',
                        help='path to pretrained weights like imagenet (default: none)')
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--num-crop', type=int, default=3, help='the place index [0,1,2]')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    args = parser.parse_args()
    return args


def get_loader(args):
    if args.crop_idx == 0:
        crop = clip_transforms.ClipCenterCrop
    elif args.crop_idx == 1:
        crop = clip_transforms.ClipFirstCrop
    elif args.crop_idx == 2:
        crop = clip_transforms.ClipThirdCrop

    test_transform = transforms.Compose([
        clip_transforms.ClipResize(size=args.crop_size),
        crop(size=args.crop_size),
        clip_transforms.ToClipTensor(),
        clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if args.time_dim == "T" else transforms.Lambda(
            lambda clip: torch.cat(clip, dim=0))
    ])

    test_dataset = VideoRGBTestDataset(args.list_file, num_clips=args.num_clips,
                                       transform=test_transform, root_path=args.root_path,
                                       clip_length=args.clip_length, num_steps=args.num_steps,
                                       num_segments=args.num_segments,
                                       format=args.format)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=test_sampler, drop_last=False)
    return test_loader


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

    [misskeys, unexpkeys] = model.load_state_dict(state_dict, strict=False)
    logger.info('Missing keys: {}'.format(misskeys))
    logger.info('Unexpect keys: {}'.format(unexpkeys))
    logger.info("==> loaded checkpoint '{}'".format(args.pretrained_model))

def merge_score(opt, logger):
    num_gpu = opt.num_gpu
    num_crop = opt.num_crop
    num_cls = opt.num_classes
    num_clip = opt.num_clips
    score_dir = opt.output_dir
    list_dir = opt.list_file

    for crop_id in range(num_crop):
        all_num = 0
        all_data = []
        for gpu_id in range(num_gpu):
            all_data.append(np.load(os.path.join(score_dir, 'all_scores_' + str(crop_id * num_gpu + gpu_id) + '.npy')))
            all_num += all_data[-1].shape[0]

        merge_data = np.empty((all_num, num_cls))
        for gpu_id in range(num_gpu):
            merge_data[gpu_id::num_gpu, :] = all_data[gpu_id]

        # make ave score
        num_video = all_num // num_clip
        merge_data = merge_data[0:num_video * num_clip, :]
        if crop_id == 0:
            reshape_data = np.zeros((num_video, num_clip, num_cls))

        reshape_data += np.reshape(merge_data, (num_video, num_clip, num_cls)) / num_crop

    # make gt
    gt = np.zeros((num_video,))
    lines = open(list_dir, 'r').readlines()
    for idx, line in enumerate(lines):
        ss = line.split(' ')
        label = ss[-1]
        gt[idx] = int(label)

    pred = (reshape_data.mean(axis=1)).argmax(axis=1)
    acc = (pred == gt).mean()

    logger.info('Top-1 accuracy is {}'.format(acc))


def main(args):
    model = build_model(args)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    for i in range(args.num_crop):
        args.crop_idx = i
        test_loader = get_loader(args)
        n_data = len(test_loader.dataset)
        logger.info("{}th crop for testing, length of testing dataset: {}".format(i, n_data))

        # routine
        all_scores = np.zeros([len(test_loader) * args.batch_size, args.num_classes], dtype=np.float)
        top_idx = 0

        with torch.no_grad():
            for idx, (x, cls) in enumerate(test_loader):
                if (idx % 100 == 0) or (idx == len(test_loader) - 1):
                    logger.info('{}/{}'.format(idx, len(test_loader)))
                bsz = x.size(0)
                score = model(x)
                if isinstance(score, list):
                    score_numpy = (score[0].data.cpu().numpy() + score[1].data.cpu().numpy()) / 2
                else:
                    score_numpy = score.data.cpu().numpy()
                all_scores[top_idx: top_idx + bsz, :] = score_numpy
                top_idx += bsz

        all_scores = all_scores[:top_idx, :]
        np.save(os.path.join(args.output_dir, 'all_scores_{}.npy'.format(
            torch.distributed.get_world_size() * args.crop_idx + args.local_rank)), all_scores)
    dist.barrier()
    # evaluate
    merge_score(args, logger)
    logger.info('Finish !')


if __name__ == '__main__':
    opt = parse_option()
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="seco-finetune")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "seco_finetune_eval.config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
    main(opt)
