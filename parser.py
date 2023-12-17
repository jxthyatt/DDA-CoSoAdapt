# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json

def parser_(parser):
    parser.add_argument('--root', type=str, default='/mnt/traffic1/data/jxt')
    parser.add_argument('--model_name', type=str, default='deeplabv2')
    parser.add_argument('--name', type=str, default='gta2cty')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--freeze_bn', action='store_true')
    parser.add_argument('--epochs', type=int, default=84)
    parser.add_argument('--train_iters', type=int, default=80000)
    parser.add_argument('--moving_prototype', action='store_true')
    parser.add_argument('--bn', type=str, default='sync_bn')
    #training
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--stage', type=str, default='stage1')
    parser.add_argument('--finetune', action='store_true')
    #model
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_bn', action='store_true')
    parser.add_argument("--student_init", default='stage1', type=str)
    parser.add_argument("--proto_momentum", default=0.0001, type=float)
    parser.add_argument("--bn_clr", action='store_true')
    #data
    parser.add_argument('--src_dataset', type=str, default='gta5')
    parser.add_argument('--tgt_dataset', type=str, default='cityscapes')
    parser.add_argument('--src_rootpath', type=str, default='datasets/GTA5')
    parser.add_argument('--tgt_rootpath', type=str, default='datasets/cityscapes')
    parser.add_argument('--path_LP', type=str, default='Pseudo/hard_label')
    parser.add_argument('--path_soft', type=str, default='Pseudo/soft_label')
    parser.add_argument("--train_thred", default=0, type=float)
    parser.add_argument('--used_save_pseudo', action='store_true')
    parser.add_argument('--no_droplast', action='store_true')

    parser.add_argument('--resize', type=int, default=2200)
    parser.add_argument('--rcrop', type=str, default='896,512')
    parser.add_argument('--hflip', type=float, default=0.5)

    parser.add_argument('--n_class', type=int, default=19)
    parser.add_argument('--num_workers', type=int, default=6)
    #loss
    parser.add_argument('--gan', type=str, default='Vanilla')
    parser.add_argument('--adv', type=float, default=0.01)
    parser.add_argument('--proto_pseudo_src', type=float, default=0.0)
    parser.add_argument("--rce", action='store_true')
    parser.add_argument("--rce_alpha", default=0.1, type=float)
    parser.add_argument("--rce_beta", default=1.0, type=float)
    parser.add_argument("--regular_w", default=0, type=float)
    parser.add_argument("--regular_type", default='MRENT', type=str)
    parser.add_argument('--proto_consistW', type=float, default=1.0)
    parser.add_argument("--distill", default=0, type=float)

    parser.add_argument('--proto_pseudo', type=float, default=0.0)

    #print
    parser.add_argument('--print_interval', type=int, default=20)
    parser.add_argument('--val_interval', type=int, default=1000)

    parser.add_argument('--noshuffle', action='store_true')
    parser.add_argument('--noaug', action='store_true')

    parser.add_argument('--proto_rectify', action='store_true')
    parser.add_argument('--proto_temperature', type=float, default=1.0)
    #stage2
    parser.add_argument("--threshold", default=-1, type=float)
    return parser

def relative_path_to_absolute_path(opt):
    opt.rcrop = [int(opt.rcrop.split(',')[0]), int(opt.rcrop.split(',')[1])]
    opt.resume_path = os.path.join(opt.root, 'DDA/cosoadapt', opt.resume_path)
    opt.src_rootpath = os.path.join(opt.root, 'DDA/cosoadapt', opt.src_rootpath)
    opt.tgt_rootpath = os.path.join(opt.root, 'DDA/cosoadapt', opt.tgt_rootpath)
    opt.path_LP = os.path.join(opt.root, 'DDA/cosoadapt', opt.path_LP)
    opt.path_soft = os.path.join(opt.root, 'DDA/cosoadapt', opt.path_soft)
    opt.logdir = os.path.join(opt.root, 'DDA/cosoadapt', 'logs', opt.name)
    return opt