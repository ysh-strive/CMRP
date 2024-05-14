import sys

from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs
from utils.utilsLog import Logger

logger = setup_logger('Test', save_dir="./log/", if_train="test", distributed_rank=20)
log_path = './log/test_log/'
sys.stdout = Logger(log_path + time.strftime("%Y%m%d_%H%M%S", time.localtime()) +'dis.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    # parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/configs.yaml')

    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)