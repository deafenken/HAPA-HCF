import argparse
from random import seed
import os

def init_args(args):
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    return args
descript = 'Pytorch Implementation of VAF'
parser = argparse.ArgumentParser(description=descript)
parser.add_argument('--output_path', type=str, default='test_outputs/')
parser.add_argument('--save_model_path', type=str, default='saved_models/ap87.65/', help='the path of trained model file')
parser.add_argument('--rgb-list', default='./lists/video_train.list', help='list of rgb features')
parser.add_argument('--flow-list', default='./lists/flow_train.list', help='list of flow features')
parser.add_argument('--audio-list', default='./lists/audio_train.list', help='list of audio features')
parser.add_argument('--test-rgb-list', default='./lists/video_test.list', help='list of test rgb features ')
parser.add_argument('--test-flow-list', default='./lists/flow_test.list', help='list of test flow features')
parser.add_argument('--test-audio-list', default='./lists/audio_test.list', help='list of test audio features')
parser.add_argument('--workers', default=8, help='number of workers in dataloader')
parser.add_argument('--max_seqlen', type=int, default=200, help='maximum sequence length during training')
parser.add_argument('--gt', default='./models/gt.npy', help='file of ground truth ')