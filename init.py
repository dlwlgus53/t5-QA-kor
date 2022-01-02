import torch
import numpy as np
import random
from log_conf import init_logger

def init_experiment(args):
    init_logger(f'{args.save_prefix}QA.log')
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
