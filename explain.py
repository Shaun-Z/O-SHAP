import time
from tqdm import tqdm
from options.train_options import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options