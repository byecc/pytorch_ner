import torch
import argparse
import pack_embedding
from ner.datadef import *
import torch.optim as optim
import random
import numpy as np


seed_num=233
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def train(data):
    print("Training model...")
    if data.optimizer == "SGD":
        optim.SGD()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiLSTM+CRF")
    parser.add_argument('--config', help="Config File")

    args = parser.parse_args()
    data = Data()
    data.read_config(args.config)
    status = data.status
    # data.show_config()

    if status.lower() == "train":
        data.get_instance("train")
        data.get_instance("dev")
        data.get_instance("test")
        data.build_alphabet()
        data.get_instance_index("train")
        data.get_instance_index("dev")
        data.get_instance_index("test")
        train(data)