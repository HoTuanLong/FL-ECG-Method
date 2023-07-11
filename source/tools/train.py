import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ECGDataset
from nets.nets import SEResNet18
from engines import train_fn

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str)
args = parser.parse_args()

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ECGDataset(
            df_path = "../../datasets/PhysioNet/{}/csvs/train.csv".format(args.dataset), data_path = "../../datasets/PhysioNet/{}/ecgs".format(args.dataset), 
        ), 
        num_workers = 8, batch_size = 224, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ECGDataset(
            df_path = "../../datasets/PhysioNet/{}/csvs/val.csv".format(args.dataset), data_path = "../../datasets/PhysioNet/{}/ecgs".format(args.dataset), 
        ), 
        num_workers = 8, batch_size = 224, 
        shuffle = True, 
    ), 
}
model = SEResNet18(
    num_classes = 30, 
)
optimizer = optim.Adam(
    model.parameters(), weight_decay = 5e-5, 
    lr = 1e-3, 
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    eta_min = 1e-4, T_max = 50, 
)

save_ckp_dir = "../../ckps/{}".format(args.dataset)
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders = train_loaders, num_epochs = 50, 
    model = model, 
    optimizer = optimizer, 
    scheduler = scheduler, 
    save_ckp_dir = save_ckp_dir, 
)