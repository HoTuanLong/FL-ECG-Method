import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ECGDataset
from nets.nets import SEResNet18
from engines import test_fn

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str)
args = parser.parse_args()

test_loader = torch.utils.data.DataLoader(
    ECGDataset(
        df_path = "../../datasets/PhysioNet/{}/csvs/test.csv".format(args.dataset), data_path = "../../datasets/PhysioNet/{}/ecgs".format(args.dataset), 
    ), 
    num_workers = 8, batch_size = 80, 
    shuffle = False, 
)
model = torch.load(
    "../../ckps/{}/best.ptl".format(args.dataset), 
    map_location = "cpu", 
)

test_fn(
    test_loader, 
    model, 
)