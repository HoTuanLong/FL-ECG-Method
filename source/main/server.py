import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *
from data import ECGDataset
from models.models import ResNet18
from engines import server_test_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9999)
    parser.add_argument("--dataset", type = str, default = "PhysioNet"), parser.add_argument("--subdataset", type = str)
    parser.add_argument("--num_classes", type = int, default = 30)
    parser.add_argument("--num_clients", type = int, default = 4)
    parser.add_argument("--num_rounds", type = int, default = 100)
    parser.add_argument("--num_epochs", type = int, default = 1)
    args = parser.parse_args()

    if args.subdataset is not None:
        test_loaders = {
            "test":torch.utils.data.DataLoader(
                ECGDataset(
                    df_path = "../../datasets/{}/{}/csvs/test.csv".format(args.dataset, args.subdataset), data_dir = "../../datasets/{}/{}/ecgs".format(args.dataset, args.subdataset), 
                ), 
                num_workers = 0, batch_size = 80, 
                shuffle = True, 
            ), 
        }
    server_model = ResNet18(
        num_classes = 30, 
    )

    initial_parameters = [value.cpu().numpy() for key, value in server_model.state_dict().items()]
    initial_parameters = flwr.common.ndarrays_to_parameters(initial_parameters)
    save_ckp_dir = "../../ckps/{}".format(args.dataset)
    if not os.path.exists(save_ckp_dir):
        os.makedirs(save_ckp_dir)
    if args.subdataset is None:
        flwr.server.start_server(
            server_address = "{}:{}".format(args.server_address, args.server_port), 
            config = flwr.server.ServerConfig(num_rounds = args.num_rounds), 
            strategy = FedAvg(
                min_available_clients = args.num_clients, min_fit_clients = args.num_clients, 
                server_model = server_model, 
                initial_parameters = initial_parameters, 
                save_ckp_dir = save_ckp_dir, 
            ), 
        )

    server_model = torch.load(
        "{}/server-best.ptl".format(save_ckp_dir), 
        map_location = "cpu", 
    )
    results = server_test_fn(
        test_loaders["test"], 
        server_model, 
        device = torch.device("cuda"), 
    )