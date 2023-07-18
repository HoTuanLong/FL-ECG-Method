import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *
from nets.nets import SEResNet18
from data import ECGDataset
from engines import client_fit_fn

class Client(fl.client.NumPyClient):
    def __init__(self, 
        fit_loaders, client_num_epochs, 
        client_model, 
        optimizer, 
        scheduler, 
    ):
        self.fit_loaders, self.client_num_epochs,  = fit_loaders, client_num_epochs, 
        self.client_model = client_model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.client_model = self.client_model.cuda()
        self.round = 1

    def get_parameters(self, 
        config, 
    ):
        parameters = [value.cpu().numpy() for key, value in self.client_model.state_dict().items()]

        return parameters

    def set_parameters(self, 
        parameters, 
    ):
        keys = [key for key in self.client_model.state_dict().keys()]
        self.client_model.load_state_dict(
            collections.OrderedDict({key:torch.tensor(value) for key, value in zip(keys, parameters)}), 
            strict = False, 
        )
    def fit(self, 
        parameters, 
        config, 
    ):
        self.set_parameters(parameters)
        self.client_model.train()

        self.scheduler.step()
        results = client_fit_fn(
            self.fit_loaders, self.client_num_epochs, 
            self.client_model, 
            self.optimizer, 
        )
        self.round += 1
        wandb.log(
            {
                "fit_loss":results["fit_loss"], "fit_f1":results["fit_f1"]
            }, 
            step = epoch, 
        )
        wandb.log(
            {
                "evaluate_loss":results["evaluate_loss"], "evaluate_f1":results["evaluate_f1"]
            }, 
            step = epoch, 
        )

        return self.get_parameters({}), len(self.fit_loaders["fit"].dataset), results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9999)
    parser.add_argument("--dataset", type = str)
    args = parser.parse_args()
    wandb.init(
        project = "[fl-ecg] federated", name = args.dataset, 
    )

    fit_loaders = {
        "fit":torch.utils.data.DataLoader(
            ECGDataset(
                df_path = "../../datasets/PhysioNet/{}/csvs/fit.csv".format(args.dataset), data_path = "../../datasets/PhysioNet/{}/ecgs".format(args.dataset), 
            ), 
            num_workers = 0, batch_size = 120, 
            shuffle = True, 
        ), 
        "evaluate":torch.utils.data.DataLoader(
            ECGDataset(
                df_path = "../../datasets/PhysioNet/{}/csvs/evaluate.csv".format(args.dataset), data_path = "../../datasets/PhysioNet/{}/ecgs".format(args.dataset), 
            ), 
            num_workers = 0, batch_size = 120, 
            shuffle = True, 
        ), 
    }
    client_model = SEResNet18(
        num_classes = 30, 
    )
    optimizer = optim.Adam(
        client_model.parameters(), weight_decay = 5e-5, 
        lr = 1e-3, 
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        eta_min = 1e-4, T_max = 50, 
    )

    client_num_epochs = 1
    client = Client(
        fit_loaders = fit_loaders, client_num_epochs = client_num_epochs, 
        client_model = client_model, 
        optimizer = optimizer, 
        scheduler = scheduler, 
    )
    fl.client.start_numpy_client(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        client = client, 
    )