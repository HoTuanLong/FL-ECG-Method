import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *
from data import ECGDataset
from models.models import ResNet18
from engines_method import client_fit_fn

class Client(flwr.client.NumPyClient):
    def __init__(self, 
        fit_loaders, num_epochs, 
        client_model, 
        client_optim, 
        save_ckp_dir, 
        dataset,
    ):
        self.fit_loaders, self.num_epochs,  = fit_loaders, num_epochs, 
        self.client_model = client_model
        self.client_optim = client_optim
        self.save_ckp_dir = save_ckp_dir
        self.dataset = dataset
        self.round = 1

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            client_optim, 
            eta_min = 1e-4, T_max = client_optim.num_rounds, 
        )

        self.evaluate_loss, self.evaluate_f1 = 0.0, 0.0
    
    def calculate_alpha(self, dataset_1, dataset_2):
        df = pd.read_csv("../../datasets/PhysioNet/train_num_samples.csv")
        metrics_1 = df[df['Dataset'] == dataset_1].values.tolist()[0][1:]
        metrics_2 = df[df['Dataset'] == dataset_2].values.tolist()[0][1:]
        
        metrics_1 = np.array(metrics_1)
        metrics_2 = np.array(metrics_2)
        cosine = np.dot(metrics_1, metrics_2)/(np.linalg.norm(metrics_1)*np.linalg.norm(metrics_2))
        
        return cosine
        
    # Do the aggragate metrics for each clients' weight
    def aggregate(self) -> NDArrays:
        target_weight = torch.load("../../temp_models/{}.ptl".format(self.dataset))
        other_weights_name = [f for f in os.listdir("../../temp_models") if os.path.isfile(os.path.join("../../temp_models", f))]
        model_target = target_weight.state_dict()
        list_alpha = []
        for weight_name in other_weights_name:
            if weight_name.split(".")[-2] == self.dataset:
                list_alpha.append(1)
                continue
            else:
                print("CHECK")
                # weight = torch.load("../../temp_models/{}".format(weight_name))
                # for (name1, params1), (name2, params2) in zip(target_weight.named_parameters(), weight.named_parameters()):
                print(weight_name.split(".")[-2])
                alpha = self.calculate_alpha(weight_name.split(".")[-2], self.dataset)
                #     params1.data = (params1.data) + (params2.data) * alpha
                list_alpha.append(alpha)
                model_source = torch.load("../../temp_models/{}".format(weight_name)).state_dict()
                for key in model_target:
                    model_target[key] = model_target[key] + model_source[key] * alpha
        # case 2: Divided by the total of alpha values
        print("list_alpha:", list_alpha)
        for key in model_target:
            model_target[key] = model_target[key] / sum(list_alpha)
        return model_target

    def get_parameters(self, 
        config, 
    ):
        parameters = [value.cpu().numpy() for key, value in self.client_model.state_dict().items()]
        return parameters

    def set_parameters(self, 
        parameters, 
    ):
        if self.round == 1:
            keys = [key for key in self.client_model.state_dict().keys()]
            self.client_model.load_state_dict(
                collections.OrderedDict({key:torch.tensor(value) for key, value in zip(keys, parameters)}), 
                strict = False, 
            )
        else:
            self.client_model.load_state_dict(self.aggregate())

    def fit(self, 
        parameters, config, 
    ):
        # keys = [key for key in self.client_model.state_dict().keys()]
        # self.client_model.load_state_dict(
        #     collections.OrderedDict({key:torch.tensor(value) for key, value in zip(keys, parameters)}), 
        #     strict = False, 
        # )
        self.set_parameters(parameters)

        self.lr_scheduler.step()
        results = client_fit_fn(
            self.fit_loaders, self.num_epochs, 
            self.client_model, 
            self.client_optim, 
            self.dataset,
            device = torch.device("cuda"), 
        )
        
        evaluate_loss, evaluate_f1 = results["evaluate_loss"], results["evaluate_f1"]
        if evaluate_f1 > self.evaluate_f1:
            torch.save(
                self.client_model, 
                "{}/client-best.ptl".format(self.save_ckp_dir), 
            )
            self.evaluate_f1 = evaluate_f1
        
        wandb.log(
            {"fit_loss":results["fit_loss"], "fit_f1":results["fit_f1"]}, 
            step = self.round, 
        )
        wandb.log(
            {"evaluate_loss":results["evaluate_loss"], "evaluate_f1":results["evaluate_f1"]}, 
            step = self.round, 
        )
        self.round += 1
        return self.get_parameters({}), len(self.fit_loaders["fit"].dataset), results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9990)
    parser.add_argument("--dataset", type = str, default = "PhysioNet"), parser.add_argument("--subdataset", type = str)
    parser.add_argument("--num_classes", type = int, default = 30)
    parser.add_argument("--num_clients", type = int, default = 4)
    parser.add_argument("--num_rounds", type = int, default = 500)
    parser.add_argument("--num_epochs", type = int, default = 1)
    parser.add_argument("--wandb_key", type = str, default = "258ef894d4067010995a4d5afc371a67fd185a7b")
    parser.add_argument("--wandb_entity", type = str, default = "longht")
    args = parser.parse_args()
    
    wandb.login(key = args.wandb_key)
    wandb.init(
        entity = args.wandb_entity, 
        project = "[fl-ecg] federated", 
        name = f"{args.subdataset} - method_v3", 
        mode = "offline"
    )

    fit_loaders = {
        "fit":torch.utils.data.DataLoader(
            ECGDataset(
                df_path = "../../datasets/{}/{}/csvs/fit.csv".format(args.dataset, args.subdataset), data_dir = "../../datasets/{}/{}/ecgs".format(args.dataset, args.subdataset), 
            ), 
            num_workers = 4, batch_size = 80, 
            shuffle = True, 
        ), 
        "evaluate":torch.utils.data.DataLoader(
            ECGDataset(
                df_path = "../../datasets/{}/{}/csvs/evaluate.csv".format(args.dataset, args.subdataset), data_dir = "../../datasets/{}/{}/ecgs".format(args.dataset, args.subdataset), 
            ), 
            num_workers = 4, batch_size = 80, 
            shuffle = True, 
        ), 
    }
    client_model = ResNet18(
        num_classes = 30, 
    )
    client_optim = optim.Adam(
        client_model.parameters(), weight_decay = 5e-5, 
        lr = 1e-3, 
    )
    client_optim.num_rounds = args.num_rounds

    save_ckp_dir = "../../ckps/{}/{}".format(args.dataset, args.subdataset)
    if not os.path.exists(save_ckp_dir):
        os.makedirs(save_ckp_dir)
    client = Client(
        fit_loaders, args.num_epochs, 
        client_model, 
        client_optim, 
        save_ckp_dir, 
        args.subdataset,
    )
    flwr.client.start_numpy_client(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        client = client, 
    )