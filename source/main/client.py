import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *
from data import ECGDataset
from models.models import ResNet18
from engines import client_fit_fn
class Client(flwr.client.NumPyClient):
    def __init__(self, 
        fit_loaders, num_epochs, 
        client_model, 
        client_optim, 
        save_ckp_dir, 
        num_classes,
        dataset,
    ):
        self.fit_loaders, self.num_epochs,  = fit_loaders, num_epochs, 
        self.client_model = client_model
        self.freeze_model = copy.deepcopy(self.client_model)
        self.client_optim = client_optim
        self.save_ckp_dir = save_ckp_dir
        self.dataset = dataset
        self.num_classes = num_classes
        self.round = 1

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            client_optim, 
            eta_min = 1e-4, T_max = client_optim.num_rounds, 
        )

        self.evaluate_loss, self.evaluate_f1 = 0.0, 0.0
    
    
    #calculate the alpha for each weight by using cosine
    def calculate_alpha(self, dataset_1, dataset_2):
        df = pd.read_csv("../../datasets/PhysioNet/train_num_samples.csv")
        metrics_1 = df[df['Dataset'] == dataset_1].values.tolist()[0][1:]
        metrics_2 = df[df['Dataset'] == dataset_2].values.tolist()[0][1:]
        
        metrics_1 = np.array(metrics_1)
        metrics_2 = np.array(metrics_2)
        
        cosine = np.dot(metrics_1, metrics_2)/(np.linalg.norm(metrics_1)*np.linalg.norm(metrics_2))
        
        return cosine
    
    #return the number of samples of each dataset
    def calculate_num_samples(self, dataset):
        
        df = pd.read_csv("../../datasets/PhysioNet/num_samples.csv")
        metrics = df[df['Dataset'] == dataset].values.tolist()[0][1]
        return metrics
    
    def num_samples_class(self, dataset, class_):
        df = pd.read_csv("../../datasets/PhysioNet/train_num_samples.csv")
        metrics = df[df['Dataset'] == dataset].values.tolist()[0][1:]
        return metrics[class_]

    def aggregate(self) -> NDArrays:
        target_weight = torch.load("../../temp_models/{}.ptl".format(self.dataset))
        other_weights_name = [f for f in os.listdir("../../temp_models") if os.path.isfile(os.path.join("../../temp_models", f))]
        model_target = target_weight.state_dict()
        original_model = copy.deepcopy(model_target)
        keys = {key: [] for key in model_target}
        print(keys)
            
        # list_num_samples = []
        sample = {i: 0 for i in range(self.num_classes)}        
        list_alpha = []
        
        classifier_layers = []
        fe_layers = []
        total_samples = 0
        for key in model_target:
            if "classifiers" in key:
                classifier_layers.append(key)
            else:
                fe_layers.append(key)
        lower_layers = []
        higher_layers = []
        p = 0.3
        count = 0
        for key in fe_layers:
            # if "stage_3" in key or "stage_4" in key:
            if  "stage_2" in key or "stage_3" in key or "stage_4" in key:
                higher_layers.append(key)
            else:
                lower_layers.append(key)
                
        for weight_name in other_weights_name:
            if weight_name.split(".")[-2] == self.dataset:
                # list_alpha.append(1)
                print(f"Data: {self.dataset} - {self.calculate_num_samples(self.dataset)}")
                alpha = self.calculate_num_samples(self.dataset)
                list_alpha.append(alpha)
                
                for k in model_target:
                    if "classifiers" not in k and k in lower_layers:
                        keys[k].append(model_target[k] * alpha)
                    else:
                        continue
            else:
                # list_num_samples.append(self.calculate_num_samples(weight_name.split(".")[-2]))
                # alpha = self.calculate_alpha(weight_name.split(".")[-2], self.dataset)
                alpha = self.calculate_num_samples(weight_name.split(".")[-2])
                list_alpha.append(alpha)
                total_samples += alpha
                model_source = torch.load("../../temp_models/{}".format(weight_name)).state_dict()

                for key in model_target:
                    if "classifiers" not in key:
                        # model_target[key] = model_target[key] + model_source[key]
                        # keys[key].append(model_source[key] * self.calculate_num_samples(weight_name.split(".")[-2]))
                        keys[key].append(model_source[key] * alpha)
                
        # print("list_alpha:", list_alpha)
        # print("sample:", sample)
        # print("list_num_samples:", list_num_samples)

        for key in model_target:
            if "classifiers" not in key:
                if key in lower_layers:
                    # model_target[key] = model_target[key] / len(other_weights_name)
                    keys[key] = sum(keys[key])/sum(list_alpha)
                    model_target[key] = keys[key]
                    original_model[key] = keys[key]
                elif key in higher_layers:
                    keys[key] = sum(keys[key])/total_samples
                    model_target[key] = keys[key]

                    
        print("list_alpha:", list_alpha)
        print("total_samples:", total_samples)
        return model_target, original_model

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
            target_model, original_model = self.aggregate()
            self.client_model.load_state_dict(original_model)
            self.freeze_model.load_state_dict(target_model)


    def fit(self, 
        parameters, config, 
    ):

        self.set_parameters(parameters)

        self.lr_scheduler.step()
        results = client_fit_fn(
            self.fit_loaders, self.num_epochs, 
            self.client_model, 
            self.freeze_model,
            self.client_optim, 
            self.dataset,
            self.num_classes,
            device = torch.device("cuda"), 
        )
        
        evaluate_loss, evaluate_f1 = results["evaluate_loss"], results["evaluate_f1"]
        if evaluate_f1 > self.evaluate_f1:
            torch.save(
                self.client_model, 
                "{}/client-best.ptl".format(self.save_ckp_dir), 
            )
            self.evaluate_f1 = evaluate_f1
        
        # wandb.log(
        #     {"fit_loss":results["fit_loss"], "fit_f1":results["fit_f1"]}, 
        #     step = self.round, 
        # )
        wandb.log(
            {"evaluate_loss":results["evaluate_loss"], "evaluate_f1":results["evaluate_f1"],
            "fit_loss":results["fit_loss"], "fit_f1":results["fit_f1"]}, 
            step = self.round, 
        )
        self.round += 1
        return self.get_parameters({}), len(self.fit_loaders["fit"].dataset), results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9999)
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
        name = f"{args.subdataset} - keep_classifier_cosine", 
        mode = "offline"
    )

    fit_loaders = {
        "fit":torch.utils.data.DataLoader(
            ECGDataset(
                df_path = "../../datasets/{}/{}/csvs/fit_new.csv".format(args.dataset, args.subdataset), data_dir = "../../datasets/{}/{}/ecgs".format(args.dataset, args.subdataset), 
            ), 
            num_workers = 4, batch_size = 80, 
            shuffle = True, 
        ), 
        "evaluate":torch.utils.data.DataLoader(
            ECGDataset(
                df_path = "../../datasets/{}/{}/csvs/evaluate_new.csv".format(args.dataset, args.subdataset), data_dir = "../../datasets/{}/{}/ecgs".format(args.dataset, args.subdataset), 
            ), 
            num_workers = 4, batch_size = 80, 
            shuffle = True, 
        ), 
    }
    client_model = ResNet18(
        num_classes = int(args.num_classes), 
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
        args.num_classes,
        args.subdataset
    )
    flwr.client.start_numpy_client(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        client = client, 
    )
1