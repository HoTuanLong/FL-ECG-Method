import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *
from nets.nets import SEResNet18

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9999)
    parser.add_argument("--num_rounds", type = int, default = 60)
    args = parser.parse_args()

    server_model = SEResNet18(
        num_classes = 30, 
    )
    initial_parameters = [value.cpu().numpy() for key, value in server_model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)
    save_ckp_dir = "../../ckps"
    if not os.path.exists(save_ckp_dir):
        os.makedirs(save_ckp_dir)
    fl.server.start_server(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        config = fl.server.ServerConfig(num_rounds = args.num_rounds), 
        strategy = FedAvg(
            min_available_clients = 4, min_fit_clients = 4, 
            server_model = server_model, initial_parameters = initial_parameters, 
            save_ckp_dir = save_ckp_dir, 
        ), 
    )