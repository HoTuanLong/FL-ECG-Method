import os, sys
from libs import *

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        server_model, 
        save_ckp_dir = "./", 
        *args, **kwargs, 
    ):
        self.server_model = server_model
        self.save_ckp_dir = save_ckp_dir
        super().__init__(*args, **kwargs, )

    def aggregate_fit(self, 
        server_round, 
        results, failures, 
    ):
        aggregated_parameters = super().aggregate_fit(
            server_round, 
            results, failures, 
        )[0]
        aggregated_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
        aggregated_keys = [key for key in self.server_model.state_dict().keys()]
        self.server_model.load_state_dict(
            collections.OrderedDict({key:torch.tensor(value) for key, value in zip(aggregated_keys, aggregated_parameters)}), 
            strict = False, 
        )
        torch.save(
            self.server_model, 
            "{}/server_model.ptl".format(self.save_ckp_dir), 
        )

        aggregated_parameters = [value.cpu().numpy() for key, value in self.server_model.state_dict().items()]
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters, {}