import os, sys
from libs import *

class FedAvg(flwr.server.strategy.FedAvg):
    def __init__(self, 
        server_model, 
        *args, **kwargs, 
    ):
        self.server_model = server_model
        super().__init__(*args, **kwargs, )

    def aggregate_classifiers(self, 
    ):
        client_models = [torch.load(f) for f in glob.glob("../../ckps/PhysioNet/*/client-last.ptl")]
        for i in range(len(self.server_model.classifiers)):
            server_classifier = self.server_model.classifiers[i].state_dict()
            for key in server_classifier.keys():
                server_classifier[key] = sum([client_model.classifiers[i].state_dict()[key]*client_model.tgt_lens[i] for client_model in client_models])
                server_classifier[key] = server_classifier[key]/sum([client_model.tgt_lens[i] for client_model in client_models])
            self.server_model.classifiers[i].load_state_dict(server_classifier)

    def aggregate_fit(self, 
        server_round, 
        results, failures, 
    ):
        aggregated_parameters = super().aggregate_fit(
            server_round, 
            results, failures, 
        )[0]
        aggregated_parameters = flwr.common.parameters_to_ndarrays(aggregated_parameters)

        aggregated_keys = [key for key in self.server_model.state_dict().keys()]
        self.server_model.load_state_dict(
            collections.OrderedDict({key:torch.tensor(value) for key, value in zip(aggregated_keys, aggregated_parameters)}), 
            strict = False, 
        )
        self.aggregate_classifiers()
        aggregated_parameters = [value.cpu().numpy() for key, value in self.server_model.state_dict().items()]
        aggregated_parameters = flwr.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters, {}