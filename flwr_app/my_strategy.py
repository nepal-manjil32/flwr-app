##-- FedAvg is the basis for all the other starategies --##
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
import json
from .task import Net, set_weights
import torch
from datetime import datetime
import os
import wandb
from datetime import datetime

# Inheriting FedAvg class
class CustomFedAvg(FedAvg):

    # Constructor
    def __init__(self, *args, **kawrgs):
        super().__init__(*args, **kawrgs)

        self.results_to_save = {}

        # Log those same metrics to wandb
        # name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # wandb.init(project="flwr-app", name=f"custom-strategy-{name}")

    def aggregate_fit(self, 
                      server_round: int, 
                      results: list[tuple[ClientProxy, FitRes]], 
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      ) -> tuple[Parameters | None , dict[str, bool | bytes | float | int | str]]:
        
        parameters_aggregated, metrics_aggregated =  super().aggregate_fit(server_round, results, failures) # method of the parent class (FedAvg)

        # Convert parameters object into ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        # Instantiate our PyTorch model for global evaluation
        model = Net()
        set_weights(model, ndarrays)
        
        # Save the global model in the standard PyTorch way
        # Every time the global model is updated after receiving the parameters, it gets saved

        # Ensure directory exists
        save_path = f"./saved_models/global_model_round_{server_round}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model state
        torch.save(model.state_dict(), save_path)

        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self,
                 server_round: int, 
                 parameters: Parameters) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        
        loss, metrics =  super().evaluate(server_round, parameters)

        # Save all these results into a json
        my_results = {"loss": loss, **metrics}

        self.results_to_save[server_round] = my_results

        with open("results.json", "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        
        # Log to wand
        # wandb.log(my_results, step=server_round)


        return loss, metrics
    