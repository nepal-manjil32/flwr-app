"""flwr-app: A Flower / PyTorch app."""
import warnings
warnings.filterwarnings("ignore")

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr_app.task import Net, get_weights , set_weights, test, get_transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from flwr.common.logger import log
from logging import INFO
import json
from flwr_app.my_strategy import CustomFedAvg

##-- Callback(function) to perform centralized evaluation of our model --##
def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised testset."""
        # Instantiate model
        net = Net()
        # Apply global_model parameters
        set_weights(net, parameters_ndarrays)
        net.to(device)
        # Run test
        loss, accuracy = test(net, testloader, device)
        log(INFO, f"Round: {server_round} -> Acc: {accuracy:.4f}, Loss: {loss:.4f}")

        return loss, {"centralized_accuracy": accuracy}

    return evaluate

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    # Loop trough all metrics received compute accuracies x examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Return weighted average accuracy
    return {"weighted_accuracy": sum(accuracies) / total_examples}

def on_fit_config(server_round: int) -> Metrics:
    """ Adjust learning rate based on the current round """
    # New hyperparameters not defined previously
    lr = 0.01
    if server_round > 2:
        lr = 0.05

    return {"lr": lr}


##-- Callback(function) for fit method of client --##
def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """ Handle metrics (received after each round) from fit method in clients """
    b_value = []
    for _, m in metrics:
        my_metric_str = m["my_metric"]
        my_metric = json.loads(my_metric_str)
        b_value.append(my_metric["b"])
        #print(b_value)

    return {"max_b": max(b_value)}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Load global test set
    testset = load_dataset("zalando-datasets/fashion_mnist")["test"] 
    testloader =  DataLoader(testset.with_transform(get_transforms()),  batch_size=32) # to deal with hugging face dataset

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        min_available_clients=2,
        initial_parameters=parameters,

        ##-- Making use of callbacks --##
        evaluate_metrics_aggregation_fn=weighted_average, 
        fit_metrics_aggregation_fn=handle_fit_metrics,   # callback to aggregate the metrics returned by the fit method of client
        on_fit_config_fn=on_fit_config, # passing to the fit function
        evaluate_fn=get_evaluate_fn(testloader, device="cpu")
        
    ) 
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
 