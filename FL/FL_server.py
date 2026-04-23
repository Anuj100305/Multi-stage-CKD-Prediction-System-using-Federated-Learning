import flwr as fl
from flwr.server.strategy import FedAvg

def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}

    accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
    return {"accuracy": accuracy}

strategy = FedAvg(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)