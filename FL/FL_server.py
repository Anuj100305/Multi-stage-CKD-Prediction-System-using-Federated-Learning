import flwr as fl
from flwr.server.strategy import FedAvg

# Force server to wait for all clients
strategy = FedAvg(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)