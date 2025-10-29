# Flower server logic

import flwr as fl
import src.config as config

def main():
    print(f"Starting server for {config.NUM_ROUNDS} rounds...")

    # Define strategy
    # FedProx is good for non-IID data like ours.
    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,           # Train on 100% of selected clients
        fraction_evaluate=1.0,      # Evaluate on 100% of selected clients
        min_fit_clients=2,          # Wait for 2 clients to be ready
        min_available_clients=2,    # Needs 2 clients to ever be available
        proximal_mu=0.01,           # FedProx hyperparameter
        
        # Pass client-side epochs to the client config
        on_fit_config_fn=lambda server_round: {"local_epochs": config.LOCAL_EPOCHS}
    )

    # Start the Flower server
    fl.server.start_server(
        server_address=config.SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy
    )

if __name__ == "__main__":
    main()