# Flower client logic

import flwr as fl
import torch
import argparse
import os
import src.config as config
from src.model import CnnLstmMultiTask
from src.dataset import get_dataloader
from src.train import train, evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for the WESAD Multi-Task project.
    """
    def __init__(self, cid, data_dir):
        self.cid = cid
        self.data_path = os.path.join(data_dir, self.cid, "data.npz")
        self.model = CnnLstmMultiTask().to(DEVICE)
        self.trainloader, self.valloader, self.num_examples = None, None, 0

    def _load_data(self):
        """Loads the client's local data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found for client {self.cid}")
        self.trainloader, self.valloader, self.num_examples = get_dataloader(self.data_path)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config_dict):
        print(f"[Client {self.cid}] Training...")
        if self.trainloader is None:
            self._load_data()
        
        self.set_parameters(parameters)
        
        # Get local epochs from server config
        local_epochs = config_dict.get("local_epochs", config.LOCAL_EPOCHS)
        
        train(model=self.model, trainloader=self.trainloader, epochs=local_epochs, device=DEVICE)
        
        return self.get_parameters(config={}), self.num_examples, {}

    def evaluate(self, parameters, config_dict):
        print(f"[Client {self.cid}] Evaluating...")
        if self.valloader is None:
            self._load_data()
            
        self.set_parameters(parameters)
        
        loss, metrics = evaluate(model=self.model, testloader=self.valloader, device=DEVICE)
        
        return float(loss), self.num_examples, metrics


if __name__ == "__main__":
    # This block allows us to run this script from the command line
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--cid", required=True, type=str, help="Client ID (e.g., S2, S3)"
    )
    args = parser.parse_args()

    print(f"Starting client {args.cid}...")
    fl.client.start_numpy_client(
        server_address=config.SERVER_ADDRESS,
        client=FlowerClient(cid=args.cid, data_dir=config.FED_DATA_DIR),
    )