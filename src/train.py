# Local training/evaluation logic

import torch
import torch.nn as nn

# Define the multi-task loss function
def multi_task_loss(pred_anxiety, pred_negaffect, targets_anxiety, targets_negaffect):
    """
    Calculates the total loss for the multi-task model.
    We use Mean Squared Error (MSE) for both regression tasks.
    """
    loss_anxiety = nn.MSELoss()(pred_anxiety, targets_anxiety)
    loss_negaffect = nn.MSELoss()(pred_negaffect, targets_negaffect)
    
    # Simple sum of losses. Can be weighted (e.g., 0.5 * loss_anxiety + ...)
    total_loss = loss_anxiety + loss_negaffect
    return total_loss, loss_anxiety, loss_negaffect

def train(model, trainloader, epochs, device):
    """Train the model on the client's local data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for windows, labels in trainloader:
            windows, labels = windows.to(device), labels.to(device)
            
            # Get the two target labels
            targets_anxiety = labels[:, 0].view(-1, 1)
            targets_negaffect = labels[:, 1].view(-1, 1)
            
            # Forward pass
            optimizer.zero_grad()
            pred_anxiety, pred_negaffect = model(windows)
            
            # Calculate loss
            loss, _, _ = multi_task_loss(
                pred_anxiety, pred_negaffect, targets_anxiety, targets_negaffect
            )
            
            loss.backward()
            optimizer.step()

def evaluate(model, testloader, device):
    """Evaluate the model on the client's local test data."""
    model.eval()
    total_loss = 0.0
    total_loss_anxiety = 0.0
    total_loss_negaffect = 0.0
    
    with torch.no_grad():
        for windows, labels in testloader:
            windows, labels = windows.to(device), labels.to(device)
            
            targets_anxiety = labels[:, 0].view(-1, 1)
            targets_negaffect = labels[:, 1].view(-1, 1)
            
            pred_anxiety, pred_negaffect = model(windows)
            
            loss, loss_anx, loss_neg = multi_task_loss(
                pred_anxiety, pred_negaffect, targets_anxiety, targets_negaffect
            )
            total_loss += loss.item()
            total_loss_anxiety += loss_anx.item()
            total_loss_negaffect += loss_neg.item()
            
    avg_loss = total_loss / len(testloader)
    avg_loss_anxiety = total_loss_anxiety / len(testloader)
    avg_loss_negaffect = total_loss_negaffect / len(testloader)
    
    return avg_loss, {"anxiety_loss": avg_loss_anxiety, "negaffect_loss": avg_loss_negaffect}