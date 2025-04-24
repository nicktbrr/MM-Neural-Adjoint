##
# Question: 
# 1. Why is the that I calculated different than the hardcoded means from the original code?
##

from mm_neural_adjoint import Network
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def main():

    Convert to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    a,b,x = model.predict_geometry(y_tensor[0], file_name='test_predictions.csv', save_top=10)
    print(a.shape)
    print(b.shape)
    print(x.shape)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Split the dataset into train, validation, and test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    na = Network(8, 300, device=device)

    
    
    # Wrap the training with tqdm
    with tqdm(total=1, desc='Training Progress') as pbar:
        na.train(1, train_loader, val_loader, progress_bar=pbar, save=True)
    na.evaluate_geometry(test_loader)
    print(y.shape)

if __name__ == "__main__":
    main()