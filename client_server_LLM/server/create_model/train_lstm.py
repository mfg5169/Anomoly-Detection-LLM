import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
import boto3
import io
import sys
import logging
import torch.nn.functional as F


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.bn1(lstm_out[:, -1, :])
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)  
        return out

def load_data_from_s3(data_dir):
    """Load data directly from S3"""
    logger.info(f"Loading data from: {data_dir}")
    
    try:
        if data_dir.startswith('s3://'):
            data_dir = data_dir[5:]
        else:

            
            logger.info(f"Reading training data from: {data_dir}")
            logger.info(f"Directory contents: {os.listdir(data_dir)}")
            

            X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
            y_train = np.load(os.path.join(data_dir, 'y_train.npy'))


            X_train = torch.tensor(X_train, dtype=torch.float32) 
            y_train = torch.tensor(y_train, dtype=torch.float32)  
            
            if y_train.dtype == np.int64:
                y_train = y_train.long()

            logger.info(f"X_train shape: {X_train.shape}, Y_train shape: {y_train.shape}")
            return X_train, y_train
        

        bucket_name = data_dir.split('/')[0]
        prefix = '/'.join(data_dir.split('/')[1:])
        
        logger.info(f"Bucket: {bucket_name}")
        logger.info(f"Prefix: {prefix}")
        
        s3_client = boto3.client('s3')
        
        data_dict = {}
        for name in ['X_train', 'y_train']:
            try:
                # Create buffer
                buffer = io.BytesIO()
                
                # Download from S3
                key = f"{prefix}/{name}.npy"
                logger.info(f"Downloading {key} from bucket {bucket_name}")
                s3_client.download_fileobj(bucket_name, key, buffer)
                
                # Reset buffer position
                buffer.seek(0)
                
                # Load array
                data_dict[name] = np.load(buffer)
                logger.info(f"Loaded {name} with shape: {data_dict[name].shape}")
                
            except Exception as e:
                logger.error(f"Error loading {name}: {str(e)}")
                raise
        
        return (torch.tensor(data_dict['X_train'], dtype=torch.float32), 
                torch.tensor(data_dict['y_train'], dtype=torch.float32))
    
    except Exception as e:
        logger.error(f"Error in load_data_from_s3: {str(e)}")
        raise

def train(args):
    try:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")


        logger.info(f"Loading data from {args.data_dir}")
        X_train, y_train = load_data_from_s3(args.data_dir)
        print("TRAINING DATA: " , len(X_train))
        print(X_train)

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        
        logger.info(f"Data loaded successfully. X shape: {X_train.shape}, y shape: {y_train.shape}")


        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0
        )


        logger.info(f"Initializing model with input size: {X_train.shape[2]}")

        model = LSTMModel(
            input_size=X_train.shape[2],
            hidden_size=128,  
            num_layers=3,     
            output_size=2,
            dropout=0.2      
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2
        )

               
        y_train = F.one_hot(y_train.long(), num_classes=2).float()
        

        logger.info("Starting training...")
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()

            
                outputs = model(batch_X)

               




                loss = criterion(outputs, batch_y.long() )
                loss.backward()
                optimizer.step()
     

                predictions = torch.argmax(outputs, dim=1)
                true_labels = batch_y.long()
                accuracy = (predictions == true_labels).float().mean()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
         
            scheduler.step(avg_loss)


        logger.info(f"Saving model to {args.model_dir}")
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': args.epochs,
            'loss': avg_loss,
        }, model_path)
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    
    try:
        args = parser.parse_args()
        logger.info("Starting script with arguments:")
        logger.info(f"data_dir: {args.data_dir}")
        logger.info(f"model_dir: {args.model_dir}")
        logger.info(f"batch_size: {args.batch_size}")
        logger.info(f"epochs: {args.epochs}")
        
        train(args)
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        sys.exit(1)