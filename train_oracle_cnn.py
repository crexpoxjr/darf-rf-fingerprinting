#!/usr/bin/env python3
"""
Complete Example: Train CNN on Synthetic RF Data

This script demonstrates the full pipeline:
1. Generate synthetic RF dataset
2. Create CNN model
3. Train and evaluate
4. Save results
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.toy_generator import create_dataset
from src.datasets.synthetic_dataset import RFDataset
from src.models.cnn1d import RF_CNN
from src.evaluation.metrics import calculate_metrics


class RFCNNTrainer:
    """Training wrapper for RF fingerprinting CNN."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device = None,
        lr: float = 0.001,
        output_dir: Path = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            test_loader: Testing DataLoader
            device: GPU/CPU device
            lr: Learning rate
            output_dir: Directory to save results
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = Path(output_dir) if output_dir else Path("./results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizer and loss with L2 regularization
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            
            # Forward pass
            pred = self.model(x)
            loss = self.criterion(pred, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"    Batch {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> tuple:
        """
        Evaluate model.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in data_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            
            pred = self.model(x)
            loss = self.criterion(pred, y)
            
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(pred, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = np.mean(all_preds == all_labels)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def fit(self, epochs: int = 20):
        """
        Train model for multiple epochs.
        
        Args:
            epochs: Number of epochs
        """
        print(f"\n{'='*70}")
        print(f"Training on {self.device}")
        print(f"{'='*70}")
        
        for epoch in range(1, epochs + 1):
            print(f"\n[Epoch {epoch}/{epochs}]")
            
            # Train
            train_loss = self.train_epoch()
            train_loss_eval, train_acc, _, _ = self.evaluate(self.train_loader)
            
            # Evaluate
            test_loss, test_acc, test_preds, test_labels = self.evaluate(
                self.test_loader
            )
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            
            print(f"  Train Loss: {train_loss_eval:.4f}, Acc: {train_acc:.4f}")
            print(f"  Test Loss:  {test_loss:.4f}, Acc: {test_acc:.4f}")
        
        print(f"\n{'='*70}")
        print("Training Complete")
        print(f"{'='*70}")
        
        # Evaluate on test set
        test_loss, test_acc, test_preds, test_labels = self.evaluate(
            self.test_loader
        )
        
        print(f"\nFinal Test Results:")
        print(f"  Accuracy: {test_acc:.4f}")
        
        # Calculate metrics
        metrics = calculate_metrics(test_labels, test_preds)
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Precision: {np.mean(metrics['precision']):.4f}")
        print(f"  Recall: {np.mean(metrics['recall']):.4f}")
        
        return metrics
    
    def save_results(self, metrics: dict, dataset_info: dict):
        """Save training results."""
        results = {
            "history": self.history,
            "final_metrics": metrics,
            "dataset_info": dataset_info
        }
        
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
        
        # Save model
        model_file = self.output_dir / "rf_cnn_oracle.pt"
        torch.save(self.model.state_dict(), model_file)
        print(f"✓ Model saved to {model_file}")


def main():
    """Main training pipeline."""
    print(f"\n{'='*70}")
    print("Synthetic RF Fingerprinting - CNN Training")
    print(f"{'='*70}")
    
    # Generate synthetic dataset
    print(f"\n[1/4] Generating synthetic dataset...")
    try:
        X, y = create_dataset()
        print(f"  ✓ Dataset generated")
        print(f"  X shape: {X.shape}")
        print(f"  Classes: {np.unique(y)}")
        print(f"  Class counts: {np.bincount(y)}")
        
        # Create PyTorch dataset
        dataset = RFDataset(X, y)
        
        # Split into train/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        metadata = {
            "num_samples": len(dataset),
            "num_train": train_size,
            "num_test": test_size,
            "num_classes": len(np.unique(y)),
            "input_shape": X[0].shape
        }
        
        print(f"  ✓ Dataset split")
        print(f"  Train samples: {metadata['num_train']}")
        print(f"  Test samples: {metadata['num_test']}")
        print(f"  Classes: {metadata['num_classes']}")
        print(f"  Input shape: {metadata['input_shape']}")
    except Exception as e:
        print(f"  ✗ Failed to generate dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create model
    print(f"\n[2/4] Creating model...")
    model = RF_CNN()
    print(f"  ✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    print(f"\n[3/4] Setting up training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = RFCNNTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        lr=0.001,
        output_dir=Path(__file__).parent / "results" / "oracle_cnn"
    )
    print(f"  ✓ Trainer ready")
    print(f"  Device: {device}")
    
    # Train
    print(f"\n[4/4] Training model...")
    metrics = trainer.fit(epochs=5)
    
    # Save results
    trainer.save_results(metrics, metadata)
    
    print(f"\n{'='*70}")
    print("✓ Pipeline Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
