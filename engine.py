import time
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, Any
from tqdm import trange

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 data: Data, 
                 lr: float = 0.01, 
                 weight_decay: float = 5e-4):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.data = data.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, mask_type: str = 'test') -> Dict[str, float]:
        """
        Evaluates the model on the specified mask (val or test) and returns comprehensive metrics.
        """
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)
        
        # Select the appropriate mask
        mask = self.data.test_mask if mask_type == 'test' else self.data.val_mask
        
        # Move to CPU for sklearn metrics
        y_true = self.data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

        # Calculate Metrics
        acc = (pred[mask] == self.data.y[mask]).sum().item() / mask.sum().item()
        
        # 'macro' average calculates metrics for each label, and finds their unweighted mean.
        # This does not take label imbalance into account.
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

        return {
            "accuracy": acc,
            "f1_macro": f1,
            "precision": precision,
            "recall": recall,
            "y_true": y_true, # Returning these for Confusion Matrix plotting later
            "y_pred": y_pred
        }

    def count_parameters(self) -> int:
        """Counts the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def run(self, epochs: int = 200, eval_every: int = 1) -> Dict[str, Any]:
            """
            Runs the training loop and tracks history for plotting.
            """
            history = {
                "loss": [],
                "accuracy": [],
                "time_epoch": [],  # Time taken for specific epoch
                "time_cumulative": [] # Total time elapsed since start
            }
            
            start_global = time.time()
            cumulative_time = 0.0
            
            # Use trange so we get a tqdm object and can call set_postfix
            pbar = trange(epochs, desc="Training", unit="epoch", leave=True)
            # for epoch in range(epochs):
            for epoch in pbar:
                start_epoch = time.time()
                loss = self.train_epoch()
                end_epoch = time.time()
                
                epoch_time = end_epoch - start_epoch
                cumulative_time += epoch_time
                
                # Record metrics
                history["loss"].append(loss)
                history["time_epoch"].append(epoch_time)
                history["time_cumulative"].append(cumulative_time)
                
                # Evaluate (expensive, so maybe not every epoch for huge graphs)
                if epoch % eval_every == 0:
                    metrics = self.evaluate(mask_type='test')
                    acc = metrics["accuracy"]
                    history["accuracy"].append(metrics["accuracy"])
                else:
                    # Forward fill previous accuracy if skipping
                    history["accuracy"].append(history["accuracy"][-1] if history["accuracy"] else 0)

                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    acc=f"{acc:.4f}",
                    # ep_time=f"{epoch_time:.2f}s",
                    # cum_time=f"{cumulative_time:.1f}s"
                )

            total_time = time.time() - start_global
            
            # Get final detailed metrics
            final_metrics = self.evaluate(mask_type='test')
            
            return {
                "model_name": self.model._get_name(),
                "history": history,
                "total_time": total_time,
                "final_acc": final_metrics["accuracy"],
                "f1_macro": final_metrics["f1_macro"],
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "params": self.count_parameters()
            }