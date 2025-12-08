import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from typing import List, Dict, Any

# PyG Imports
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.data import Dataset, Data

# Local Imports
from models import GCN, GraphSAGE, GAT
from engine import Trainer
from visualize import visualize_graph_segment

# ---------------------------------------------------------
# CONFIGURATION & SETUP
# ---------------------------------------------------------
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

dataset_store = {'Cora'  : Planetoid(root='/tmp/Cora', name='Cora'), 
                 'PubMed': Planetoid(root='/tmp/PubMed', name='PubMed'),
                #  'Flickr': Flickr(root='/tmp/Flickr')
                 }

EPOCHS = 150

def save_training_curves(results: List[Dict[str, Any]], dataset_name: str) -> None:
    """Saves Accuracy vs Epoch and Time plots to the results folder."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy vs Epochs
    for res in results:
        hist = res['history']
        ax1.plot(hist['accuracy'], label=res['model_name'])
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title(f'{dataset_name}: Accuracy vs Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy vs Time
    for res in results:
        hist = res['history']
        ax2.plot(hist['time_cumulative'], hist['accuracy'], label=res['model_name'])
        
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title(f'{dataset_name}: Accuracy vs Wall-Clock Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    save_path = os.path.join(RESULTS_DIR, f"{dataset_name}_training_curves.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Saved training curves to {save_path}")

def save_graph_visualization(data: Data, dataset_name: str) -> None:
    """Wraps the visualization function to save the output."""
    # We essentially hijack the plot by creating a figure before calling the helper
    # Note: visualize_graph_segment in visualize.py calls plt.show(), 
    # so we might need to modify visualize.py OR just run it interactively.
    # Here, we assume standard matplotlib behavior.
    
    try:
        # Pick a node with high degree for better visualization
        # center_node = torch.argmax(data.edge_index[0].bincount()).item()
        
        plt.figure(figsize=(10, 10))
        visualize_graph_segment(data, node_idx=20, hops=2)
        
        save_path = os.path.join(RESULTS_DIR, f"{dataset_name}_structure_viz.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[Plot] Saved graph visualization to {save_path}")
    except Exception as e:
        print(f"[Warning] Could not visualize graph for {dataset_name}: {e}")

# ---------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Loop through Datasets
    for dataset_name, dataset_object in dataset_store.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'='*60}")
        
        data = dataset_object[0]
        
        print(f"Stats -> Nodes: {data.num_nodes} | Edges: {data.num_edges} | Features: {dataset_object.num_node_features} | Classes: {dataset_object.num_classes}")

        # 2. Visualize Graph Segment
        print(f"Generating graph visualization for {dataset_name}...")
        save_graph_visualization(data, dataset_name)
        
        # 3. Define Models for this specific dataset
        # (Must re-init because num_features/classes change per dataset)
        models_dict = {
            "GCN": GCN(dataset_object.num_node_features, 64, dataset_object.num_classes),
            "GraphSAGE": GraphSAGE(dataset_object.num_node_features, 32, dataset_object.num_classes),
            # "GAT": GAT(dataset_object.num_node_features, 64, dataset_object.num_classes, heads=4) 
            # Note: Reduced heads/hidden dim slightly to prevent OOM on Flickr Full-Batch
        }

        dataset_results = []

        # 4. Train Models
        for model_name, model in models_dict.items():
            print(f"--- Training {model_name} on {dataset_name} ---")
            
            trainer = Trainer(model, data, lr=0.01)
            metrics = trainer.run(epochs=EPOCHS)
            
            # Store Metrics
            dataset_results.append(metrics)

        # 5. Process & Save Results for this Dataset
        save_training_curves(dataset_results, dataset_name)
        
        # print(tabulate(dataset_results))
        # # Create Table
        df = pd.DataFrame(dataset_results)
        # print(df.columns)
        df_clean = df.drop(columns=['history'])

        # # Print Table
        print(f"\nFinal Results for {dataset_name}:")
        print(tabulate(df_clean.round(4), headers='keys', tablefmt='psql', showindex=False))
        
        # # Save CSV
        csv_path = os.path.join(RESULTS_DIR, f"{dataset_name}_metrics.csv")
        df_clean.to_csv(csv_path, index=False)
        print(f"[Data] Saved metrics to {csv_path}")

    print("\n\nAll experiments completed successfully! Check the 'results' folder.")