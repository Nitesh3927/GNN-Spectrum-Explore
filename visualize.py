import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph

# Helper needed for the k_hop_subgraph import inside the function context
import torch_geometric.utils

import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def visualize_graph_segment(data: Data, node_idx: int = 0, hops: int = 2) -> None:
    """
    Visualizes a k-hop subgraph centered around a specific node.
    This makes the visualization readable compared to plotting the whole dataset.
    
    Args:
        data (Data): The PyG dataset object.
        node_idx (int): The central node to visualize.
        hops (int): Depth of neighbors to include.
    """
    # 1. Get k-hop subgraph
    subset, edge_index, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(
        node_idx, hops, data.edge_index, relabel_nodes=True
    )
    
    # 2. Convert to NetworkX
    # Create a small data object for conversion
    sub_data = Data(edge_index=edge_index, num_nodes=subset.size(0))
    g = to_networkx(sub_data, to_undirected=True)

    # 3. Plot
    # plt.figure(figsize=(10, 10))
    plt.title(f"Visualization: {hops}-Hop Neighborhood of Node {node_idx}")
    
    # Color the central node differently
    node_colors = []
    for node in g.nodes():
        if node == mapping:  # The central node became 'mapping' after relabeling
            node_colors.append('red') # Center
        else:
            node_colors.append('skyblue') # Neighbors

    nx.draw(g, 
            node_color=node_colors, 
            with_labels=True, 
            node_size=800, 
            font_size=10, 
            edge_color="gray")
    # plt.show()
    

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """
    Plots a confusion matrix using Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {title}')
    plt.show()  


def plot_training_curves(results: list[dict]):
    """
    Plots Accuracy vs Epoch and Accuracy vs Time for multiple models.
    
    Args:
        results: List of dicts containing 'Model' name and 'History' dict.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy vs Epochs (Convergence Speed in Iterations)
    for res in results:
        history = res['history']
        epochs = range(len(history['accuracy']))
        ax1.plot(epochs, history['accuracy'], label=res['Model'])
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Learning Efficiency (Per Epoch)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy vs Wall-Clock Time (Real World Speed)
    for res in results:
        history = res['history']
        ax2.plot(history['time_cumulative'], history['accuracy'], label=res['Model'])
        
    ax2.set_xlabel('Training Time (seconds)')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Computational Efficiency (Per Second)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
