# src/utils/iterative.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from src.rrt.rrt_utils import TreeNode
from src.point_source.point_source import PointSourceField

def plot_current_tree(tree_nodes: List[TreeNode], current_node: TreeNode, chosen_branch: List, scenario: PointSourceField):
    """Plot the current tree, the current node, and the chosen branch."""
    fig, ax = plt.subplots()

    # Plot all nodes in the tree
    for node in tree_nodes:
        if node.parent:
            ax.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], 'b-')
    
    # Highlight the current node
    if current_node:
        ax.plot(current_node.point[0], current_node.point[1], 'go', markersize=10, label="Current Node")
    
    # Highlight the chosen branch
    if chosen_branch:
        chosen_branch = np.array(chosen_branch)
        ax.plot(chosen_branch[:, 0], chosen_branch[:, 1], 'r-', linewidth=2, label="Chosen Branch")
    
    # Plot sources
    for source in scenario.sources:
        ax.plot(source[0], source[1], 'ro', markersize=10, label="Source")
    
    ax.legend()
    plt.show()


def plot_best_estimate(sources, estimates, scenario):
    """Plots the best estimates of sources along with actual sources."""
    fig, ax = plt.subplots()
    ax.set_xlim([0, scenario.workspace_size[0]])
    ax.set_ylim([0, scenario.workspace_size[1]])
    
    # Plot actual sources
    for source in sources:
        ax.add_patch(Circle((source[0], source[1]), 0.5, color='red', label='Actual Source', alpha=0.7))
    
    # Plot estimated sources
    for estimate in estimates:
        ax.add_patch(Circle((estimate[0], estimate[1]), 0.5, color='green', label='Estimated Source', fill=False, alpha=0.7))
    
    plt.legend()
    plt.title('Best Estimate of Sources')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
