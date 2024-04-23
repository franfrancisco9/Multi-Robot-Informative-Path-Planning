# RRT Utils for RRT Path Planning
import numpy as np

class TreeNode:
    """Represents a node in a tree structure."""
    def __init__(self, point, parent=None, cost=0):
        """
        Initializes a new instance of the TreeNode class.
        
        Parameters:
        - point: The (x, y) coordinates of the node.
        - parent: The parent TreeNode. Default is None.
        """
        self.point = point
        self.parent = parent
        self.children = []
        self.cost = cost

    def add_child(self, child):
        """Adds a child node to this node."""
        self.children.append(child)
        child.parent = self

class InformativeTreeNode(TreeNode):
    def __init__(self, point, parent=None):
        super().__init__(point, parent)
        self.information = 0  # Initialize the information gain attribute

class TreeCollection:
    """Represents a collection of trees."""
    def __init__(self):
        """Initializes a new instance of the TreeCollection class."""
        self.trees = []

    def add(self, tree):
        """Adds a tree to the collection."""
        self.trees.append(tree)

    def __iter__(self):
        return iter(self.trees)

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)

def node_selection_key_distance(node, target_point):
    return np.linalg.norm(node.point - target_point)

def choose_parent(X_near, x_nearest, x_new):
    c_min = cost(x_nearest) + line_cost(x_nearest, x_new)
    x_min = x_nearest

    for x_near in X_near:
        if obstacle_free(x_near.point, x_new.point) and \
                cost(x_near) + line_cost(x_near, x_new) < c_min:
            c_min = cost(x_near) + line_cost(x_near, x_new)
            x_min = x_near

    x_new.parent = x_min
    return x_new

def cost(node):
    cost = 0
    while node.parent is not None:
        cost += np.linalg.norm(node.point - node.parent.point)
        node = node.parent
    return cost

def line_cost(x1, x2):
    return np.linalg.norm(x1.point - x2.point)

def obstacle_free(x1, x2):
    # Add your actual obstacle check logic here
    return True

def rewire(X_near, x_new):
    for x_near in X_near:
        if obstacle_free(x_new.point, x_near.point) and \
                cost(x_new) + line_cost(x_new, x_near) < cost(x_near):
            x_near.parent = x_new

def near(x_new, tree_nodes, d_waypoint_distance):
    return [x for x in tree_nodes if np.linalg.norm(x_new.point - x.point) < d_waypoint_distance]

def steer(x_nearest, x_rand, d_max_step):
    direction = x_rand - x_nearest.point
    distance = np.linalg.norm(direction)
    direction = direction / distance if distance > 0 else direction
    distance = min(distance, d_max_step)
    return x_nearest.point + direction * distance

def nearest(tree_nodes, x_rand):
    return min(tree_nodes, key=lambda x: np.linalg.norm(x.point - x_rand))

def add_node(tree_nodes, x_new, x_parent):
    x_parent.add_child(x_new)
    tree_nodes.append(x_new)

def trace_path_to_root(selected_leaf):
    path = []
    node = selected_leaf
    while node is not None:
        path.append(node.point)
        node = node.parent
    path.reverse()
    return path