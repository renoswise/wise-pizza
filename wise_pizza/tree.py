from typing import Optional, List, Union
from enum import Enum

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.tree import DecisionTreeRegressor


class SplitCriterion(Enum):
    CATEGORICAL = 'categorical'
    CONTINUOUS = 'continuous'


class ContinuousSide(Enum):
    GREATER_OR_EQUAL = 'greater_or_equal'
    LESS = 'less'


class SplitCategories:
    def __init__(self, categories: List[str]):
        self.categories = categories


class SplitContinuous:
    def __init__(self, split_value: float, side: ContinuousSide):
        self.split_value = split_value
        self.side = side


class Node:
    def __init__(self, feature_index, threshold, left, right, impurity, is_leaf, level, path_features, path_thresholds,
                 moves):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.impurity = impurity
        self.is_leaf = is_leaf
        self.level = level
        self.path_features = path_features
        self.path_thresholds = path_thresholds
        self.moves = moves

    def __repr__(self):
        path = ', '.join(f"{idx} <= {thr}" for idx, thr in zip(self.path_features, self.path_thresholds))
        return (f"Node(level={self.level}, path='{path}', "
                f"Is leaf: {self.is_leaf}, Impurity: {self.impurity}")


class DecisionTree:
    def __init__(self, data: pd.DataFrame, nodes: List[Node]):
        self.data = data
        self.nodes = nodes


def decision_tree_simple(
        dim_df: pd.DataFrame,
        feature_cols: [str],
        target_cols: [str],
        max_depth: int = 3,
        num_leaves: Optional[int] = None,
        min_impurity_decrease: float = 0.0

):
    """
    Build a simple decision tree to predict totals from weights.
    @param dim_df: dataset to be used for training
    @param feature_cols: columns considered as features
    @param target_cols: columns considered as target
    @param max_depth: maximum depth of the tree
    @param num_leaves: maximum number of leaves in the tree
    @param min_impurity_decrease: minimum impurity decrease required for a split
    @return:
    """
    regressor = DecisionTreeRegressor(random_state=0, max_depth=max_depth, max_leaf_nodes=num_leaves,
                                      min_impurity_decrease=min_impurity_decrease)

    y = dim_df[target_cols]
    X_encoded, encoding_mapping = target_encode(dim_df, feature_cols, target_cols)

    regressor.fit(X_encoded, y)

    # separate categorical and continuous features in two lists
    categorical_features = [feature for feature in feature_cols if dim_df[feature].dtype == 'object']
    continuous_features = [feature for feature in feature_cols if feature not in categorical_features]

    nodes = extract_nodes(regressor)

    print('pause')

    return


def extract_nodes(decision_tree):
    tree = decision_tree.tree_
    nodes = {}

    def recurse(node_id, level=0, path_features=[], path_thresholds=[], moves=[]):
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        is_leaf = tree.children_left[node_id] == tree.children_right[node_id] == -1

        node = Node(
            feature_index=feature_idx,
            threshold=threshold,
            left=None,
            right=None,
            impurity=tree.impurity[node_id],
            is_leaf=is_leaf,
            level=level,
            path_features=path_features.copy(),
            path_thresholds=path_thresholds.copy(),
            moves=moves.copy()
        )
        nodes[node_id] = node

        if not is_leaf:
            next_path_features = path_features.copy()
            next_path_thresholds = path_thresholds.copy()
            next_moves = moves.copy()

            next_path_features.append(feature_idx)
            next_path_thresholds.append(threshold)

            node.left = recurse(tree.children_left[node_id], level + 1, next_path_features, next_path_thresholds, next_moves + ['left'])
            node.right = recurse(tree.children_right[node_id], level + 1, next_path_features, next_path_thresholds, next_moves + ['right'])

        return node_id

    recurse(0)
    return nodes


def cluster_extractor(nodes, encoding_mapping):
    return


def target_encode(df: pd.DataFrame, feature_cols: [str], target_cols: [str]):
    """
    Target encode the feature columns in the dataframe using the target columns.
    @param df: dataframe to encode
    @param feature_cols: columns to encode
    @param target_cols: target columns
    @return: encoded dataframe and encoding mapping
    """
    X = df[feature_cols]
    y = df[target_cols]

    encoder = TargetEncoder(cols=feature_cols)
    X_encoded = encoder.fit_transform(X, y)

    encoding_mapping = {}
    for col in X.columns:
        encoded_col = X_encoded[col]
        original_col = X[col]
        mapping = {encoded: original for encoded, original in zip(encoded_col, original_col)}
        encoding_mapping[col] = mapping

    return X_encoded, encoding_mapping


def tree_solver(
    dim_df: pd.DataFrame,
    weights: np.ndarray,
    totals: np.ndarray,
    time_basis: np.ndarray,
    max_depth: int = 3,
    num_leaves: Optional[int] = None,
):
    """
    Build a tree to predict totals from weights and time_basis.
    Args:
        dim_df (): dataset with dimension columns (e.g. ['TYPE', 'REGION', 'FIRST_PRODUCT', 'CURRENCY'] )
        weights (): array of weights that denotes the importance of each row in the dataset (default is txn count)
        totals (): array that contains the target variable (e.g. volume)
        time_basis ():
        max_depth (): maximum depth of the tree
        num_leaves ():

    Returns:

    """
    # TODO: fill in
    # Build a tree in the following fashion:
    # 1. Start with a single node containing the whole dataset
    # 2. At each node, find the best split by looping over all dimensions, for each dimension
    # solving the problem of which values to take in the left and right subtrees,
    # by running a regression of totals/weights on time basis in both subsets separately
    # and optimizing the total squared error.
    # the best combination of (node, dimension) is the next one due to be split
    # If expanding the best node would exceed maximum depth:
    # If num_leaves is None: stop
    # If it's not, expand the best node that would not exceed max_depth, until num_leaves is reached

    return
