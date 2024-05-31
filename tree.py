import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = None

    def is_leaf(self):
        # Tells that this is a leaf node since only leaf nodes have non-null values
        return self.value is not None


class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, num_features=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    def fit(self, x, y):
        self.num_features = x.shape[1] if not self.num_features else min(
            x.shape[1], self.num_features)
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x, y, depth=0):
        num_samples, n_feats = x.shape
        num_labels = len(np.unique(y))
        # check stopping criteria
        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        features_idx = np.random.choice(
            n_feats, self.num_features, replace=False)

        # find best split
        best_feature, best_threshold = self._best_split(x, y, features_idx)

        # create child node
        left_idxs, right_idxs = self._split(x[:, best_feature], best_threshold)
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, x, y, feats_idx):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx, in feats_idx:
            x_col = x[:, feat_idx]
            thresholds = np.unique(x_col)
            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, x_col, thr)

                # if the best gain is lower than current gain, we reassign the new value to best_gain
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, x_col, thr):
        # Information gained is the entropy of the parent - [weighted avg]*entropy of children

        # parent entropy
        parent_entropy = self._entropy(y)
        # create children
        left_idx, right_idx = self._split(x_col, thr)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        # calculate weighted entropy of children
        n = len(y)
        n_left, n_right = len(left_idx), len(right_idx)
        entropy_left, entropy_right = self.entropy(
            y[left_idx]), self.entropy(y[right_idx])
        child_entropy = ((n_left/n) * entropy_left)+((n_right/n)*entropy_right)

        # calc information gained
        information_gain = parent_entropy - child_entropy
        return information_gain

    # creates children

    def _split(self, x_col, split_thresh):
        left_idxs = np.argwhere(x_col <= split_thresh).flatten()
        right_idxs = np.argwhere(x_col >= split_thresh).flatten()
        return left_idxs, right_idxs

    # calculates entropy of given node
    def _entropy(self, y):
        hist = np.bincount(y)
        # calculate ps that represents prob of all values that occur within hist
        ps = hist / len(y)
        # calculate entropy for all values in ps
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, x):
        return np.array([self._traverse_tree(val, self.root) for val in x])
