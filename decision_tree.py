import numpy as np
import pandas as pd
import math
import time
from sklearn.model_selection import train_test_split



class DecisionTree:
    def __init__(self):
        self.tree = Node()

    def learn(self, X, y, impurity_measure='entropy', pruning=False):
        if pruning:
            X_train, X_prune, y_train, y_prune = train_test_split(X, y, test_size=0.25, random_state=42)
            build_tree(X_train, y_train, impurity_measure, self.tree)

        else:
            build_tree(X, y, impurity_measure, self.tree)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(get_prediction_label(x, self.tree))
        return predictions



class Data:
    def __init__(self, split_value, split_index):
        self.split_value = split_value
        self.split_index = split_index


class Node:
    def __init__(self, label=None, data=None):
        self.label = label
        self.data = data
        self.left = None
        self.right = None

    def is_leaf(self):
        if self.left is None and self.right is None:
            return True
        return False


def get_prediction_label(x, node):
    if node.is_leaf():
        return node.label
    elif x[node.data.split_index] < node.data.split_value:
        return get_prediction_label(x, node.left)
    else:
        return get_prediction_label(x, node.right)



def prune(X, y, node):
    if node.is_leaf():
        ...



def build_tree(X, y, impurity_measure, node):
    unique_labels_in_y = set(y)
    df = pd.concat([X, y], axis=1)

    if len(unique_labels_in_y) == 1:
        node.label = y.iloc[0]
        return
    elif has_identical_feature_values(X):
        node.label = get_majority_label(df)
        return
    else:
        split_info = get_feature_with_highest_information_gain(df, impurity_measure)

        node.data = Data(split_info['split_value'], split_info['split_index'])
        node.left = Node()
        node.right = Node()


        X_below = split_info['below_split'].iloc[:, :-1]
        y_below = split_info['below_split'].iloc[:, -1]

        X_above = split_info['above_split'].iloc[:, :-1]
        y_above = split_info['above_split'].iloc[:, -1]

        build_tree(X_below, y_below, impurity_measure, node.left)
        build_tree(X_above, y_above, impurity_measure, node.right)



def calculate_impurity(data, impurity_measure):
    labels = data.iloc[:, -1].unique()

    total_rows = len(data)
    total_entropy = 0
    total_gini = 0
    for label in labels:
        number_of_rows_with_current_label = len(data.loc[data.iloc[:, -1] == label])
        prob_current_label = number_of_rows_with_current_label / total_rows

        if impurity_measure == "entropy":
            entropy_of_current_label = - prob_current_label * math.log2(prob_current_label)
            total_entropy += entropy_of_current_label

        if impurity_measure == "gini":
            gini_of_current_label = 1 - (prob_current_label ** 2)
            total_gini += gini_of_current_label

    return total_entropy


def calculate_information_gain_of_feature(data, column_index, split, impurity_measure):
    split_value = 0
    if split == 'mean':
        split_value = data.iloc[:, column_index].mean()
    elif split == 'median':
        split_value = data.iloc[:, column_index].median()
    else:
        raise Exception('Split mode not recognized')

    above_split = data.loc[data.iloc[:, column_index] >= split_value]
    below_split = data.loc[data.iloc[:, column_index] < split_value]

    impurity_above_split = calculate_impurity(above_split, impurity_measure=impurity_measure)
    impurity_below_split = calculate_impurity(below_split, impurity_measure=impurity_measure)


    information = len(above_split) / len(data) * impurity_above_split + len(below_split) / len(
        data) * impurity_below_split


    information_gain = calculate_impurity(data, "entropy") - information

    split_info = {
        "information_gain": information_gain,
        "split_value": split_value,
        "split_index": column_index,
        "above_split": above_split,
        "below_split": below_split
    }

    return split_info


def get_feature_with_highest_information_gain(data, impurity_measure, split='mean'):
    information_gains = []
    for i in range(data.shape[1] - 1):
        information_gains.append(
            calculate_information_gain_of_feature(data, i, split=split, impurity_measure=impurity_measure))

    obj_with_highest_information_gain = information_gains[0]
    for i in range(1, len(information_gains)):
        if information_gains[i]["information_gain"] > obj_with_highest_information_gain["information_gain"]:
            obj_with_highest_information_gain = information_gains[i]

    return obj_with_highest_information_gain


def has_identical_feature_values(X):
    first_row = X.iloc[0, :]
    for i in range(1, len(X)):
        current_row = X.iloc[i, :]
        for j in range(len(current_row)):
            if current_row[j] != first_row[j]:
                return False

    return True


def get_majority_label(df):
    value_counts = df.iloc[:, -1].value_counts()
    return value_counts.sort_values(ascending=False).keys()[0]


data = pd.read_csv('magic04.data', header=None)
dt = DecisionTree()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

start = time.time()
dt.learn(X, y)
end = time.time()
print('Time to train: ', (end - start))
