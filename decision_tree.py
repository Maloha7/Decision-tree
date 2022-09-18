import numpy as np
import pandas as pd
import math

class DecisionTree:
    def __init__(self):
        self.root = Node()

    def learn(self, X, y, impurity_measure='entropy'):
        unique_labels_in_y = set(y)
        df = pd.concat([X, y], axis=1)

        if len(unique_labels_in_y) == 1:
            return Node(y[0])
        elif has_identical_feature_values(X):
            majority_label = get_majority_label(df)
            return Node(majority_label)

        else:
            obj = get_feature_with_highest_information_gain(df)


            # self.root.add_left(self.root, obj["below_split"], obj["split_value"], obj["split_index"])
            # self.root.add_left(self.root, obj["above_split"], obj["split_value"], obj["split_index"])

    def predict(x):
        ...


class Node:
    def __init__(self, parent_node=None, data=None, split_value=None, split_index=None):
        self.parent_node = parent_node
        self.data = data
        self.split_value = split_value
        self.split_index = split_index
        self.left = None
        self.right = None

    def is_leaf(self):
        if self.left is None and self.right is None:
            return True
        return False

    def add_left(self, node):
        if self.left is None:
            self.left = node
        else:
            raise Exception('Left node already has a child')

    def add_right(self, node):
        if self.right is None:
            self.right = node
        else:
            raise Exception('Right node already has a child')





def calculate_impurity(data, impurity_measure):
    labels = data.iloc[:, -1].unique()

    total_rows = len(data)
    total_entropy = 0
    for label in labels:
        number_of_rows_with_current_label = len(data.loc[data.iloc[:, -1] == label])
        prob_current_label = number_of_rows_with_current_label/total_rows

        if impurity_measure == "entropy":
            entropy_of_current_label = - prob_current_label * math.log2(prob_current_label)
            total_entropy += entropy_of_current_label

        if impurity_measure == "gini":
            entropy_of_current_label = 1 - (prob_current_label**2)
            total_entropy += entropy_of_current_label


    return total_entropy

def calculate_information_gain_of_feature(data, column_index, split='mean'):
    split_value = 0
    if split == 'mean':
        split_value = data.iloc[:, column_index].mean()
    elif split == 'median':
        split_value = data.iloc[:, column_index].median()
    else:
        raise Exception('Split mode not recognized')

    above_split = data.loc[data.iloc[:, column_index] >= split_value]
    below_split = data.loc[data.iloc[:, column_index] < split_value]

    entropy_above_split = calculate_impurity(above_split, "entropy")
    entropy_below_split = calculate_impurity(below_split, "entropy")

    information = len(above_split)/len(data) * entropy_above_split + len(below_split)/len(data) * entropy_below_split

    information_gain = calculate_impurity(data, "entropy") - information

    obj = {
        "information_gain": information_gain,
        "split_value": split_value,
        "split_index": column_index,
        "above_split": above_split,
        "below_split": below_split
    }

    return obj

def get_feature_with_highest_information_gain(data, split='mean'):
    information_gains = []
    for i in range(data.shape[1] - 1):
        information_gains.append(calculate_information_gain_of_feature(data, i))

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


