import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self):
        ...

    def learn(X, y, imurity_measure='entropy'):
        unique_labels_in_y = set(y)
        df =

        if len(unique_labels_in_y) == 1:
            return Node(y[0])
        else:


    def predict(x):
        ...


class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def add_node(self, data):
        if self.left is None:
            self.left = data

        elif self.left is None:
            self.right = data

        else:
            raise Exception('Node already has 2 children')
