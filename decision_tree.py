import numpy as np
import pandas as pd
import math
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self):
        self.tree = Node()
    def __str__(self):
        return "Decision tree using impurity measure= " + str(self.impurity_measure) + ' and with pruning' if self.pruning else ' without pruning'

    """
    Fits the decision tree to data provided
    Arguments:
        X: A pandas dataframe containing all features in a dataset
        y: A pandas series containing all labels in a dataset
        impurity_measure: A String. Can either be entropy or gini
        pruning: Boolean. If true: reduced error pruning will be performed
    Returns:
        Nothing
    """
    def learn(self, X, y, impurity_measure='entropy', pruning=False):
        self.impurity_measure = impurity_measure
        self.pruning = pruning

        if self.pruning:
            #Split into training and pruning data
            X_train, X_prune, y_train, y_prune = train_test_split(X, y, test_size=0.25, random_state=42)
            #Fit the decision tree to the training data
            create_tree(X_train, y_train, impurity_measure, self.tree)
            #Perform reduced error pruning
            prune(X_prune, y_prune, self.tree)

        else:
            #Fit the decision tree to all data given to learn()
            create_tree(X, y, impurity_measure, self.tree)

    """
    Predicts label for all rows x in a matrix X
    Arguments:
        X: A pandas dataframe containing all features in a dataset
    Returns:
        List containing all the predicted labels
    """
    def predict(self, X):
        #Stores all predictions
        predictions = []
        #For each row in x; traverse the tree to get the predicted label and append it to the list
        for i in range(len(X)):
            x = X.iloc[i, :]
            predictions.append(get_prediction_label(x, self.tree))
        return predictions

"""
Predicts label for a given row x
Arguments:
    x: A pandas series containing a row in a dataset
    node: a Node (the current node). This node is used to perform the method recursively
Returns:
    String - Predicted label
"""
def get_prediction_label(x, node):
    # Return label when a leaf is reached
    if node.is_leaf():
        return node.label
    # Otherwise continue traversing the tree (recursively)
    elif x[node.data.split_index] < node.data.split_value:
        return get_prediction_label(x, node.left)
    else:
        return get_prediction_label(x, node.right)

"""
Function to perform reduced error pruning
Arguments:
    X: A pandas dataframe containing the features in a dataset
    y: A pandas series containing the labels in a dataset
    node: a Node (the current node). This node is used to perform the method recursively
Returns:
    int - Label error. This result is used in the function callbacks
"""
def prune(X, y, node):
    if node.is_leaf():
        #Returns amount of label errors
        return len(y) - y.tolist().count(node.label)

    # If no datapoints below/above split return 0 label errors
    if X.empty:
        return 0

    dataset = pd.concat([X, y], axis=1)

    #Split the dataset
    above_split, below_split = split_dataset(dataset, node.data.split_index, node.data.split_value)

    #Extract X and y in above and below
    X_below = below_split.iloc[:, :-1]
    y_below = below_split.iloc[:, -1]
    X_above = above_split.iloc[:, :-1]
    y_above = above_split.iloc[:, -1]


    label_errors_left_node = prune(X_below, y_below, node.left)
    label_errors_right_node = prune(X_above, y_above, node.right)
    label_errors_majority_label = len(y) - y.tolist().count(node.data.majority_label)

    #Cut off subtree if we get fewer or the same amount of errors using the majority label
    if label_errors_majority_label <= label_errors_left_node + label_errors_right_node:
        node.label = node.data.majority_label
        node.left = None
        node.right = None
        return label_errors_majority_label
    return label_errors_left_node + label_errors_right_node

'''
Fits the decision tree to data provided
Arguments:
    X: A pandas dataframe containing all features in a dataset
    y: A pandas series containing all labels in a dataset
    impurity_measure: A String. Can either be entropy or gini
    node: a Node (the current node). This node is used to perform the method recursively
Returns:
    Nothing
'''
def create_tree(X, y, impurity_measure, node):
    df = pd.concat([X, y], axis=1)

    #If all labels in y are equal: assign that label to the node
    unique_labels_in_y = set(y)
    if len(unique_labels_in_y) == 1:
        node.label = y.iloc[0]
        return
    #If all feature values in X are identical: assign the majority label to the node
    elif has_identical_feature_values(X):
        node.label = get_majority_label(df)
        return
    else:
        #Find out wich feature gives the highest information gain
        split_info = get_feature_with_highest_information_gain(df, impurity_measure)

        #Assign the optimal split value and split index to the current node | Also set the majority label as we need this later for pruning
        node.data = Data(split_info['split_value'], split_info['split_index'], get_majority_label(df))
        node.left = Node()
        node.right = Node()

        X_below = split_info['below_split'].iloc[:, :-1]
        y_below = split_info['below_split'].iloc[:, -1]

        X_above = split_info['above_split'].iloc[:, :-1]
        y_above = split_info['above_split'].iloc[:, -1]

        #Recursively continue to the left and right
        create_tree(X_below, y_below, impurity_measure, node.left)
        create_tree(X_above, y_above, impurity_measure, node.right)

'''
Calculates impurity
Arguments:
    data: a pandas dataframe containing both features and labels
    impurity_measure: A String. Can either be entropy or gini
Returns:
    float: impurity
'''
def calculate_impurity(data, impurity_measure):
    y = data.iloc[:, -1]
    _, labels = np.unique(y, return_counts=True)
    prob_current_label = labels / np.sum(labels)

    if impurity_measure == 'entropy':
        return -1 * np.sum(prob_current_label * np.log2(prob_current_label))

    if impurity_measure == 'gini':
        return 1 - np.sum(prob_current_label ** 2)

'''
Splits a dataset based on the value of a given index and a given value
Argements:
    data: a pandas dataframe containing both features and labels
    column_index: int - The index of the column index to split on
    split_value: float - The value to split on
Returns:
    above_split: a pandas dataframe containing the rows above the split
    below_split: a pandas dataframe containing the rows below the split
'''
def split_dataset(data, column_index, split_value):
    above_split = data.loc[data.iloc[:, column_index] >= split_value]
    below_split = data.loc[data.iloc[:, column_index] < split_value]
    return above_split, below_split

'''
Calculates the information gain from a feature
Arguments:
    data: a pandas dataframe containing both features and labels
    column_index: int - The index of the column index to split on
    split: float - The value to split on
    impurity_measure: A String. Can either be entropy or gini
Returns:
    split_info: A dictionary - 
        "information_gain": information_gain, float
        "split_value": split_value, float
        "split_index": column_index, int
        "above_split": above_split, pandas dataframe
        "below_split": below_split pandas dataframe
'''
def calculate_information_gain_of_feature(data, column_index, split, impurity_measure):
    split_value = 0
    if split == 'mean':
        split_value = data.iloc[:, column_index].mean()
    elif split == 'median':
        split_value = data.iloc[:, column_index].median()
    else:
        raise Exception('Split mode not recognized')

    above_split, below_split = split_dataset(data, column_index, split_value)

    impurity_above_split = calculate_impurity(above_split, impurity_measure=impurity_measure)
    impurity_below_split = calculate_impurity(below_split, impurity_measure=impurity_measure)

    information = len(above_split) / len(data) * impurity_above_split + len(below_split) / len(
        data) * impurity_below_split

    information_gain = calculate_impurity(data, impurity_measure=impurity_measure) - information

    split_info = {
        "information_gain": information_gain,
        "split_value": split_value,
        "split_index": column_index,
        "above_split": above_split,
        "below_split": below_split
    }

    return split_info

'''
Finds which feature gives the highest information gain as well as the features index and split value
Arguments:
    data: a pandas dataframe containing both features and labels
    impurity_measure: A String. Can either be entropy or gini
Returns:
    features_with_highest_information_gain: A dictionary - 
        "information_gain": information_gain, float
        "split_value": split_value, float
        "split_index": column_index, int
        "above_split": above_split, pandas dataframe
        "below_split": below_split pandas dataframe
'''
def get_feature_with_highest_information_gain(data, impurity_measure, split='mean'):
    information_gains = []
    for i in range(data.shape[1] - 1):
        information_gains.append(
            calculate_information_gain_of_feature(data, i, split=split, impurity_measure=impurity_measure))

    feauture_with_highest_information_gain = information_gains[0]
    for i in range(1, len(information_gains)):
        if information_gains[i]["information_gain"] > feauture_with_highest_information_gain["information_gain"]:
            feauture_with_highest_information_gain = information_gains[i]

    return feauture_with_highest_information_gain

'''
Checks if all the rows in a dataframe are equal
Arguments:
    X: a pandas dataframe containing the features of a dataset
Returns:
    Boolean: True if all rows in the dataframe are equal; False otherwise
'''
def has_identical_feature_values(X):
    # Finds firs row
    first = X.iloc[0, :]

    # Creates a new boolean dataframe based on which rows in X are equal to the first row
    df = X == first

    # Returns true if all values in df are true; False otherwise
    return df.all().all()

'''
Finds the majority label in a dataset
Arguments:
    data: a pandas dataframe
Returns:
    String: the majority label
'''
def get_majority_label(data):
    # Get counts for each label
    value_counts = data.iloc[:, -1].value_counts()

    # Sort in descending order and return the largest count
    return value_counts.sort_values(ascending=False).keys()[0]


class Data:
    def __init__(self, split_value, split_index, majority_label):
        self.split_value = split_value
        self.split_index = split_index
        self.majority_label = majority_label


class Node:
    def __init__(self, label=None, data=None):
        self.label = label
        self.data = data
        self.left = None
        self.right = None

    '''
    Checks if this node is a leaf
    Returns:
        Boolean: True if node is leaf; False otherwise
    '''
    def is_leaf(self):
        if self.left is None and self.right is None:
            return True
        return False

    #Used for debugging
    def __str__(self):
        if self.is_leaf():
            return "Leaf node with label " + str(self.label)
        else:
            return 'Split index ' + str(self.data.split_index) + '\nSplit value ' + str(
                self.data.split_value) + '\nMajority label ' + str(self.data.majority_label)


# Reading the data
data = pd.read_csv('magic04.data', header=None)

# Splitting data into X and y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Splitting data into X_train, y_train, X_val, y_val, X_test, y_test
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Training
models = []

decision_tree_entropy = DecisionTree()
decision_tree_entropy.learn(X_train, y_train)
models.append(decision_tree_entropy)

decision_tree_entropy_pruning = DecisionTree()
decision_tree_entropy_pruning.learn(X_train, y_train, pruning=True)
models.append(decision_tree_entropy_pruning)

decision_tree_gini = DecisionTree()
decision_tree_gini.learn(X_train, y_train, impurity_measure='gini')
models.append(decision_tree_gini)

decision_tree_gini_pruning = DecisionTree()
decision_tree_gini_pruning.learn(X_train, y_train, impurity_measure='gini', pruning=True)
models.append(decision_tree_gini_pruning)


#Model selection
train_accuracies = []
val_accuracies = []

for model in models:
    print('Decision tree', model.impurity_measure, "Pruning" if model.pruning else '')
    train_acc = accuracy_score(y_train, model.predict(X_train))
    train_accuracies.append(train_acc)
    print("Training accuracy: ", train_acc)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    val_accuracies.append(val_acc)
    print("Validation accuracy: ", val_acc)
    print()

best_model_index = list.index(val_accuracies, max(val_accuracies))
best_model = models[best_model_index]
best_model_train_acc = train_accuracies[best_model_index]
best_model_val_acc = val_accuracies[best_model_index]

print("Best model:")
print(best_model)
print("Training accuracy: ", best_model_train_acc)
print("Validation accuracy: ", best_model_val_acc)
print()


#Model evaluation
print("Estimating the performance of the selected model on unseen data points")
test_pred = best_model.predict(X_test)
best_model_test_acc = accuracy_score(y_test, test_pred)
print("Best model's accuracy on test data: ", best_model_test_acc)
start = time.time()
best_model.learn(X_train, y_train)
end = time.time()
best_model_time = end - start
print("Time used to train model: ", best_model_time)
print()
print("------------------------------")


#Comparing to existing implementation
print("Comparison between existing implementation")
sk_learn_decision_tree = DecisionTreeClassifier(random_state=42, criterion=str(best_model.impurity_measure))

start = time.time()
sk_learn_decision_tree.fit(X_train, y_train)
end = time.time()
sk_learn_time = end - start

sk_learn_train_acc = accuracy_score(y_train, sk_learn_decision_tree.predict(X_train))
sk_learn_val_acc = accuracy_score(y_val, sk_learn_decision_tree.predict(X_val))
sk_learn_test_acc = accuracy_score(y_test, sk_learn_decision_tree.predict(X_test))

print("Sklearn's descision tree classifier: ")
print("Training accuracy: ", sk_learn_train_acc)
print("Validation accuracy: ", sk_learn_val_acc)
print("Test accuracy: ", sk_learn_test_acc)
print("Time used to train model: ", sk_learn_time)
print()

print("Difference between best model and Sklearn's decision tree classifier:")
print("Training accuracy: ", best_model_train_acc - sk_learn_train_acc)
print("Validation accuracy: ", best_model_val_acc - sk_learn_val_acc)
print("Test accuracy: ", best_model_test_acc - sk_learn_test_acc)
print("Time difference: ", best_model_time - sk_learn_time)

