"""
Julia Adamczyk
Submission date: 9/23/2021
History 
9/11/21 - set up the decision tree file
9/12/21 - implemented entropy function, started information gain function 
9/14/21 - gone terrible, deleted everything
9/20/21 - started again, implemented Decision Tree algorithm as class with nodes and leafs, 
implemented entropy, information_gain, create_split
9/21/21 - implemented build tree, and rest of the functions, realized there are more specs in the file
9/22/21 - finalized the file
"""
import math
import numpy as np

class Leaf:
    def __init__(self, value, a):
        self.value = value
        self.accuracy = a

class Node:
    def __init__(self, feature=None, yes=None, no=None):
        self.feature = feature
        self.true_side = yes
        self.false_side = no

class DecisionTree:
    def __init__(self, max_nodes, f_dim):
        self.root = None
        self.max_depth = max_nodes
        self.feature_dim = f_dim
        
    def entropy(self, labels):
        # get number of elements in the set
        n_of_labels = len(labels)
        # if it's less or equal to 1 the entropy is 0
        if n_of_labels <= 1:
            return 0.0
        # get the values and the number of times they appear in the set
        _, counts = np.unique(labels, return_counts=True)
        # get the probabilities of each class
        probabiliites = counts / n_of_labels
        e = 0.0
        for prob in probabiliites:
            e -= prob * np.log2(prob)
        # return entropy
        return e

    def information_gain(self, before_split, yes, no):
        prob_yes = float(len(yes) / len(before_split))
        return self.entropy(before_split) - (prob_yes*self.entropy(yes) + (1-prob_yes)*self.entropy(no))

    def create_split(self, dataset, target_feature_index):
        yes, no = [], []
        for sample in dataset:
            if sample[target_feature_index] == 1:
                yes.append(sample)
            else:
                no.append(sample)
        return np.array(yes), np.array(no)

    def get_best_split(self, X, Y):
        best_ig = 0.0
        best_feature_index = None
        true_split = None
        false_split = None
        n_features = X.shape[1]
        # for each feature
        for f in range(n_features):
            # concatenate data in order to keep the labels with split samples
            data = np.concatenate((X, np.expand_dims(Y, axis=1)), axis=1)
            # split data
            yes, no = self.create_split(data, f)
            # if one of the splits is empty do not create a split
            if len(yes) == 0 or len(no) == 0:
                continue
            # calculate information gain and decide if it's best
            ig = self.information_gain(data[:,-1], yes[:,-1], no[:,-1])
            if ig > best_ig:
                best_ig = ig
                best_feature_index = f
                true_split = yes
                false_split = no
        """        
        print("best feature is feature ", best_feature_index)
        print("with IG = ", best_ig)
        print("true split: ", true_split)
        print("false split: ", false_split)
        """
        return best_feature_index, best_ig, true_split, false_split    

    def build_tree(self, X, Y, counter):
        # check for max_depth
        if self.max_depth < 0:
            # if it's -1 then check only for information gain and features 
            if  X.shape[0] >= 2 and not np.all(Y == Y[0]):
                # get split and information gain
                feature_index, information_gain, true_side, false_side = self.get_best_split(X, Y)
                # if information gain is greater than 0
                if information_gain > 0:
                    # recursively build left and right subtree
                    left_branch = self.build_tree(false_side[:,:-1], false_side[:,-1], counter)
                    right_branch = self.build_tree(true_side[:,:-1], true_side[:,-1], counter)
                    # print("Decision Node idx: ", feature_index)
                    return Node(feature_index, right_branch, left_branch)
            else:
                # print("Leaf Node value: ", Y)
                # calculate accuracy
                values, counts = np.unique(Y, return_counts=True)
                ind = np.argmax(counts)
                acc = max(counts) / len(Y)
                return Leaf(values[ind], acc)            
        else:
            # if the node shouldn't be a leaf
            # i.e if we did not exceed the max_depth yet, 
            # and there isn't less than two samples to split
            # and the labels are not all the same (information gain is 0)
            if counter <= self.max_depth and X.shape[0] >= 2 and not np.all(Y == Y[0]):
                # print("counter is ", counter)
                # get split and information gain
                feature_index, information_gain, true_side, false_side = self.get_best_split(X, Y)
                # if information gain is greater than 0
                if information_gain > 0:
                    # add another decision node
                    counter += 1
                    # recursively build left and right subtree
                    left_branch = self.build_tree(false_side[:,:-1], false_side[:,-1], counter)
                    right_branch = self.build_tree(true_side[:,:-1], true_side[:,-1], counter)
                    # print("Decision Node idx: ", feature_index)
                    return Node(feature_index, right_branch, left_branch)
            else:
                # print("Leaf Node value: ", Y)
                # calculate accuracy
                values, counts = np.unique(Y, return_counts=True)
                ind = np.argmax(counts)
                acc = max(counts) / len(Y)
                return Leaf(values[ind], acc)
        
    def train(self, X, Y):
        n_nodes = 1
        self.root = self.build_tree(X, Y, n_nodes)

    def print_tree(self, node):
        if isinstance(node, Leaf):
            print("value of the node = ", node.value)
            #print("accuracy in the leaf = ", node.accuracy)
            return
        if isinstance(node, Node):
            print("binary split over feature of index: ", node.feature)
            self.print_tree(node.false_side)
            self.print_tree(node.true_side)

    def calculate_acc_helper(self, node, accuracy_list):
        if isinstance(node, Leaf):
            accuracy_list.append(node.accuracy)
            return
        if isinstance(node, Node):
            self.calculate_acc_helper(node.false_side, accuracy_list)
            self.calculate_acc_helper(node.true_side, accuracy_list)

    def calculate_total_accuracy(self):
        # basically traverse the whole tree and store all leaf accuracies 
        accuracy_list = []
        self.calculate_acc_helper(self.root, accuracy_list)
        return sum(accuracy_list) / len(accuracy_list)

    def test_set(self, X, Y):
        labels = []
        for sample in X:
            label = self.predict_single(sample, self.root)
            labels.append(label)
        n_correct_predictions = 0
        for index in range(len(labels)):
            if labels[index] == Y[index]:
                n_correct_predictions += 1
        return n_correct_predictions/len(labels)

    def predict_single(self, x, node):
        if isinstance(node, Leaf):
            value = node.value
            return value
        if isinstance(node, Node):
            feature = x[node.feature]
            if feature == 0:
                # search on no side
                value = self.predict_single(x, node.false_side)
                return value
            else:
                # search on yes side
                value = self.predict_single(x, node.true_side)
                return value

def DT_train_binary(X,Y,max_depth):
    n_features = X.shape[1]
    decision_tree = DecisionTree(max_depth, n_features)
    decision_tree.train(X, Y)
    # decision_tree.print_tree(decision_tree.root)
    return decision_tree

def DT_test_binary(X,Y,DT):
    accuracy = DT.test_set(X,Y)
    return accuracy

def DT_make_prediction(x,DT):
    if x.shape[0] != DT.feature_dim:
        print("Dimensionality of the provided sample is not compliant with the DecisionTree dimension. Cannot make a prediction")
        return None
    else:
        return DT.predict_single(x, DT.root)
