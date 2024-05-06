# import needed libraries
import random
from sklearn.model_selection import KFold
import csv
from collections import defaultdict
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class NaiveBayesClassifier:
    def __init__(self):
        self.class_prob = {}
        self.word_prob = defaultdict(lambda: defaultdict(float))
        self.classes = set()
        self.vocabulary = set()

    def fit(self, training_data):
        total_samples = len(training_data)
        num_features = len(training_data[0]) - 4  # Number of features (excluding the last 4 attributes)
        
        # Count the occurrences of each class
        class_counts = defaultdict(int)
        for sample in training_data:
            label = sample[-1]
            class_counts[label] += 1
            self.classes.add(label)
        
        # Calculate class probabilities
        for label, count in class_counts.items():
            self.class_prob[label] = count / total_samples
        
        # Count occurrences of each word given each class
        for sample in training_data:
          label = sample[0]
          for i in range(num_features - 1):
                 if float(sample[i+1]) > 0:  # Word exists in the email
                    self.word_prob[label][i] += 1
                    self.vocabulary.add(i)

        # Laplace smoothing and calculating probabilities
        for label in self.classes:
            num_words = sum(self.word_prob[label].values()) + len(self.vocabulary)
            for word in self.vocabulary:
                self.word_prob[label][word] = (self.word_prob[label][word] + 1) / num_words

    def predict(self, sample):
        max_probability = float('-inf')
        predicted_class = None
        for label in self.classes:
            prob = self.class_prob[label]
            for i in range(len(sample) - 5): # skip last four columns and decrement to account for skipping the first index
                if float(sample[i+1]) > 0:  # Word exists in the email
                    prob *= self.word_prob[label][i]
            if prob > max_probability:
                max_probability = prob
                predicted_class = label
        return predicted_class


def model(train_data, val_data):
    clf = NaiveBayesClassifier()
    clf.fit(train_data)

    correct_predictions = 0
    total_predictions = len(val_data)
    for sample in val_data:
        predicted_class = clf.predict(sample)
        if predicted_class == sample[-1]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_similarity(email1, email2):
    # calculate dot product  
    dot_product = np.dot(email1[:-1], email2[:-1])
    magnitude_email1 = np.linalg.norm(email1[:-1])
    magnitude_email2 = np.linalg.norm(email2[:-1])

    # find the similarity between the two compared emails 
    similarity = dot_product / (magnitude_email1 * magnitude_email2)

    # return the similarity 
    return similarity

def knn_evaluate(train_data, val_data, k):
    def predict_email(email):
        similarities = [(calculate_similarity(email, train_email), train_email[-1]) for train_email in train_data]
        similarities.sort(reverse=True)
        top_k_labels = [label for (_, label) in similarities[:k]]
        predicted_label = max(set(top_k_labels), key=top_k_labels.count)
        return predicted_label

    correct_predictions = 0
    total_predictions = len(val_data)
    
    # Using parallel processing to speed up predictions
    with ProcessPoolExecutor() as executor:
        predicted_labels = list(executor.map(predict_email, val_data))

    for i, sample in enumerate(val_data):
        if predicted_labels[i] == sample[-1]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def sigmoid(val):
    return 1 / (1 + np.exp(-val))

def add_intercept(value):
    return np.c_[np.ones((len(X), 1)), value]

def LR_model_learning(X_train, y_train, X_val, y_val, epoch=100, learning_rate=0.01):
    # Step 1: Add intercept column to X_train and X_val
    X_train = add_intercept(X_train)
    X_val = add_intercept(X_val)
    
    # Step 2: Initialize weights randomly
    M = np.random.randn(X_train.shape[1], 1)
    
    best_model = M
    best_performance = 0
    
    # Step 3: Performing gradient descent on the whole dataset
    for i in range(epoch):
        pred_y = sigmoid(np.dot(X_train, M))
        loss = -np.sum(y_train * np.log(pred_y) + (1 - y_train) * np.log(1 - pred_y)) / len(y_train)
        gm = np.dot(X_train.T, (pred_y - y_train)) * 2 / len(y_train)
        
        # Update weights
        M = M - learning_rate * gm
        
        # Model evaluation on validation dataset
        val_pred_y = sigmoid(np.dot(X_val, M))
        val_pred_labels = (val_pred_y > 0.5).astype(int)
        accuracy = np.mean(val_pred_labels == y_val)
        
        # Check if the current model outperforms the best model
        if accuracy > best_performance:
            best_model = np.copy(M)
            best_performance = accuracy
    
    return best_model

def performance(model, data):
  print("result:", model)
    
def main():
    # download the data set
    filename = 'spambase.csv'
    temp = []
    with open(filename, 'r') as file:
      temp = file.readline().strip().split(',')
      data = [[]] * len(temp)
      for i in range(len(temp)):
        val = temp[i]
        data[i] = [val]
      for j in range(4600): # had to brute force cause I couldn't remember/ figure out how to check for E.O.F. in python :/
        temp = file.readline().strip().split(',')
        for i in range(len(temp)):
          data[i].append(temp[i])
    
    random.shuffle(data) # change if you are using pandas dataframe
    split_index = int(len(data) * 0.8)
    training = data[:split_index]
    test = data[split_index:]
    
    fold5 = KFold(n_splits=5)
    
    # Perform 5-fold cross-validation
    for train_idx, val_idx in fold5.split(training):
        sub_val = [training[i] for i in val_idx]
        sub_train = [training[i] for i in train_idx]
        nb_clf = model(sub_train, sub_val)  # Train the model
        performance(nb_clf, test)  # Evaluate the model on test data

main()
