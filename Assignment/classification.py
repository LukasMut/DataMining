import numpy as np
import pandas as pd
import re

from collections import defaultdict
from utils import *

def accuracy_score(y_true:np.ndarray, y_hat:np.ndarray):
    array_comparison = y_true == y_hat
    return len(array_comparison[array_comparison == True]) / len(array_comparison)

def str_to_num(classes:np.ndarray):
    # we might want to convert strings to numerical labels (not necessarily for this particular task, but perhaps for another task)
    str_to_num = {label:idx for idx, label in enumerate(classes)}
    num_to_str = {idx:label for idx, label in enumerate(classes)}
    return str_to_num, num_to_str

class NaiveBayes:
    
    def __init__(self, method:str):
        methods = ['count', 'cond_proba']
        if method in methods:
            self.method = method
        else:
            raise ValueError("Method must be one of ['count', 'cond_proba']")
            
    @staticmethod
    def extract_features(sent:str, test = False):
        # normalize test sentences (i.e., remove punctuation and lower case all words) (train sents are already normalized)
        sent = re.sub('[^\w\s]', '', sent).lower() if test else sent
        # return number of words used in sentence
        return len(sent.split())
    
    @staticmethod
    def sort_dict(counts:dict): return dict(sorted(counts.items(), key=lambda kv:kv[1], reverse=True))
    
    
    def train(self, data):
        """
            Args: train dataset with classes (i.e., different study programmes),
                  and attributes to condition on (i.e., words used to describe motivation for taking this course)
            Return: nested dictionary with attributes per class (counts or conditional probabilities)
        """
        attr_per_class = defaultdict(dict)
        for _, (motivation, study_programme) in data.iterrows():
            X, y = motivation, study_programme
            f_x = self.extract_features(motivation)
            if f_x not in attr_per_class[y]:
                attr_per_class[y][f_x] = 0
            attr_per_class[y][f_x] += 1
            
            if self.method == 'count':
                # add additional key with number of examples per class (crucial to compute probabilities in predict method)
                if 'n_total' not in attr_per_class[y]:
                    attr_per_class[y]['n_total'] = 0
                attr_per_class[y]['n_total'] += 1

        if self.method == 'count':
            # sort dictionary according to number of observed attributes per label
            attr_per_class = {label: self.sort_dict(counts) for label, counts in attr_per_class.items()}
            return attr_per_class
        
        else:
            # count examples per class
            examples_per_class = {label: data[data.study_programme == label].count()[0] for label in data.study_programme.unique()}
            # compute conditional probabilities per class per attribute
            cond_probas_per_class = {label: {f_x: count/examples_per_class[label] for f_x, count in counts.items()} 
                                     for label, counts in attr_per_class.items()}
            return cond_probas_per_class
    
    def compute_posteriors(self, attr_per_class:dict, f_x:int):
        """
            Args: nested dict with attributes per class (either counts of n_words used or conditional probas),
                  number of words in test sentence (i.e., f_x)
            Return: maximum likelihood estimates per class
        """
        # compute maximum likelihood estimates
        mle = dict()
        for label in attr_per_class:
            # initialise to 0
            mle[label] = 0
            if f_x in attr_per_class[label]:
                if self.method == 'cond_proba':
                    mle[label] = attr_per_class[label][f_x]
                else:
                    # compute conditional probability per class per attribute
                    mle[label] = attr_per_class[label][f_x] / attr_per_class[label]['n_total']
        return mle
    
    def min_dist(self, attr_per_class:dict, f_x:int):
        """
            Args: nested dict with attributes per class, number of words in test sentence (i.e., f_x)
            Return: closest class according to distribution of number of words used in train sentences (minimized distance)
        """
        mean_wordlen_per_class = {}
        for label, distribution in attr_per_class.items():
            labels_per_class = []
            for n_words, freq in distribution.items():
                if n_words != 'n_total':
                    for _ in range(freq):
                        labels_per_class.append(n_words)
            # compute weighted average (more weight is assigned to n_words that appeared more frequently in train set)
            mean_wordlen_per_class[label] = np.mean(labels_per_class)
        distances = [abs(f_x - mean_wordlen) for mean_wordlen in mean_wordlen_per_class.values()]
        min_dist = np.argmin(distances)
        return list(mean_wordlen_per_class.keys())[min_dist]
        
    def argmax(self, mle:dict):
        """
            Args: dictionary with maximum likelihood estimates per class
            Return: most probable class given maximum likelihood estimates
        """
        max_proba = np.argmax(list(mle.values()))
        return list(mle.keys())[max_proba]
    
    def predict_proba(self, attr_per_class:dict, sentence:str):
        """
            Args: nested dict with attributes per class (either counts of n_words used or conditional probas), test sentence
            Return: most probable class given test sentence
        """
        f_x = self.extract_features(sentence, test=True)
        # compute max likelihood estimates for all classes in dataset
        mle = self.compute_posteriors(attr_per_class, f_x)
        # if f_x (n_words in test sentence) was never observed in any class in train dataset D, 
        # compute mean word length per class and return closest class (i.e., argmin)
        if len(set(mle.values())) == 1 and next(iter(set(mle.values()))) == 0:
            return self.min_dist(attr_per_class, f_x)
        else:
            # compute argmax, and thus maximize posterior probability to predict the most likely class for a given sentence
            most_prob_class = self.argmax(mle)
            return most_prob_class