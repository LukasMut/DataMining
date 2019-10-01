import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class KMeans:
    
    def __init__(self, random_state:int, n_clusters:int, method='k-means'):
        self.random_state = random_state
        self.n_clusters = n_clusters
        methods = ['k-means', 'k-medoids']
        if method in methods:
            self.method = method
        else:
            raise ValueError("Invalid value for 'method': must be one of [k-means, k-medoids]")
            
    def init_means(self, data):
        """            
            Args: object and dataset
            Return: k randomly chosen data points that are a subset of D
            
            Intialize the means (or medians) by randomly taking k points from the dataset (both for k-Means or k-Medoids)
        """
        N = data.shape[0]
        random.seed(self.random_state) # pick a fixed random seed to make experiment reproducible 
        idx_means = [random.randint(0, N-1) for _ in range(self.n_clusters)]
        return [data[idx, :2] for idx in idx_means]
    
    def euclidean_dist(self, data, means):
        """
            Args: dataset, current mean for class one, current mean for class two
            Return: dataset with points assigned to closest mean
            
            Calculate distance between each data point and the two mean values, and find the argmin per computation;
            Update classes accordingly
        """
        # objective function: l2-norm (euclidean distance) for k-Means; l1-norm (absolut error criterion) for k-Medoids
        norm = 2 if self.method == 'k-means' else 1
        # fastest way, I am aware of, to update classes of all data points simultaneously (maybe there is a faster way? let me know :) )
        data[:, 2] = [np.argmin([np.linalg.norm(point-mean, norm) for mean in means]) for point in data[:, :2]]
        return data
    
    def update_means(self, data): 
        """
            Args: dataset
            Return: updated mean values per class
        """
        if self.method=='k-means':
            return [np.mean(data[data[:, 2] == c][:, :2], axis=0) for c in range(self.n_clusters)]
        elif self.method=='k-medoids':
            return [np.median(data[data[:, 2] == c][:, :2], axis=0) for c in range(self.n_clusters)]
    
    def fit_predict(self, data, epochs):
        """
            Args: dataset and number of epochs to train
            Return: dataset and optimized mean values for each of the two classes
        """
        means = self.init_means(data)
        #iterative relocation loop
        for epoch in range(epochs):
            # create data copy to keep track of moving points (NumPy computes changes in place - thus, we need to create a copy of D)
            data_previous = np.copy(data)
            # compute distance between each point and the two mean values, and assign classes
            data = self.euclidean_dist(data, means)
            algorithm_label = r'(k-Means)' if self.method == 'k-means' else r'(k-Medoids)'
            centroid_label = r'$\bar x' if self.method == 'k-means' else r'$\tilde x'
            plot_clusters(data, means, algorithm_label, centroid_label)
            if np.all(data[:, 2] == data_previous[:, 2]):
                 # if no data point moved this round (i.e., centroids did not change), then break loop and stop optimization
                print('0 points moved this round. Optimization will stop.')
                print()
                print('Algorithm needed {} iterations until convergence.'.format(epoch+1))
                break
            else:
                booleans = data[:, 2] == data_previous[:, 2]
                n_points_moved = len(booleans[booleans==False])
                p_string = 'points' if n_points_moved > 1 else 'point'
                print('{} {} moved this round. Optimization continues.'.format(n_points_moved, p_string))
            means = self.update_means(data)
        return data, means

def plot_clusters(data, means, algorithm_label, centroid_label):    
    plt.title('Clusters and corresponding centroids' + ' ' + algorithm_label)
    plt.scatter(data[data[:, 2] == 0][:,0], data[data[:, 2] == 0][:, 1], color = 'turquoise', label='0')
    plt.scatter(data[data[:, 2] == 1][:,0], data[data[:, 2] == 1][:, 1], color = 'cornflowerblue', label='1')
    if len(means) == 3:
        plt.scatter(data[data[:, 2] == 2][:,0], data[data[:, 2] == 2][:, 1], color = 'yellow', label='2')
    plt.scatter(means[0][0], means[0][1], color = 'red', label = centroid_label + '_0$')
    plt.scatter(means[1][0], means[1][1], color = 'darkorange', label = centroid_label + '_1$')
    if len(means) == 3:
        plt.scatter(means[2][0], means[2][1], color = 'black', label = centroid_label + '_2$')
    plt.xlabel('Shoe size (normalized)')
    plt.ylabel('Height (normalized)')
    plt.legend(fancybox=True, framealpha=1, loc='lower right', prop={'size':10})
    plt.show()