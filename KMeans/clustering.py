import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 


def data_normalization(df, scaling:str):
    """
        Args: pd.DataFrame, feature scaling method (see below which scaling methods are implemented)
        Return: normalized data set (converted into NumPy matrix)
    """
    scalings = ['standardize', 'min-max', 'mean-norm']
    if scaling not in scalings:
        raise ValueError('Scaling method must be one of {standardization, min-max or mean-norm scaling}')
    for column in df.columns:
        if scaling == 'standardize':
            mean, std = df[column].values.mean(), df[column].values.std()
            df[column] -= mean 
            df[column] /= std
        elif scaling == 'min-max':
            minimum, maximum = df[column].values.min(), df[column].values.max()
            df[column] -= mininmum
            df[column] /= (maximum - minimum)
        elif scaling == 'mean-norm':
            mean, maximum, minimum = df[column].values.mean(), df[column].values.max(), df[column].values.min()  
            df[column] -= mean
            df[column] /= (maximum - minimum)
    # convert pd.DataFrame into NumPy matrix to make computations (e.g., LinAlg operations) faster and Boolean indexing easier
    mat = df.to_numpy()
    # add column with zeros to the end of the matrix (at first, assign every data point to the same class)
    return np.c_[mat, np.zeros(mat.shape[0])]

class KMeans:
    
    def __init__(self, random_state:int, method='k-means'):
        self.random_state = random_state
        methods = ['k-means', 'k-medoids']
        if method in methods:
            self.method = method
        else:
            raise ValueError("Invalid value for 'method': must be one of {k-means, k-medoids}")
            
    def init_means(self, data):
        """            
            Args: object and dataset
            Return: two randomly chosen data points that are a subset of D
            
            Intialize the means (or medians) by randomly taking two points from the dataset (both for k-Means or k-Medoids)
        """
        N = data.shape[0]
        random.seed(self.random_state) # pick a fixed random seed to make experiment reproducible 
        idx_mean_0, idx_mean_1 = random.randint(0, N-1), random.randint(0, N-1)
        return data[idx_mean_0, :2], data[idx_mean_1, :2]
    
    def euclidean_dist(self, data, mean_0, mean_1):
        """
            Args: dataset, current mean for class one, current mean for class two
            Return: dataset with points assigned to closest mean
            
            Calculate distance between each data point and the two mean values, and find the argmin per computation;
            Update classes accordingly
        """
        means = [mean_0, mean_1]
        # objective function: l2-norm (euclidean distance) for k-Means; l1-norm (absolut error criterion) for k-Medoids
        norm = 2 if self.method == 'k-means' else 1
        # fastest way, I am aware of, to update classes of all data points simultaneously (maybe there is a faster way? let me know :) )
        data[:, 2] = [np.argmin([np.linalg.norm(point-mean, norm) for mean in means]) for point in data[:, :2]]
        return data
    
    def update_means(self, data, k=2): 
        """
            Args: dataset and number of classes
            Return: updated mean values per class
        """
        if self.method=='k-means':
            return (np.mean(data[data[:, 2] == i][:, :2], axis=0) for i in range(k))
        elif self.method=='k-medoids':
            return (np.median(data[data[:, 2] == i][:, :2], axis=0) for i in range(k))
    
    def fit_predict(self, data, epochs):
        """
            Args: dataset and number of epochs to train
            Return: dataset and optimized mean values for each of the two classes
        """
        mean_0, mean_1 = self.init_means(data)
        #iterative relocation loop
        for epoch in range(epochs):
            # create data copy to keep track of moving points (NumPy computes changes in place - thus, we need to create a copy of D)
            data_previous = np.copy(data)
            # compute distance between each point and the two mean values, and assign classes
            data = self.euclidean_dist(data, mean_0, mean_1)
            algorithm_label = r'(k-Means)' if self.method == 'k-means' else r'(k-Medoids)'
            centroid_label = r'$\bar x' if self.method == 'k-means' else r'$\tilde x'
            plot_clusters(data, mean_0, mean_1, algorithm_label, centroid_label)
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
            mean_0, mean_1 = self.update_means(data)
        return data, mean_0, mean_1

def plot_clusters(data, mean_0, mean_1, algorithm_label, centroid_label):    
    plt.title('Clusters and corresponding centroids' + ' ' + algorithm_label)
    plt.scatter(data[data[:, 2] == 0][:,0], data[data[:, 2] == 0][:, 1], color = 'turquoise', label='0')
    plt.scatter(data[data[:, 2] == 1][:,0], data[data[:, 2] == 1][:, 1], color = 'cornflowerblue', label='1')
    plt.scatter(mean_0[0], mean_0[1], color = 'red', label = centroid_label + '_0$')
    plt.scatter(mean_1[0], mean_1[1], color = 'darkorange', label = centroid_label + '_1$')
    plt.legend(fancybox=True, framealpha=1, loc='lower right', prop={'size':10})
    plt.show()