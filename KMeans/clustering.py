import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 


def normalize_data(df, scaling:str):
    """
        Args: pd.DatFrame, feature scaling method (see below which are possible to choose from)
        Return: normalized data set (converted into NumPy matrix)
    """
    scalings = ['standardize', 'min-max', 'mean-norm']
    if scaling not in scalings:
        raise Exception('Features can only be scaled according to standardization, min-max or mean-norm scaling')
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
    
    def __init__(self, init:str):
        if init == 'random':
            self.init = init
        else:
            raise Exception('Means have to be initialised with randomly chosen data points')
            
    def init_means(self, data):
        """            
            Args: object and dataset
            Return: two randomly chosen data points or two points at zero
            
            Intialize the means by randomly taking two points from the dataset
            or initialise them with zeros (investigate which method more rapidly leads to optimal means)
        """
        N = data.shape[0]
        if self.init=='random':
            idx_mean_0, idx_mean_1 = random.randint(0, N-1), random.randint(0, N-1)
        else:
            raise Exception('Initialise means with randomly chosen data points')
        return data[idx_mean_0, :2], data[idx_mean_1, :2]
    
    def euclidean_dist(self, data, mean_0, mean_1):
        """
            Args: dataset, current mean for class one, current mean for class two
            Return: dataset with points assigned to closest mean
            
            Calculate distance between each data point and the two mean values, and find the argmin per computation;
            Update classes accordingly
        """
        means = [mean_0, mean_1]
        # fastest way, I am aware of, to update classes of all data points simultaneously (maybe there is a faster way? let me know :) )
        data[:, 2] = [np.argmin([np.linalg.norm(point-mean) for mean in means]) for point in data[:, :2]]
        return data
    
    def update_means(self, data, k=2): 
        """
            Args: dataset and number of classes
            Return: updated mean values per class
        """
        return (np.mean(data[data[:, 2] == i][:, :2], axis=0) for i in range(k))
    
    def fit_predict(self, data, epochs):
        """
            Args: dataset and number of epochs to train
            Return: dataset and optimized mean values for each of the two classes
        """
        mean_0, mean_1 = self.init_means(data)
        for epoch in range(epochs):
            # create data copy to keep track of moving points (NumPy computes changes in place)
            data_copy = np.copy(data)
            # compute distance between each point and the two mean values, and assign classes
            data = self.euclidean_dist(data, mean_0, mean_1)
            plot_centroids(data, mean_0, mean_1)
            if np.all(data[:, 2] == data_copy[:, 2]):
                print('0 points moved this round. We could stop optimization, but lets also compare the centroids first.')
            else:
                booleans = data[:, 2] == data_copy[:, 2]
                n_points_moved = len(booleans[booleans==False])
                string = 'points' if n_points_moved > 1 else 'point'
                print('{} {} moved this round. Optimization continues.'.format(n_points_moved, string))
            mean_0_current, mean_1_current = self.update_means(data)
            # if mean values have not changed, then break loop and stop optimization
            if np.all(mean_0_current == mean_0) and np.all(mean_1_current == mean_1):
                print('Algorithm needed {} iterations until optimal mean values were found.'.format(epoch+1))
                break
            else:
                mean_0, mean_1 = mean_0_current, mean_1_current
        return data, mean_0, mean_1

def plot_centroids(data, mean_0, mean_1):    
    plt.title('Data distribution and corresponding centroids')
    plt.scatter(data[data[:, 2] == 0][:,0], data[data[:, 2] == 0][:, 1], color = 'green', label='0')
    plt.scatter(data[data[:, 2] == 1][:,0], data[data[:, 2] == 1][:, 1], color = 'blue', label='1')
    plt.scatter(mean_0[0], mean_0[1], color = 'red', label=r'$\bar x_0$')
    plt.scatter(mean_1[0], mean_1[1], color = 'orange', label=r'$\bar x_1$')
    plt.legend(fancybox=True, framealpha=1, loc='lower right', prop={'size':10})
    plt.show()