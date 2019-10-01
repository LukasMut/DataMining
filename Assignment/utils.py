import numpy as np 
import pandas as pd


# replace NaN values to no lose data examples (we have too few examples to afford losing data)
def replace_nans(data, sbj_indices, sbj_shoe_sizes): 
    """
        Args: pd.DataFrame, indices of subjects with missing values, shoe sizes of subjects with missing values
        Return: dataframe without NaNs (missing values are replaced according to the strategies defined in the for loop)
    """
    no_missings = [idx for idx in range(len(data)) if idx not in sbj_indices]
    for (idx, shoe_size) in zip(sbj_indices, sbj_shoe_sizes):
        # if shoe size exists in data set, replace NaN with mean height for this shoe size
        if len(data.loc[no_missings, ['height']].height[data.shoe_size == shoe_size]) > 0:
            data.loc[idx, 'height'] = data.loc[no_missings, ['height']].height[data.shoe_size == shoe_size].mean()
        # if shoe size does not exist in data set, resort to the following strategy
        else:
            # if shoe size is 36 (= min shoe size of subjects with missing values) replace NaN value with minimum height in the dataset
            if shoe_size == np.min(sbj_shoe_sizes):
                data.loc[idx, 'height'] = data.height.min()
            # if shoe size is 40.5 (= mean shoe size of subjects with missing values) replace NaN value with mean height in the dataset
            # NOTE: this elif statement will never be executed, since shoe sizes of subjects with NaNs are only 36, 36 and 45, 
            #       and yet I implemented this elif statement just to be clean and make the function general (for additional examples)
            elif shoe_size == np.mean(sbj_shoe_sizes):
                data.loc[idx, 'height'] = data.height.mean()
            # if shoe size is 45 (= max shoe size of subjects with missing values) replace NaN value with maximum height in the dataset
            elif shoe_size == np.max(sbj_shoe_sizes):
                data.loc[idx, 'height'] = data.height.max()
    return data

def data_normalization(data, scaling:str, numpy=False, clustering=False):
    """
        Args: pd.DataFrame, feature scaling method (see below which scaling methods are implemented)
        Return: normalized data set (converted into NumPy matrix)
    """
    scalings = ['standardize', 'min-max', 'mean-norm']
    if scaling not in scalings: raise ValueError("Scaling method must be one of ['standardization', 'min-max', 'mean-norm scaling']")
    
    # if dataset has more than one column, compute data normalization within a loop
    if len(data.shape) > 1 and data.shape[1] > 1:
        for column in data.columns:
            if scaling == 'standardize':
                mean, std = data[column].values.mean(), data[column].values.std()
                data[column] -= mean 
                data[column] /= std
            elif scaling == 'min-max':
                minimum, maximum = data[column].values.min(), data[column].values.max()
                data[column] -= mininmum
                data[column] /= (maximum - minimum)
            else:
                mean, maximum, minimum = data[column].values.mean(), data[column].values.max(), data[column].values.min()  
                data[column] -= mean
                data[column] /= (maximum - minimum)
    else:
        if scaling == 'standardize':
            # subtract the mean value from each data point and normalize by the standard deviation
            data = (data - np.mean(data)) / np.std(data)
        elif scaling == 'min-max':
            # subtract the min value from each data point and normalize by the difference between the max and min value
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            # subtract the mean value from each data point and normalize by the difference between the max and min value
            data = (data - np.mean(data)) / (np.max(data) - np.min(data))
            
    # convert pd.DataFrame into NumPy matrix to make computations (e.g., LinAlg operations) faster and Boolean indexing easier
    data = data.to_numpy() if numpy else data
    
    # if clustering, add column with zeros to the end of the matrix (at first, assign every data point to the same class)
    if clustering: return np.c_[data, np.zeros(data.shape[0])]
    else: return data
