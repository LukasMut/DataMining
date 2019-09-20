"""
This is one way of implementing the algorithm. Feel free to implement the functions in another way.
This way of implementing keeps the datatype to be pandas dataframe or pandas series (pretty much the same thing...)
through the entire code. Maybe look into numpy arrays if you want to implement it differently.

Plotting the distribution and inspecting pandas dataframes can be easier to understand in a ipynb.
Feel free to copy this code into Jupyter Notebook or a Google Colab.
I will show you my implementation in a jupyter notebook at the end of the class.
"""

import random
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('geyserData.csv') # Read in data as dataframe
df = df.reset_index(drop=True) # Remove the index from the file
df['class'] = 1 # Assign all points to class 1
df['eruptions'] = df['eruptions'] / max(df['eruptions']) # Normalize eruptions
df['waiting'] = df['waiting'] / max(df['waiting']) # Normalize waiting

plt.title('Scatter plot of the Distribution of data')   #
plt.scatter(df['eruptions'],df['waiting'])              # Visualizing the distribution
plt.show()                                              # 

def initializeMeans(df):
    """
    Intialize the means by randomly taking two points from the dataframe. Hint: check the function 'random.randint'
    return two pandas dataframes (or pandas series) with mean1 and mean2
    """
    raise NotImplementedError

def euclideanDist(df,pointIDX,mean1,mean2):
    """
    Take the index of the point in the dataframe you want to calculate the 
    distance from and calculate the euclidean distance to both means.
    
    return a pandas dataframe (or pandas series) with the closest mean assigned to column 'class' given the pointIDX
    """
    raise NotImplementedError

def updateMean(df):
    """
    df is the dataframe of points containing the assigned classes
    return updated mean1 and mean2 based on the value in column 'class'. mean1 and mean2 should still be pandas dataframes (or pandas series)
    """
    raise NotImplementedError
    
def Kmeans(df,iterations):
    """
    When you have implemented the three functions this function should work and run the Kmeans algorithm!
    """

    mean1,mean2 = initializeMeans(df)

    for iteration in range(iterations): # Change this when implementing how many times it should recalculate the mean

        print("Iteration {}/{}".format(iteration,iteration))
        
        for i in range(len(df)):
            df = euclideanDist(df,i,mean1,mean2)
            
        mean1,mean2 = updateMean(df)

    return df,mean1,mean2

df,mean1,mean2 = Kmeans(df,5)

plt.clf()
plt.scatter(df.loc[df['class'] == 1]['eruptions'],df.loc[df['class'] == 1]['waiting'],color='g',label='1')  #
plt.scatter(df.loc[df['class'] == 2]['eruptions'],df.loc[df['class'] == 2]['waiting'],color='b',label='2')  #
                                                                                                            #
plt.scatter(mean1['eruptions'],mean1['waiting'],s=70,label='mean1',marker='s',color='r')                    # Visualizing the final class distribution
plt.scatter(mean2['eruptions'],mean2['waiting'],s=70,label='mean2',marker='s',color='y')                    #
                                                                                                            #
plt.legend()                                                                                                #
plt.show()                                                                                                  #