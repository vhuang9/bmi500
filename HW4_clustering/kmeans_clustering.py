import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt

def run(fname):
    X, ypred, ytrue = classify(fname)
    acc, prec, recall = results(ypred, ytrue)
    visualize(X, ytrue, ypred, save_figs = 'yes')

    print('Accuracy: ',acc,'\nPrecision: ',prec,'\nRecall: ',recall)

def classify(fname):
# uses k-means clustering to perform classification on the heart.csv dataset using 3 pre-selected features
# Inputs:
#     fname: path to heart.csv file
# Outputs: 
#     X: dataframe containing features in first three columns
#     ypred: predicted labels from classification
#     ytrue: ground-truth labels

    df = pd.read_csv(fname)      # read in heart.csv into a dataframe
    
    X = df[['age','trestbps','thalach']]   # choose 3 features for the clustering model

    xs = X.iloc[:,0]
    ys = X.iloc[:,1]
    zs = X.iloc[:,2]
    ytrue = df['target']    # save the observed heart attack labels

    # initialize and fit kmeans clustering model from sklearn using 3 features
    kmeans = KMeans(n_clusters=2, init='k-means++',random_state=20)    
    kmeans.fit(X)

    ypred = kmeans.predict(X)
    
    return X, ypred, ytrue

def results(ypred, ytrue):
# returns metrics of the classification task
# Inputs:
#     ypred: predicted labels from classification
#     ytrue: ground-truth labels
# Outputs:
#     accuracy
#     precision
#     recall
    return metrics.accuracy_score(ytrue, ypred), metrics.precision_score(ytrue, ypred), metrics.recall_score(ytrue, ypred)

def visualize(X,ytrue,ypred,save_figs = 'yes'):
# creates a 3D scatter plot of the true labels and predictions from clustering
# Inputs:
#     X: dataframe containing features in first three columns
#     ypred: predicted labels from classification
#     ytrue: ground-truth labels
#     save_figs (optional): yes to save figures (default) otherwise does not save

    xs = X.iloc[:,0]
    ys = X.iloc[:,1]
    zs = X.iloc[:,2]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs,ys,zs,c=ypred)
    ax.set_xlabel('age')
    ax.set_ylabel('resting BP')
    ax.set_zlabel('cholesterol')
    plt.title('Heart attack predictions')
    if save_figs == 'yes':
        plt.savefig('predictions.png')        
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs,ys,zs,c=ytrue)
    ax.set_xlabel('age')
    ax.set_ylabel('resting BP')
    ax.set_zlabel('cholesterol')
    plt.title('Heart attack observations')
    if save_figs == 'yes':
        plt.savefig('true_labels.png')
    plt.show()
    

