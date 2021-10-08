from helper_functions import random_point
import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001):  # default parameteres similar to those of sklearn.cluster.KMeans
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = {c:random_point() for c in range(self.n_clusters)} # init centroid to random points
    
    def fit(self,data):
        centroids_history = [] # for the demo and to calculate the evolution difference
        classes_history = [] # for the demo

        for i in range(self.max_iter):

            self.classes = {c:[] for c in range(self.n_clusters)} # init empty set of classes for each centroid

            # CALCULATE DISTANCE BETWEEN POINTS AND CENTROIDS AND CLASSIFY POINTS
            for point in data:
                distances = [np.linalg.norm(point - self.centroids[c]) for c in range(self.n_clusters)] # calculate the euclidean distance between the point and each centroid
                closest = np.argmin(distances) # select closest centroid to the point
                self.classes[closest].append(point) # append the point to centroid's cluster
            

            classes_history.append(dict(self.classes)) # for the demo later
            centroids_history.append(dict(self.centroids)) # saving the current centroids for calculating the difference after update

            #UPDATE CENTROIDS
            for c in range(self.n_clusters):
                if len(self.classes[c]) > 0 :
                    self.centroids[c] = np.average(self.classes[c],axis=0) # centroid becomes average of its cluster


            # CHECK IF GOOD ENOUGH (<tolerance)
            end_training = True # init it to true and make it false if difference < tolerance 
            
            for c in range(self.n_clusters):
                prev_centroid = centroids_history[len(centroids_history)-1][c] # previous centroids
                current_centroid = self.centroids[c]
                difference = np.sum(current_centroid - prev_centroid)

                if difference > self.tol:
                    end_training = False 

            if end_training:
                break

        # SAVING THE LAST UPDATE
        classes_history.append(dict(self.classes)) # for the demo
        centroids_history.append(dict(self.centroids)) # for the demo
        return (centroids_history, classes_history)

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[c]) for c in range(self.n_clusters)]
        return np.argmin(distances) # select closest point
