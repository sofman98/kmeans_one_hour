from helper_functions import generate_dataset
from kmeans import KMeans
import matplotlib.pyplot as plt


X = generate_dataset()

plt.scatter(X[:,0], X[:,1], s=100)
plt.show()

colors = 99*["b", "g", "r", "c", "m", "y", "k", "w"] # to make sure we don't run out of colors lol

model = KMeans(n_clusters=3)
history = model.fit(X)

# IMPORTANT NOTE: IF YOU WANT TO SEE THE ENTIRE EVOLUTION OF THE CENTROIDS, PLEASE CHANGE THE VARIABLE BELOW
see_full_history = False

start = 0 if see_full_history else len(history)-1
for t in range(start, len(history)):
    centroids = history[0][t]
    classes = history[1][t]

    for c in range(model.n_clusters):
        print(f'centroid {c}: ', centroids[c])
        plt.scatter(centroids[c][0], centroids[c][1], marker="o", color="k", s=200)
        color = colors[c]
        for point in classes[c]:
            plt.scatter(point[0], point[1], marker="^", color=color, s=100)
            
    plt.show()