import numpy as np

#Please feel free to change the following parameters
# min and max of data
DATA_MIN = (0, 0)
DATA_MAX = (1, 1)

#Generate a random point
def random_point():
    return np.random.uniform(low = DATA_MIN, high = DATA_MAX ,size=2)

#Generate dataset
def generate_dataset(size=100):
    return np.array([random_point() for i in range(size)])
