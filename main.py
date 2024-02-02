from data_preparation import Dataset
import numpy as np
import matplotlib.pyplot as plt
import time

def loss(y,a):
    return -(y*np.log(a) + (1-y)*np.log(1-a))

def get_image(dataset, i):
    return dataset.__getitem__(i)[0][0,:,:]

def vectorize(dataset):
    m = dataset.__len__()
    M = np.zeros(shape=(28*28, m))

    for i in range(m):
        x = get_image(dataset, i).view(-1).numpy().reshape(28*28,1)
        M[:,i] = x[:,0]

    return M

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def vect_sigmoid(A):
    return np.vectorize(sigmoid)(A)


if __name__ == "__main__":
    dataset = Dataset()

    train_set = dataset.train_subset
    m = train_set.__len__()

    Y = dataset.train_targets.numpy().reshape(1, m)
    X = vectorize(train_set)

    m = train_set.__len__()
    w = np.zeros((28*28,1))
    b = 0
    alpha = 0.1

    J,dw1,dw2,db=0,0,0,0

    start = time.time()

    total_episodes = 1000
    for episode in range(total_episodes):
        print("episode: ", episode)
        Z = np.dot(w.T, X) + b
        A = sigmoid(Z)
        dZ = A - Y
        dw = np.dot(X,dZ.T) / m
        db = np.sum(dZ) / m

        w = w - alpha * dw
        b = b - alpha * db


    end = time.time()
    duration = end-start
    
    print(f"Training {total_episodes} took {duration} seconds")

    test_set = dataset.test_subset
    n = test_set.__len__()
    accuracy = 0

    X = vectorize(test_set)
    Y = dataset.test_targets.numpy().reshape(1, n)
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    predictions = (A > 0.5).astype(int) 
    accuracy = np.mean(predictions == Y)  

    print("Achieved accuracy of ", accuracy * 100, "%")


    
    