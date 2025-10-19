import numpy as np


def perceptron_classifier(X, Y, c=1, epochs_max=10, verbose=False):
    size, feats = X.shape

    weights = np.zeros(feats)
    bias = 0

    for epoch in range(epochs_max):
        mistakes = 0

        for i in range(size):

            if Y[i] * (weights.dot(X[i]) + bias) <= 0:
                mistakes += 1
                weights += Y[i] * X[i] * c
                bias += Y[i] * c

                if verbose:
                    print(f"EPOCH_{epoch+1}: {X[i]} misclassified")
                    print(f"UPDATE: {weights}, {bias}")

        if verbose:
            print("\n")

        if mistakes == 0:
            if verbose:
                print(f"converged after {epoch+1} epochs")
            break

    return weights, bias


# Example
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_and = np.array([-1, -1, -1, 1])

    w, b = perceptron_classifier(X, Y_and, verbose=True)
    print(w, b)
