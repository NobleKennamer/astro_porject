import numpy as np

# def l2_error(data, y, weights):
#     y_hat = data.dot(weights)
#     sqErrors = (y_hat - y)
#     return  (1.0 / (y.size)) * (y_hat - y).T.dot((y_hat - y))


def grad_descent(data, y, weights, num_iters, l_r):
    for i in range(num_iters):
        y_hat = data.dot(weights)
        for w in range(weights.size):
            error = (y_hat - y) * data[:, w].reshape((y.size, 1))
            weights[w][0] = weights[w][0] - l_r *(1.0 / y.size) * error.sum()

    return weights

def run_linear_regression(train_x, train_y, test_x, test_y, num_iter, learning_rate):
    num_features = train_x.shape[1]
    N = train_y.size
    train_y.shape = (N, 1)
    data = np.ones(shape=(N, num_features+1))
    data[:, 1:num_features+1] = train_x
    weights = np.zeros(shape=(num_features+1, 1))
    weights = grad_descent(data, train_y, weights, num_iter, learning_rate)
    return np.hstack([np.ones(shape=(test_x.shape[0], 1)), test_x]).dot(weights).ravel()
