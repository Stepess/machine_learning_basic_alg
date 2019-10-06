# Gradient descent algorithm for linear regression
import numpy as np
import matplotlib.pyplot as plt


def lost_function(a, b, x_arr, y_arr):
    """
    calculates lost function based as sum of squared errors
    we need to minimize the "sum of squared errors"
    this is to see how far we from correct value
    :return: total error
    """
    return np.sum((y_arr - (a * x_arr + b)) ** 2) / float(len(x_arr))


def step_gradient(a_current, b_current, x_arr, y_arr, learning_rate):
    """
    on step of gradient descent
    :return: new value of a and b
    """
    num_of_points = float(len(x_arr))

    # Partial derivative of lost function by a
    a_gradient = np.sum(-(2 / num_of_points) * x_arr * (y_arr - (a_current * x_arr + b_current)))
    # Partial derivative of lost function by b
    b_gradient = np.sum(-(2 / num_of_points) * (y_arr - (a_current * x_arr + b_current)))

    # Minus because we actually need move in direction of antigradient (- grad) to minimize lost function
    new_a = a_current - (learning_rate * a_gradient)
    new_b = b_current - (learning_rate * b_gradient)

    return [new_a, new_b]


def gradient_descent_runner(x_arr, y_arr, starting_a, starting_b, learning_rate, num_iterations):
    a = starting_a
    b = starting_b

    for i in range(num_iterations):
        a, b = step_gradient(a, b, x_arr, y_arr, learning_rate)

    return [a, b]


if __name__ == "__main__":
    # init sample
    sample_size = 100

    x_sample = np.random.rand(sample_size)

    a_real = 2.5
    b_real = 1.6

    noise_arr = np.random.random_sample(sample_size)
    y_sample = a_real * x_sample + b_real + noise_arr

    plt.plot(x_sample, y_sample, 'ro')
    plt.show()

    # hyperparameters
    lr = 0.01  # how fast the data converge

    # initial guesses
    initial_a = 0
    initial_b = 0

    iter_num = 10000

    print("Starting gradient descent at a = {0}, b = {1}, error = {2}"
          .format(initial_a, initial_b,
                  lost_function(initial_a, initial_b, x_sample, y_sample)))

    print("Running...")

    [a_found, b_found] = gradient_descent_runner(x_sample, y_sample, initial_a, initial_b, lr, iter_num)

    print("After {0} iterations a = {1}, b = {2}, error = {3}"
          .format(iter_num, a_found, b_found,
                  lost_function(a_found, b_found, x_sample, y_sample)))
