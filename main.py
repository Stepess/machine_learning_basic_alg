# Gradient descent algorithm for linear regression
import numpy as np


# minimize the "sum of squared errors".
# This is how we calculate and correct our error
def compute_error_for_line_given_points(b, m, x_arr, y_arr):
    total_error = 0  # sum of square error formula
    for i in range(0, len(x_arr)):
        x = x_arr[i]
        y = y_arr[i]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(x_arr))


def step_gradient(b_current, m_current, x_arr, y_arr, learning_rate):
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    num_of_points = float(len(x_arr))
    for i in range(0, len(x_arr)):
        x = x_arr[i]
        y = y_arr[i]
        b_gradient += -(2 / num_of_points) * (y - (m_current * x + b_current))
        m_gradient += -(2 / num_of_points) * x * (y - (m_current * x + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(x_arr, y_arr, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x_arr, y_arr, learning_rate)
    return [b, m]


def run():
    # Step 1: Collect the data
    x_arr = np.random.rand(100)
    print(x_arr)

    a = 4
    b = 3

    rand_arr = np.random.random_sample(100)
    y_arr = a * x_arr + b + rand_arr
    # Step 2: Define our Hyperparameters
    learning_rate = 0.01  # how fast the data converge
    # y=mx+b (Slope formule)
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    iter_num = 100000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                              compute_error_for_line_given_points(
                                                                                  initial_b, initial_m, x_arr, y_arr)))
    print("Running...")
    [b, m] = gradient_descent_runner(x_arr, y_arr, initial_b, initial_m, learning_rate, iter_num)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(iter_num, b, m,
                                                                      compute_error_for_line_given_points(b, m,
                                                                                                          x_arr,y_arr)))


# main function
if __name__ == "__main__":
    run()
