import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from xl_to_csv import make_array
from tag_data import read_to_array

TAG_FILES = ['']

DAT_FILENAMES = ['']

x = make_array(DAT_FILENAMES)
y = read_to_array(TAG_FILES)
x_train = x[0:150].T
y_train = y[0:150].T
x_test = x[150:len(x)].T
y_test = y[150:len(y)].T




print("X shape: " + str(x_train.shape))
print("X type: " + str(type(x_test)))
print("Y shape: " + str(y_train.shape))
print("Y type: " + str(type(y_test)))

def create_placeholders(n_x, n_y):
    #n_x = size of single dataset; 600
    #n_y = number of classes: 1
    X = tf.placeholder(tf.float32, shape = [n_x, None], name = "X")
    Y = tf.placeholder(tf.float32, shape = [n_y, None], name = "Y")
    return X, Y

def init_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [25,600], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

def forward_prop(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)

    return Z3

def one_hot_matrix(labels, C):
    C = tf.constant(C, name = "C")
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)
        #print(str(one_hot[0]))
        return one_hot[0]

#print(y_train)
#one_hot_matrix(y_train, 1)

def compute_cost(Z3,Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

def split_minibatches(x, y, num_minibatches):
    x_split = np.hsplit(x, num_minibatches)
    y_split = np.hsplit(y, num_minibatches)
    return x_split, y_split


y_train = one_hot_matrix(y_train, 1)
y_test = one_hot_matrix(y_test, 1)


def model(x_train, y_train, x_test, y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 5, print_costs = True):

    tf.reset_default_graph()
    tf.set_random_seed(1)
    #seed = 3

    (n_x, m) = x_train.shape
    n_y = y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = init_parameters()
    Z3 = forward_prop(X, parameters)
    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.0
            num_minibatches = int(m/minibatch_size)
            #seed = seed + 1
            x_batches, y_batches = split_minibatches(x_train, y_train, num_minibatches)
            for batch in range(len(x_batches)):
                _, minibatch_cost = sess.run([optimizer,cost], feed_dict={X: x_batches[batch], Y: y_batches[batch]})
                epoch_cost += minibatch_cost/num_minibatches
            if print_costs == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_costs == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy: " , accuracy.eval({X: x_train, Y: y_train}))
        print("Test Accuracy: " , accuracy.eval({X: x_test, Y: y_test}))
        return parameters

parameters = model(x_train, y_train, x_test, y_test)

#split_minibatches(x_train, y_train, 10)
