import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
import numpy as np
from xl_to_csv import make_array
from tag_data import read_to_array

TAG_FILES = ['']

DAT_FILENAMES = ['']

def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name='X')
    W = tf.constant(np.random.randn(4,3), name='W')
    b = tf.constant(np.random.randn(4,1), name='b')
    Y = tf.add(tf.matmul(W,X),b)

    with tf.Session() as session:
        result = session.run(Y)

    return result
#def los(x, y, x_test, y_test):

x = make_array(DAT_FILENAMES)
y = read_to_array(TAG_FILES)
assert len(x) == len(y)


#convert into tensorflow dataset
#change 150/151 to change size of train/test set
x_train = tf.constant(x[0:150])
y_train = tf.constant(y[0:150])
x_test = tf.constant(x[151:len(x)])
y_test = tf.constant(y[151:len(y)])

###create TF Dataset Objects
train_data = ops.convert_to_tensor(x_train, dtype=dtypes.float64)
train_label = ops.convert_to_tensor(y_train, dtype=dtypes.int64)
test_data = ops.convert_to_tensor(x_test, dtype=dtypes.float64)
test_label = ops.convert_to_tensor(y_test, dtype=dtypes.int64)

#create input cues
train_input_que = tf.train.slice_input_producer([train_data, train_label], shuffle=False)
test_input_que = tf.train.slice_input_producer([test_data, test_label], shuffle=False)


train_data_batch, train_label_batch = tf.train.batch([train_data, train_label], batch_size=len(x[0:150]))
test_data_batch, test_label_batch = tf.train.batch([test_data, test_label], batch_size=len(x[151:len(x)]))

r = train_data.reshape(train_data).T
print(r.shape)
#define model
def cost(logits, labels):
    #z = tf.placeholder(tf.)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)

#eq = tf.sigmoid(x) #just a test model
    with tf.Session() as sess:
        cost = sess.run(cost)
        return cost
