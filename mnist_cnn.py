import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

numEpochs = 16
batchSize = 256
fullConW1Size = 200
conW1Fill = 20
conW2Fill = 30
outSize = 10

picDimH = 28
picDimW = 28

#28 by 28 data
def convModel():
    # with tf.variable_scope("ConvNet"):
    inPic = tf.placeholder("float32", [None, 28, 28, 1])
    trainData = tf.placeholder("float32", [None, outSize])
    #initilize variables for model
    conW1 = tf.Variable(tf.random_normal([5, 5, 1, conW1Fill]))
    conW2 = tf.Variable(tf.random_normal([5, 5, conW1Fill, conW2Fill]))
    #7 depends on step/stride size from pooling
    fullConW1 = tf.Variable(tf.random_normal([7*7*conW2Fill, fullConW1Size]))
    fullConW1B = tf.Variable(tf.random_normal([fullConW1Size]))
    outLayer = tf.Variable(tf.random_normal([fullConW1Size, outSize]))
    outLayerB = tf.Variable(tf.random_normal([outSize]))


    #reshape input
    # inPic = tf.reshape(inPic, )
    #setup model
    #convolutional layer
    convL1 = tf.nn.conv2d(inPic, conW1, strides=[1, 1, 1, 1], padding="SAME")
    convL1 = tf.nn.max_pool(convL1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    convL2 = tf.nn.conv2d(convL1, conW2, strides=[1, 1, 1, 1], padding="SAME")
    convL2 = tf.nn.max_pool(convL2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    #fully connected layer
    flIn = tf.reshape(convL2, [-1, 7*7*conW2Fill])
    fl1 = tf.matmul(flIn, fullConW1) + fullConW1B
    fl1 = tf.nn.relu(fl1)
    out = tf.matmul(fl1, outLayer) + outLayerB


    cost = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=trainData)
    cost = tf.reduce_mean(cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    return inPic, trainData, out, cost, optimizer

def train():
    #get the session
    with tf.Session() as sess:
        inPic, trainData, out, cost, optimizer = convModel()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(numEpochs):
            loss = 0
            for _ in range(int(mnist.train.num_examples/batchSize)):
                pic, correctClass = mnist.train.next_batch(batchSize)
                pic = pic.reshape([batchSize, picDimH, picDimW, 1])
                _, lose = sess.run([optimizer, cost], feed_dict={inPic: pic, trainData: correctClass})
                loss += lose

            correct = tf.equal(tf.argmax(out, 1), tf.argmax(trainData, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float32"))
            print("Epoch:", epoch, "Cost:", loss)
            print("Accuracy:", accuracy.eval({inPic:mnist.test.images.reshape([-1, picDimH, picDimW, 1]), trainData:mnist.test.labels}))
        saver.save(sess, 'mnistModel', global_step=0)

#sess = tf.InteractiveSession()

train()
