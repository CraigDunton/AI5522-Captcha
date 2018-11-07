import cv2
import tensorflow as tf
import os.path
import numpy as np
from imutils import paths
from helpers import resize_to_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

tf.logging.set_verbosity(tf.logging.INFO)

LETTERS_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5" # file extension may be different depending on how tensorflow saves them
#MODEL_LABELS_FILENAME = "model_labels.dat"

data = []
labels = []

print("Gathering data and labels...")
for image_file in paths.list_images(LETTERS_FOLDER):
    # grayscale
    image = cv2.imread(image_file)#.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize to 20x20
    image = resize_to_fit(image, 20, 20)

    # get the name of the letter from its folder
    label = image_file.split(os.path.sep)[-2]

    # add to training data
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
print("done")


# split into training and test data (is this k folding?)
# might be a good idea to play w test_size
print("split into training and test data")
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0) # maybe .1 test_size

# turn into numpy arrays for tf
train_data = np.array(X_train)
train_labels = np.array(Y_train)
eval_data = np.array(X_test)
eval_labels = np.array(Y_test)

'''
# make labels binaries (one-hot encoding) <- doesn't work with sparse_softmax loss
lb = LabelBinarizer().fit(train_labels)
train_labels = lb.transform(train_labels)
eval_labels = lb.transform(eval_labels)
'''

# encode labels into ints to be decoded later
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
eval_labels = encoder.fit_transform(eval_labels)

# Build the model **TODO: PLAY WITH HYPERPARAMETERS**
def cnn_model_fn(features, labels, mode):

    print("input layer")
    # input of 20x20 pixel image
    input_layer = tf.reshape(features["x"], [-1, 20, 20, 1])

    print("20 filter conv2d")
    # 20 filter conv2d
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=20,
        kernel_size=[5,5], # we should play w this
        padding="same",    # we should play w this
        activation=tf.nn.relu
    )

    print("max pooling")
    # max pooling
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    print("40 filter conv2d")
    # 40 filter conv2d
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=40,
        kernel_size=[5,5], # we should play w this
        padding="same",    # we should play w this
        activation=tf.nn.relu
    )

    print("max pooling")
    # max pooling
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    print("80 filter conv2d")
    # 80 filter conv2d
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=80,
        kernel_size=[5,5], # we should play w this
        padding="same",    # we should play w this
        activation=tf.nn.relu
    )

    print("max pooling")
    # max pooling
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

    print("40 filter conv2d")
    # 40 filter conv2d
    conv3 = tf.layers.conv2d(
        inputs=pool3,
        filters=40,
        kernel_size=[5,5], # we should play w this
        padding="same",    # we should play w this
        activation=tf.nn.relu
    )

    print("flatten")
    # flatten to 1d... unsure what shape conv3 is though
    # shape = conv3.get_shape()
    # print(shape.num_elements()) => 160
    # TODO: make sure this is reshaped correctly
    conv3_flat = tf.reshape(conv3, [-1, 160])

    print("1000 neuron dense")
    # 1000 neuron dense
    dense1 = tf.layers.dense(inputs=conv3_flat, units=1024, activation=tf.nn.relu)

    # use dropout here? It's used in tf CNN tutorial

    print("100 neuron dense")
    # 100 neuron dense
    dense2 = tf.layers.dense(inputs=dense1, units=100, activation=tf.nn.relu)

    print("36 neuron logit layer")
    # 26 softmax (may need to be 36 for numbers too)
    # use softmax activation here?
    logits = tf.layers.dense(inputs=dense2, units=32, name="logits")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == "VISUALIZE":
        return logits

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# classifier = tf.estimator.Estimator(
#     model_fn=cnn_model_fn, model_dir="./tmp/cnn_model")


# # Set up logging for predictions
# tensors_to_log = {"probabilities": "softmax_tensor"}
# logging_hook = tf.train.LoggingTensorHook(
#     tensors=tensors_to_log, every_n_iter=50)

# # Train the model
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": train_data},
#     y=train_labels,
#     batch_size=1,
#     num_epochs=None,
#     shuffle=True)

logits = cnn_model_fn({"x": train_data}, train_labels, "VISUALIZE")

# create summaries for visualization
with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the
    # raw logit outputs of the nn_layer above, and then average across
    # the batch.
    with tf.name_scope('total'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=train_labels, logits=logits)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    learning_rate = .01
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        cross_entropy)

    
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), train_labels)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# set up for running the session
sess = tf.InteractiveSession()

# Input placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 20, 20], name='x-input') # size determined by the dataset
    y_ = tf.placeholder(tf.int64, [None], name='y-input')
    keep_prob = tf.placeholder(tf.float32)

#feed_dict = {x: train_data, y_: train_labels, keep_prob: 1.0}

def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs = train_data
        ys = train_labels
        k = 1.0
    else:
        xs = eval_data
        ys = eval_labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

steps = 180 # was 2000, then 250

# merge summaries
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test")
tf.global_variables_initializer().run()

for i in range(steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()

# classifier.train(
#     input_fn=train_input_fn,
#     steps=2000,
#     hooks=[logging_hook])

# Evaluate the model and print results
# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": eval_data},
#     y=eval_labels,
#     num_epochs=1,
#     shuffle=False)
# eval_results = classifier.evaluate(input_fn=eval_input_fn)
# print(eval_results)

# predict from eval_data and eval_labels
# predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": np.array([eval_data[0]])},
#     num_epochs=1,
#     shuffle=False
# )

# print("LABELS: ", eval_labels[0])
# the_predictions = classifier.predict(input_fn=predict_input_fn)
# for p in the_predictions:
#     print(p)