import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA

cap_shape = {
    'b': 1,
    'c': 2,
    'x': 3,
    'f': 4,
    'k': 5,
    's': 6
}
def toCapShape(data):
    shape = data.values[0]
    return cap_shape[shape]


odor_type = {
    'a': 1,
    'l': 2,
    'c': 3,
    'y': 4,
    'f': 5,
    'm': 6,
    'n': 7,
    'p': 8,
    's': 9,
}
def toOdorType(data):
    odor = data.values[0]
    return odor_type[odor]


habitat_type = {
    'g': 1,
    'l': 2,
    'm': 3,
    'p': 4,
    'u': 5,
    'w': 6,
    'd': 7,
}
def toHabitatType(data):
    hab = data.values[0]
    return habitat_type[hab]


dataset = pd.read_csv('classification.csv')
dataset.info()

# Feature Selection

# Cap Shape
dataset[['cap-shape']] = dataset[['cap-shape']].apply(toCapShape, axis=1)

# Odor Type
dataset[['odor']] = dataset[['odor']].apply(toOdorType, axis=1)

# Habitat
dataset[['habitat']] = dataset[['habitat']].apply(toHabitatType, axis=1)

# Change class 'e' to 1 and 'p' to 0
isEdible = lambda k: 1 if k.values[0] is 'e' else 0
dataset[['class']] = dataset[['class']].apply(isEdible, axis=1)
dataset

 
# Feature Extraction
 
# Features and labels split
features = dataset[['cap-shape', 'cap-color', 'odor', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 'ring-number', 'habitat']].values
labels = dataset[['class']].values

# Data normalization
scaler = MinMaxScaler().fit(features)
features = scaler.transform(features)
print(features.shape)

pca = PCA(n_components=4)
pca_features = pca.fit_transform(features)
print(pca_features.shape)


# Architecture
epoch = 2500
lr = .15
n_features = pca_features.shape[1]
n_classes = len(np.unique(labels))

input_tensor = tf.placeholder(tf.float32)
output_tensor = tf.placeholder(tf.float32)

neurons = [64, n_classes]
n_layers = len(neurons)


# Weight bias declaration
parameters = {}
for i in range(n_layers):
    if i == 0:
        w = tf.Variable(tf.random.normal([n_features, neurons[i]]), tf.float32)
        b = tf.Variable(tf.random.normal([1, neurons[i]]), tf.float32)
    else:
        w = tf.Variable(tf.random.normal([neurons[i-1], neurons[i]]), tf.float32)
        b = tf.Variable(tf.random.normal([1, neurons[i]]), tf.float32)

    parameters[f'W{i+1}'] = w
    parameters[f'B{i+1}'] = b

print(parameters)


# Training

# Concat features and labels before shuffle
df = np.concatenate([pca_features, labels], axis=1)

train_size = int(0.7 * len(df))
valid_size = int(0.2 * len(df))
test_size = int(0.1 * len(df))

# Shuffle array
np.random.shuffle(df)

train = df[:train_size]
valid = df[train_size:valid_size + train_size]
test = df[valid_size + train_size:]

print(train.shape, valid.shape, test.shape)

# Split data into features and labels
x_train, y_train = train[:, :-1], train[:, -1]
x_valid, y_valid = valid[:, :-1], valid[:, -1]
x_test, y_test = test[:, :-1], test[:, -1]

# Reshaping Labels
y_train = y_train.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Encode labels
encoder = OneHotEncoder().fit(y_train)
y_train = encoder.transform(y_train).toarray()
y_valid = encoder.transform(y_valid).toarray()
y_test = encoder.transform(y_test).toarray()

def feed_forward(input_tensor, n_layers):
    # print('start feed')
    a = input_tensor
    for i in range(n_layers):
        w = parameters[f'W{i+1}']
        b = parameters[f'B{i+1}']

        # print(a.shape, w.shape)
        z = tf.matmul(a, w) + b
        if i < n_layers - 1:
            a = tf.nn.relu(z)
        else:
            a = tf.nn.softmax(z)
    
    return a


# Create Saver
saver = tf.train.Saver()

# Model
pred_tensor = feed_forward(input_tensor, n_layers)
loss_tensor = tf.reduce_mean(0.5 * (output_tensor - pred_tensor)**2)

true_preds_tensor = tf.equal(tf.argmax(pred_tensor, axis=1), tf.argmax(output_tensor, axis=1))
acc_tensor = tf.reduce_mean(tf.cast(true_preds_tensor, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss_tensor)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_dict = {
        input_tensor: x_train,
        output_tensor: y_train
    }

    valid_dict = {
        input_tensor: x_valid,
        output_tensor: y_valid
    }

    best_loss = float('inf')

    print(f'=== TRAINING MODEL ===')
    for i in range(1, epoch+1):
        # print(i)
        sess.run(optimizer, feed_dict=train_dict)

        if i % 25 == 0:
            loss = sess.run(loss_tensor, feed_dict=train_dict)
            print(f'EPOCH: {i} -- Error: {loss:.4f}')
        
        if i % 125 == 0:
            loss = sess.run(loss_tensor, feed_dict=valid_dict)
            print(f'EPOCH: {i} -- Validation Error: {loss:.4f}')

            if loss < best_loss:
                best_loss = loss
                saver.save(sess, './best_model.ckpt')

 
# Evaluation

with tf.Session() as sess:
    saver.restore(sess, 'best_model.ckpt')

    acc = sess.run(
        acc_tensor, 
        feed_dict={
            input_tensor: x_test,
            output_tensor: y_test
        })

    print(f'ACCURACY: {acc*100:.2f}%')
