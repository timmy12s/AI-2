import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


def toOdorType(data):
    shape = data.values[0]
    x = 0
    if (shape is 'a'):
      x = 1
    elif (shape is 'l'):
      x = 2
    elif (shape is 'c'):
      x = 3
    elif (shape is 'y'):
      x = 4
    elif (shape is 'f'):
      x = 5
    elif (shape is 'm'):
      x = 6
    elif (shape is 'n'):
      x = 7
    elif (shape is 'p'):
      x = 8
    elif (shape is 's'):
      x = 9
    
    return int(x)

stalk_shape = {
    'e': 1,
    't': 2
}
def toStalkShape(data):
    stalk = data.values[0]
    return stalk_shape[stalk]


veil_type = {
    'p': 1,
    'u': 2
}
def toVeilType(data):
    veil = data.values[0]
    return veil_type[veil]


df = pd.read_csv('clustering.csv')
print(df.info())

# FEATURE SELECTION

#Odor Type
df[['odor']] = df[['odor']].apply(toOdorType, axis=1)

#Stalk Shape
df[['stalk-shape']] = df[['stalk-shape']].apply(toStalkShape, axis=1)

#Veil Type
df[['veil-type']] = df[['veil-type']].apply(toVeilType, axis=1)

#Change class 'e' to 1 and 'p' to 0
df[['class']] = df[['class']].apply(lambda col:pd.Categorical(col).codes)
df


# FEATURE EXTRACTION

features = df[['bruises', 'odor', 'stalk-shape', 'veil-type', 'spore-print-color']].values
labels = df[['class']].values


# DATA NORMALIZATION

scaler = StandardScaler().fit(features)
features = scaler.transform(features)
print(features.shape)

pca = PCA(n_components=3)
pca_features = pca.fit_transform(features)
pca_features.shape

 
# ARCHITECTURE
class SOM:
    def __init__(self, width, height, n_features, learning_rate):
        self.width = width
        self.height = height
        self.n_features = n_features
        self.learning_rate = learning_rate

        self.cluster = []
        for i in range(height):
            self.cluster.append([])
        

        self.weight = tf.Variable(
            tf.random.normal(
                [width * height, n_features]
            ),
            tf.float32
        )
        self.input = tf.placeholder(tf.float32, [n_features])
        self.location = []
        for y in range(height):
            for x in range(width):
                self.location.append(tf.cast([y, x], tf.float32))
        
        self.bmu = self.get_bmu()
    
        self.update = self.update_neighbor()


    def get_bmu(self):
        distance = tf.sqrt(
            tf.reduce_mean((self.weight - self.input) ** 2, axis=1)
        )

        bmu_index = tf.argmin(distance)

        bmu_location = tf.cast([
            tf.div(bmu_index, self.width),
            tf.mod(bmu_index, self.width)
        ], tf.float32)

        return bmu_location

    def update_neighbor(self):
        distance = tf.sqrt(
            tf.reduce_mean((self.bmu - self.location) ** 2, axis=1)
        )

        sigma = tf.cast(tf.maximum(self.width, self.height) / 2, tf.float32)
        neighbor_strength = tf.exp(-(distance**2) / (2 * sigma ** 2))
        rate = neighbor_strength * self.learning_rate

        stacked_rate = []
        for i in range(self.width * self.height):
            stacked_rate.append(
                tf.tile(
                    [rate[i]], 
                    [self.n_features]
                )
            )
        
        delta = stacked_rate * (self.input - self.weight)
        new_weight = self.weight + delta

        return tf.assign(self.weight, new_weight)


    #Training
    def train(self, dataset, num_epoch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epoch):
                for data in dataset:
                    dict = {
                        self.input: data
                    }
                    sess.run(self.update, feed_dict = dict)
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Done")
            
            location = sess.run(self.location)
            weight = sess.run(self.weight)

            for i, loc in enumerate(location):
                self.cluster[int(loc[0])].append(weight[i])


som = SOM(10, 10, 3, 0.01)
som.train(pca_features, 2500)
plt.imshow(som.cluster)
plt.show()


