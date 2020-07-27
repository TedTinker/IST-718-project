import random
import numpy as np
from keras import backend
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



### Prepare data, split into train/test

def get_mnist():
    (images, labels), (_, _) = tf.keras.datasets.mnist.load_data()
    
    # Reshape images, normalize so pixels have value -1 to 1 instead of 0 to 255
    images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
    images = (images - 127.5) / 127.5
    
    # Make labels from like "8" to like [0,0,0,0,0,0,0,1,0,0] (one-hot vectors)
    dummied_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        dummied_labels[i, labels[i]] = 1
        
    return(images, dummied_labels)

print("\nCollecting Data...")
images, labels = get_mnist()
image_shape = images.shape[1:]
label_quantity = labels.shape[1]
print("\nData: \n\t Input: {0}, \t Output: {1}.".format(images.shape, labels.shape))
    
def train_test(images, labels, train_percent = .8):
    
    # Shuffle an index, split it into training and test
    index = [i for i in range(len(labels))]
    random.shuffle(index)
    train_index = index[:int(train_percent * len(index))]
    test_index = index[int(train_percent * len(index)):]
    
    # Split actual data with indexes
    x_train = images[train_index]
    x_test = images[test_index]
    y_train = labels[train_index]
    y_test = labels[test_index]
    
    return(x_train, x_test, y_train, y_test)
    
x_train, x_test, y_train, y_test = train_test(images, labels, train_percent = .8)
print("\nTrain: \n\t Input: {0}, \t Output: {1}.".format(x_train.shape, y_train.shape))
print("\nTest: \n\t Input: {0}, \t Output: {1}.".format(x_test.shape, y_test.shape))



### Utilities for convolutional networks

# Dropout-rate
d = .5 

# Noisiness-rate
b = .05

# Random initialization
init = RandomNormal(mean = 0, stddev = .1)

# Reflective padding
def ref_pad(tensor, paddings = [[0,0],[0,0],[0,0],[0,0]]):
    tensor = tf.pad(tensor, mode = "SYMMETRIC", paddings = paddings)
    return(tensor)

# Constraint for neurons
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
constraint = ClipConstraint(.1)



### Check how noisy we're making images

print("\nExample image:")
sample_image = images[0]
plt.imshow(sample_image, cmap='gray')
plt.show()

print("\nExample image with noise:")
noise = np.random.normal(loc=0.0, scale=b, size=sample_image.shape)
sample_image_noisy = sample_image + noise
plt.imshow(sample_image_noisy, cmap='gray')
plt.show()



### Build categorizer

def categorizer_model():
    model = tf.keras.Sequential()
    
    # Add slight noise to input images
    model.add(layers.GaussianNoise(stddev = b))
    
    # Pad with reflective padding
    model.add(layers.Lambda(lambda t: ref_pad(t,paddings = [[0,0],[2,2],[2,2],[0,0]])))
    
    # First convolutional layer
    model.add(layers.Conv2D(
            filters = 64, 
            kernel_size = (5, 5), 
            strides = (2,2),
            kernel_initializer = init,
            kernel_constraint = constraint,
            use_bias=False,
            input_shape=image_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(d))

    # Pad with reflective padding
    model.add(layers.Lambda(lambda t: ref_pad(t,paddings = [[0,0],[2,2],[2,2],[0,0]])))
    
    # Second convolutional layer
    model.add(layers.Conv2D(
            filters = 64, 
            kernel_size = (5, 5), 
            strides = (2,2),
            kernel_initializer = init,
            kernel_constraint = constraint,
            use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(d))

    # Flatten and add final categorization layer
    model.add(layers.Flatten())
    model.add(layers.Dense(label_quantity))
    
    model.build(input_shape = (None,) + image_shape)
    model.summary()

    return model

categorizer = categorizer_model()
categorizer.compile(optimizer='adam', loss='mean_squared_error')



### Train categorizer

epochs = 20
batch = 256
histories = []

history = categorizer.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        batch_size = batch,
    ).history
histories.append(history)
    


### Make plot of model accuracy

loss = histories[0]["loss"]
val_loss = histories[0]["val_loss"]
x = [i for i in range(1, len(loss) + 1)]
    
plt.plot(x,loss, color='blue')
plt.plot(x, val_loss, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.title('Training Loss and Test Loss')
plt.xlim(0,x[-1])
plt.ylim(0,max(loss + val_loss))
plt.show()



### Make confusion matrix

y_pred = list(categorizer.predict_classes(x_test))
y_true = [list(y).index(1) for y in y_test]

confusion_matrix(y_true, y_pred)