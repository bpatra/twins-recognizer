from keras import layers
from keras import models
from keras.applications import VGG16

def vgg16_base(target_size):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(target_size, target_size, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model