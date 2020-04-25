from keras import models
import matplotlib.pyplot as plt
import numpy as np


def get_activation_model(model, depth):
    layer_outputs = [layer.output for layer in model.layers[:depth]]
    return models.Model(inputs=model.input, outputs=layer_outputs)

def get_activations_tensors(model, depth, img_tensor):
    activation_model = get_activation_model(model, depth=8)
    activations = activation_model.predict(img_tensor)
    return activations

def visualize_on_tensor(model, depth, layer_index, channel_index, img_tensor):
    activations = get_activations_tensors(model, depth, img_tensor)
    plt.matshow(activations[layer_index][0, :, :, channel_index], cmap='viridis')
    
def visualize_layer(model, depth, img_tensor):
    activations = get_activations_tensors(model, depth, img_tensor)

    layer_names = []
    for layer in model.layers[:depth]:
        layer_names.append(layer.name)
        
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        
        size = layer_activation.shape[1]
        
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                
                channel_image -= channel_image.mean()
                std = channel_image.std()
                if std > 0:
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
                
        scale = 1. / size
        plt.figure(figsize=(scale*display_grid.shape[1],
                        scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')