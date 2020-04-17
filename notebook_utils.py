import matplotlib.pyplot as plt
from time import time
from keras import optimizers

def print_image_from_gen(img_tensor):
    print(img_tensor.shape)
    plt.imshow(img_tensor)
    plt.show()
    
def print_image_from_path(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    print(type(img))
    img_tensor = image.img_to_array(img)
    print(img_tensor.shape)
    img_tensor /= 255.0

    plt.imshow(img_tensor)
    plt.show()

def plot_history2(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['acc']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label="Validation loss")
    plt.title('Taining and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc_values, 'bo', label='Training acc.')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc.')
    plt.title('Taining and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    
def getName(label):
    if label < 0.5:
        return "J"
    else: 
        return "L"

def view_prediction(model, batch):
    batch1=batch[0]
    results = model.predict(batch1)
    for i in range(0, batch1.shape[0]):
        print(f"Machine predicts {getName(results[i])} when it's {getName(batch[1][i])}")

        print_image_from_gen(batch1[i])