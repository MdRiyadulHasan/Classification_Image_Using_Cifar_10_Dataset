from matplotlib import pyplot as plt
import numpy as np
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
def see_input_image(x_train, y_train):
    rows=4
    cols=4
    plt.figure(figsize=(8,8))
    for i in range(rows*cols):
        plt.subplot(rows,cols,i+1)
        plt.imshow(x_train[i])
        plt.xticks([])
        plt.yticks([])
        label=np.argmax(y_train[i])
        label_names=class_names[label]
        plt.xlabel(label_names)
        plt.savefig('figures/input_image.png', dpi=600)
    plt.show()
def draw_loss_graph(history):
    plt.plot(history.history['loss'], label='loss', color='blue')
    plt.plot(history.history['val_loss'], label='val_loss', color='orange')
    plt.title('Training Loss and validation Loss Graph')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/Train_Validation_Loss_graph.png', dpi=600)
    plt.show()
    
def draw_accuracy_graph(history):
    plt.plot(history.history['accuracy'], label='accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='val_accuracy', color='orange')
    plt.title('Train and validation accuracy graph')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('figures/Train_validation_accuracy_graph.png', dpi=600)
    plt.show()


