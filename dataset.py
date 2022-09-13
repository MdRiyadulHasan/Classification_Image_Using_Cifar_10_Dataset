import tensorflow
from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import to_categorical

def create_dataset():
    (x_train,y_train),(x_test,y_test)=cifar10.load_data()
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_train=x_train/255.0
    x_test=x_test/255.0
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    print(y_train[:5])
    print(x_train.shape)
    print(y_train.shape)
    return x_train,y_train,x_test,y_test