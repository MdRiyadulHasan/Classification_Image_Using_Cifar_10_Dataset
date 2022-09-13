from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
def data_augmentation(x_train,y_train):
    datagen=ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_data=datagen.flow(x_train,y_train, batch_size=64)
    return train_data


