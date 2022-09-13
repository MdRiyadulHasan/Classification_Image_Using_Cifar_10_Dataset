from gc import callbacks
from tabnanny import verbose
from sklearn import metrics
from tensorflow import keras 
from keras.callbacks import ModelCheckpoint, EarlyStopping
def model_training(model,train_data,x_test,y_test):
    checkpoint=ModelCheckpoint("models/my_model.h5",
                                  monitor='val_accuracy', 
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0,
                                   patience=3,
                                   verbose=1,
                                   mode='auto')
    callbacks_list=[checkpoint,early_stopping]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history=model.fit(train_data,
                      batch_size=64,
                      validation_data=(x_test,y_test),
                      verbose=1,
                      epochs=5,
                      callbacks=callbacks_list)
    print('model_training_completed')
    return history
