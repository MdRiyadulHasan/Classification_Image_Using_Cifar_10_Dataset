import dataset
import visualization
import create_model
import augment_data
import train_model


if __name__ =='__main__':
    x_train,y_train,x_test,y_test=dataset.create_dataset()
    #visualization_input_data = visualization.see_input_image(x_train,y_train)
    train_data = augment_data.data_augmentation(x_train,y_train)
    model=create_model.define_model()
    history =train_model.model_training(model,train_data,x_test,y_test)
    loss_graph = visualization.draw_loss_graph(history)
    accuracy_graph=visualization.draw_accuracy_graph(history)
    
    