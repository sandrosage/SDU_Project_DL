import os
import shutil
import glob
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras import Input
from keras.utils.vis_utils import plot_model
#import visualkeras 
from keras.losses import BinaryCrossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


'''generation of a tree directory and division of data into training, testing and validation with a ratio of 80%, 10% and 10%. 
This way the biggest part is dedicated to training and the model is able to have the highest accuracy, this was confirmed on these following websites 
(https://www.sciencedirect.com/science/article/pii/S2589004222003017, https://link.springer.com/article/10.1007/s42979-021-00695-5, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8423280/) '''
def train_test_val_split(size, train_split, test_split, src_dir):
    category = ["normal", "pneumonia"]
    split_list = ["train", "test", "val"]
    for split in split_list:
        if not os.path.exists("Project_Day\\" + split + "\\" ): #checks if a root directory exists or not, if not it will create one
            os.mkdir("Project_Day\\" + split + "\\")
        for cat in category:
            if not os.path.exists("Project_Day\\" + split + "\\" + cat + "\\"): #checks if the categories directories exist or not, if not they will be created 
                os.mkdir("Project_Day\\" + split + "\\" + cat + "\\")
   
    index = 1
    '''splitting the data from the categories based on the file name and distribute them according to the ratio mentioned above'''   
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        if (index<=(size*train_split)):
            if "normal" in jpgfile:
                shutil.copy(jpgfile, "Project_Day\\" + "train\\" + "normal\\")

            else:
                shutil.copy(jpgfile, "Project_Day\\" + "train\\" + "pneumonia\\")

            index+=1

        elif ((size*train_split)<index<=((size*train_split)+ (size*test_split))):
            if "normal" in jpgfile:
                shutil.copy(jpgfile, "Project_Day\\" + "test\\" + "normal\\")

            else:
                shutil.copy(jpgfile, "Project_Day\\" + "test\\" + "pneumonia\\")

            index+=1
        
        else:
            if "normal" in jpgfile:
                shutil.copy(jpgfile, "Project_Day\\" + "val\\" + "normal\\")

            else:
                shutil.copy(jpgfile, "Project_Day\\" + "val\\" + "pneumonia\\")
                
#train_test_val_split(2200, 0.8, 0.1, "Project_Day\\data")

#index = 0
#for file in os.listdir("Project_Day\\val\\pneumonia"):
#    index+=1
#print(index)

''' '''
def process_data (img_size, batch_size):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
            'Project_Day/train',
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary')
            
    validation_generator = val_datagen.flow_from_directory(
            'Project_Day/val',
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='binary')
    
    return train_generator, validation_generator

train_generator, validation_generator = process_data (256,32)


def build_model(): 
    input_layer = Input(shape=(256,256,3), name='input_layer')
    conv_1 = Conv2D(64, kernel_size=(5,5))(input_layer)
    conv_2 = Conv2D(32,kernel_size=(3,3))(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2,2), strides=(1,1))(conv_2)
    conv_3 = Conv2D(32, kernel_size=(3,3))(pool_1)
    flatten = Flatten()(conv_3)
    dropout = Dropout(0.3)(flatten)
    dense_1 = Dense(64, activation="relu")(dropout)
    dense_2 = Dense(32, activation="relu")(dense_1)
    output_layer = Dense(1, activation="sigmoid")(dense_2)
    model = Model(input_layer, output_layer)
    
    dot_img_file = 'model_2.png'
    plot_model(model, to_file=dot_img_file, show_shapes=True)
    model.summary()
    #visualkeras.layered_view(model) 
    
    model.compile(loss=BinaryCrossentropy(), optimizer="adam", metrics=["accuracy"])
    return model

model = build_model()
batch_size = 32
epochs = 15
steps_per_epochs=128
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, validation_steps=800)

def plot_graph(history):
    loss_train = history.history['train_loss']
    loss_val = history.history['val_loss']
    epochs = range(1,35)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
def evaluate_model(x_train, y_train, model):
    model.predict(x_train)
    score = model.evaluate(x_train, y_train)
    
    print('Accuracy on training data: {}% \n Error in training data: {}'.format(score[1], 1 - score [1]))
    
    pred_test= model.predict(x_train)
    score2 = model.evaluate(x_train, y_train)
    print('Accuracy on test data: {}% \n Error in test data: {}'.format(score[1], 1 - score [1]))