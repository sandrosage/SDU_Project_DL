import os
import shutil
import glob
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.models import Model
from keras import Input
from keras import regularizers
from keras.utils.vis_utils import plot_model
#import visualkeras 
from keras.losses import BinaryCrossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.version_utils import callbacks

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

''' 
The color values are being normalized from (1,255) to (0,1).

The images are augmented in small amounts to achieve robustness towards small inconsistencies in the input images. 
We've used most of the data augmentation parameters, but flipping is being overlooked due to asymmetries in the human body.
Assuming the input data comes from a decently controlled environment, we've chosen to keep all values low.
'''


def process_data (img_size, batch_size):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=False,
            brightness_range=[0.9, 1.1],
            rotation_range=(3)

            )

    val_datagen = ImageDataGenerator(rescale=1./255)

    """
    The method flow_from_directory is being used to fetch data from a directory small batches at a time.
    This helps with memory management during training and validation.
    Target_size corresponds to our input image dimensions. 
    The ProjectData folder holds subfolders for training, validation and testing, all of which have subfolders for both classes 'normal' and 'pneumonia'.
    """
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

'''
build_model is used to create the architecture of the structure using the functional API of keras 
Convolutional layers are being used to look for interesting features. 
    These are translation invariant by default which should help with the different shaped bodies of the patients.
    The number of filters in the convolutional layers indicate the number of different features that the layers will be looking for.
    The kernel size is one of common practice. These kernels are pretty small which corresponds to looking for low-level features. 
    By stacking multiple convolutional layers we can achieve higher and higher level features.
Dropout is introduced to have stronger individual nodes. 
MaxPooling helps tame parameter numbers.
Layer flattening has to be used to reduce dimensionality to connect convolutional layers to dense ones.

The output layer activation function is sigmoid since this is a binary classification problem.
    Sigmoid provides us with values between [0,1]. This gives us a confidence value for our prediction.
    We could also use the SoftMax activation function, but it is more suited for multi-categorical classification problems.
For a loss function we use BinaryCrossentropy for the same reason as above mentioned. incase of the Binary classification we use one output neuron.
Regarding optimizers, we've stuck to the basics of gradient descent and used Adam and Nadam (in different runs). 
    They have differences namely in how they function around local minima, but for the most part they are identical.
'''
def build_model(): 
    input_layer = Input(shape=(256, 256, 3), name='input_layer')
    dropout = Dropout(0.3)(input_layer)
    conv_1 = Conv2D(64, kernel_size=(5, 5), kernel_regularizer=regularizers.l2(0.001))(dropout)
    bnorm_1 = BatchNormalization()(conv_1),
    conv_2 = Conv2D(32,kernel_size=(3, 3))(bnorm_1)
    pool_1 = MaxPooling2D(pool_size=(2,2), strides=(1,1))(conv_2)
    conv_3 = Conv2D(32, kernel_size=(3,3))(pool_1)
    flatten = Flatten()(conv_3)
    dropout = Dropout(0.3)(flatten)
    dense_1 = Dense(64, activation="relu")(dropout)
    dense_2 = Dense(32, activation="relu")(dense_1)
    output_layer = Dense(1, activation="sigmoid")(dense_2)
    model = Model(input_layer, output_layer)
    
    #dot_img_file = 'model_2.png'
   #plot_model(model, to_file=dot_img_file, show_shapes=True)
    model.summary()
    #visualkeras.layered_view(model) 
    
    model.compile(loss=BinaryCrossentropy(), optimizer="adam", metrics=["accuracy"])
    return model

model = build_model()

batch_size = 32
epochs = 10

"""
During the training process we are keeping track of the current model's accuracy and comparing it to the best one so far.
The best model is being stored for future reference and replaced as necessary. 
As a comparison metric we use the validation accuracy.
"""
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, validation_steps=800 ) #callbacks=model_checkpoint_callback)

model.load_weights(checkpoint_filepath) # load the best model into memory

'''
Visualization method for the results. Also prints the training and validation loss.
'''


def plot_graph_2(history):
    loss_train = history.history['train_loss']
    loss_val = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs_size = 10
    epochs = range(1, (epochs_size+1))

    ig, ax = plt.subplots(2, figsize=(20, 8))
    ax[0].plot(epochs ,loss_train, "b", label="Training loss")
    ax[0].plot(epochs, loss_val, "r", label="Validation loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("%")
    ax[0].set_title("Loss of Model_V2:")
    ax[0].legend()
    ax[1].plot(epochs, accuracy, "g", label="Training accuracy: " + str(accuracy))
    ax[1].plot(epochs, val_accuracy, "y", label="Validation accuracy: " + str(val_accuracy))
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("%")
    ax[1].set_title("Accuracy of Model_V2:")
    ax[1].legend()

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.8)

    plt.show()


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

'''
Visualization method for comparing accuracies between training and validation.
'''


def plot_feature_map(rows, columns, x_test):
    activation_model = Model(inputs=model.input, outputs=model.layers[0].output)
    activations = activation_model.predict(x_test[1].reshape(1,28,28,1))
    act_index = 0
    fig, ax = plt.subplots(rows,columns, figsize=(rows*2.5, columns*2.5))
    for row in range(0,rows):
        for col in range(0,columns):
            ax[row][col].imshow(activations[0, :, :, act_index], cmap="gray")
            act_index+= 1
    plt.suptitle(str(rows*columns) + '- feature map',fontsize=20)
    plt.show()



def get_kernels(layer_name, rows, columns):
    kernels = model.get_layer(name=layer_name).get_weights()[0][:,:,0, :]
    index = 0
    fig, ax = plt.subplots(rows,columns, figsize=(20, 8))
    for row in range(0,rows):
        for col in range(0,columns):
            ax[row][col].imshow(kernels[:,:,index], cmap="gray")
            index += 1
    plt.suptitle(str(rows*columns) + ' Kernel filters',fontsize=20)
    plt.show()



def evaluate_model(x_train, y_train, x_test, y_test, model):
    model.predict(x_train)
    score = model.evaluate(x_train, y_train)
    
    print('Accuracy on training data: {}% \n Error in training data: {}'.format(score[1], 1 - score [1]))
    
    pred_test= model.predict(x_test)
    score2 = model.evaluate(x_test, y_test)
    print('Accuracy on test data: {}% \n Error in test data: {}'.format(score[1], 1 - score [1]))


get_kernels("conv2d_10", 5, 6)