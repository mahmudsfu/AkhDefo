
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                    img_to_array)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

def Lenet_Model_training(dataset="DataForTraining", model_out="foggy_not_foggy.model", plot="Model_stat_plot.png", EPOCHS = 100,
    INIT_LR = 1e-3,  BS = 32):

    '''

    This function, Lenet_Model_train(), is designed to train a convolutional neural network (CNN) using the LeNet architecture. The network is trained on a dataset of images to classify whether they are "foggy" or "not foggy".

    Parameters:
    -----------

    dataset: str
      (default="DataForTraining") Path to the directory containing the image data for training. The images are expected to be in separate directories named after their corresponding class ("foggy" or "not foggy").
    model_out: str
      (default="foggy_not_foggy.model") The name or path for the output file where the trained model will be saved in the h5 format.
    plot: str
     (default="Model_stat_plot.png") The name or path for the output image file where a plot of the training loss and accuracy will be saved.
    EPOCHS: int
      (default=100)The number of epochs to use for training.
    INIT_LR: float
      (default=1e-3)The initial learning rate for the Adam optimizer.
    BS: int
      (default=32)The batch size for training.

    Returns:
    --------
    - Trains a LeNet model on the given dataset.
    - Saves the trained model to disk in the h5 format.
    - Plots the training and validation loss and accuracy as a function of epoch number, and saves the plot to disk. The plot also includes the model summary.
    - Note: The function uses data augmentation techniques during training, including random rotations, width and height shifts, shearing, zooming, and horizontal flipping.
    - This function uses the TensorFlow, Keras, OpenCV, and matplotlib libraries.

    '''
            
    
    #from mahmud_ml.lenet import LeNet


    dataset=dataset
    model=model_out
    plot=plot
    

    # initialize the number of epochs to train for, initial learning rate, and batch size
    EPOCHS = EPOCHS
    INIT_LR = INIT_LR
    BS = BS

    # initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []

    # grab the image paths and randomly shuffle
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.seed(42)
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (28, 28))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "foggy" else 0
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width=28, height=28, depth=3, classes=2)
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(model_out, save_format="h5")

    #############

    

    # Load the model
    model = load_model(model_out)

    # # Print model summary
    # print("\nModel Summary:")
    #model.summary()

    import io
    import re

    import matplotlib.pyplot as plt

    # Save the summary to a string
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()

    # Preprocess the string to remove unwanted characters
    summary_string = re.sub('_+', '', summary_string)
    summary_string = re.sub('=+', '', summary_string)


    ###############

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(15,5))
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on fog/Not fog")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")


    # Add the summary string as a textbox
    # Add text to figure with a bounding box
    bbox_props = dict(boxstyle="round, pad=0.3", fc="white", ec="k", lw=2, alpha=0.6)
    plt.figtext(0.75, 0.5, summary_string, horizontalalignment='left', verticalalignment='center', fontsize=6, bbox=bbox_props)

    plt.tight_layout()

    plt.savefig(plot)

    plt.show()


from os import listdir, makedirs
from os.path import isdir, isfile, join
import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm


def classification(input_dir="dataset_imagery", trained_model="foggy_not_foggy.model"):
    """
    Classifies images in the specified directory using a trained model.

    Inputs:
    -------
        - input_dir (str, optional): Path to the directory containing the input images. Defaults to "dataset_imagery".
        - trained_model (str, optional): Path to the trained model file. Defaults to "foggy_not_foggy.model".

    

    Returns:
    --------
        - The function assumes that the input directory contains image files in JPG format.
        - The function uses a trained convolutional neural network model to classify the images.
        - It saves the classified images into separate directories based on their classification.

    
    """
    # Setting required file directories
    dir_list = ['filtered_images_noFog', 'filtered_images_Fog', 'ClearImages_daily']
    for directory in dir_list:
        if not isdir(directory):
            try:
                makedirs(directory)
            except OSError as e:
                raise OSError(f"Error creating directory '{directory}': {e}")

    No_Fogg_path = "filtered_images_noFog"
    Foggy_Path = "filtered_images_Fog"
    dailyimages = "ClearImages_daily"
    mypath = input_dir

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    images = np.empty(len(onlyfiles), dtype=object)

    # Load the trained convolutional neural network outside of loop
    model = load_model(trained_model)

    # Loop through each image and use tqdm for progress bar
    pbar = tqdm(range(len(onlyfiles)), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', colour='#00ff00', dynamic_ncols=True)

    for n in pbar:
        pbar.set_description(f"Processing image {n + 1}")
        image_path = join(mypath, onlyfiles[n])

        image = cv2.imread(image_path)
        orig = image.copy()

        # Pre-process the image for classification
        image = cv2.resize(image, (28, 28))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image.astype("float") / 255.0

        # Classify the input image
        (not_foggy, foggy) = model.predict(image)[0]
        label = "NotFoggy" if not_foggy > foggy else "Foggy"
        proba = not_foggy if not_foggy > foggy else foggy
        label = f"{label}: {proba * 100:.2f}%"

        cv2.putText(orig, label, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        im = Image.fromarray(orig)

        file = str(onlyfiles[n])
        position = file.index(".jpg")
        filename = file[:position]

        if proba > 0.95 and not_foggy > foggy:
            label = "Not Foggy"
            im.save(join(No_Fogg_path, f"{proba}-{filename}.jpg"))
            im.save(join(dailyimages, f"{filename}.jpg"))
        else:
            label = "Foggy"
            im.save(join(Foggy_Path, f"{proba}-{filename}.jpg"))

    print("\nNo more files left to process")  # Print final message on a new line
