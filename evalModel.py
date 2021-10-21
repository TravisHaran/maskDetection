from tensorflow.keras.models import load_model
import argparse
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical


def evalModel(myModel):
    model = load_model(myModel)

    datasetPath = "./dataset"

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, default=datasetPath,
                    help="path to input dataset")
    ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="path to output face mask detector model")
    args = vars(ap.parse_args())

    # Initialize batch size
    BS = 32

    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)

    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    evalReport = classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_)
    print(myModel + " Evaluation Report")
    return print(evalReport)


while True:
    modelChoice = input("Enter 1 if you would like to evaluate the original model\n"
                        "Enter 2 if you would like to evaluate the optimized model\n"
                        "Enter 3 if you would like to evaluate both models")
    if modelChoice == "1":
        print("default mask detection model selected")
        evalModel("mask_detector.model")
        break
    elif modelChoice == "2":
        print("optimized mask detection model selected")
        evalModel("opt_mask_detection_delta_1.h5")
        break
    elif modelChoice == "3":
        print("both models selected")
        evalModel("mask_detector.model")
        evalModel("opt_mask_detection_delta_1.h5")
        break
    else:
        print("Incorrect input!\n")
