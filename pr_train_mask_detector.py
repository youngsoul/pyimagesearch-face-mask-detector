from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to output face mask detector model")
args = vars(ap.parse_args())

# initialize the intial learning rate, the number of epochs,
# to train for, and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# grab the list of images in our dataset directory, then initialize
# the list of data ( i.e. images ) class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	# path like:
	# .../dataset/with_mask/0-with-mask.jpg
	# -2 = with_mask
	label = imagePath.split(os.path.sep)[-2]

	# load the input image (224x224) and preprocess it
	# use the Keras routing to load and resize the image
	image = load_img(imagePath, target_size=(224,224))
	image = img_to_array(image)
	# use mobilenet_v2 preprocess routine
	image = preprocess_input(image)

	# update the data and labels list
	data.append(image)
	labels.append(label)

# convert the data and labels into numpy arrays
# models, and ml routines prefer numpy arrays
data = np.array(data, dtype='float32')
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels_bin = lb.fit_transform(labels)
# labels_bin.shape = 1376,1
# print(labels_bin.shape)
# print(lb.classes_) # ['with_mask', 'without_mask']
# print(list(zip(labels_bin, labels)))
labels_cat = to_categorical(labels_bin)
# labels_cat.shape = (1376,2)
# where the
# print(labels_cat.shape)

# partition the data into training and testing splits using 80% of the
# data for training and the remaining 20% for testing
X = data
y = labels_cat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# construct the training image generator for data augmentation
# The imageDataGenerator will allow for on-the-fly mutations to the
# input set of images to improve generatlization
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest"
)

# prepare the MobileNetV2 for Fine-Tuning:
# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
# include_top=False
# will remove the fully connected layers so that we can add a new
# un-trained fully connected layer
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

# construct the head of the model that will be placed on top of the
# base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model ( this will become
# the actual model we will train )
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freezse them so they will
# NOT be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("compiling model")
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the head of the network
# this will not retrain the convolutional layers
print("Training head...")
H = model.fit(
	aug.flow(X_train, y_train, batch_size=BS),
	steps_per_epoch=len(X_train)//BS,
	validation_data=(X_test, y_test),
	validation_steps=len(X_test)//BS,
	epochs=EPOCHS
)

# make predictions on the testing set
print("Evaluating Network...")
predIdxs = model.predict(X_test, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
# in other words was it index=0 ( with_mask ) that had the highest
# probability, or was it index=1 (without_mask) that had the
# highest probability
predIdxs = np.argmax(predIdxs, axis=1)

# recall that y_test is a one-hot encoded array
# so we have to perform argmax on this to match the predIdxs

# show a nicely formatted classificaiton report
print(classification_report(y_test.argmax(axis=1), predIdxs, target_names=lb.classes_))

# serialize the model to disk
print("Saving mask detector model...")
model.save(args['model'], save_format='h5')

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])