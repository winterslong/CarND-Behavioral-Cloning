import cv2
import csv
import numpy as np
import os

simu_data_path = './data'

train_mode = "nVidia"
save_model_file = 'model_nv_optdata_e2.h5'
nepochs = 2

#train_mode = "LeNet" 
#save_model_file = 'model_lenet_optdata_e2.h5'
#nepochs = 2

def getLinesFromDrivingLogs(dataPath, skipHeader=False):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    lines = []
    driving_log_file = dataPath + '/driving_log.csv'
    with open(driving_log_file) as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def findImages(dataPath):
    """
    Finds all the images needed for training on the path `dataPath`.
    Returns `([centerPaths], [leftPath], [rightPath], [steering])`
    """
    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    dataDirectories.sort()
    
    centerTotal = []
    leftTotal = []
    rightTotal = []
    steeringTotal = [] 
    for directory in dataDirectories:
        lines = getLinesFromDrivingLogs(directory, True)
        center = []
        left = []
        right = []
        steering = []
        for line in lines:
            steering.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
            
        centerTotal.extend(center)
        leftTotal.extend(left)
        rightTotal.extend(right)
        steeringTotal.extend(steering)
        print("directory=", directory)
        print("    center=", len(center))
        print("    left  =", len(left))
        print("    right =", len(right))
        print("    steerings =", len(steering))
    return (centerTotal, leftTotal, rightTotal, steeringTotal)

def extendImages(center, left, right, steering, correction):
    """
    Extend the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [steerings])
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    
    steerings = []
    steerings.extend(steering)
    steerings.extend([x + correction for x in steering])
    steerings.extend([x - correction for x in steering])
    return (imagePaths, steerings)



# Reading images locations.
centerPaths, leftPaths, rightPaths, steerings = findImages(simu_data_path)
print('Total:')
print('centerPaths : {}'.format( len(centerPaths)))
print('leftPaths   : {}'.format( len(leftPaths)))
print('rightPaths  : {}'.format( len(rightPaths)))
print('steerings   : {}'.format( len(steerings)))
print()

imagePaths, steerings = extendImages(centerPaths, leftPaths, rightPaths, steerings, 0.2)
print('Total Images   : {}'.format( len(imagePaths)))
print('Total steerings: {}'.format( len(steerings)))


import sklearn

def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `steering`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, steering in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(steering)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(steering*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Conv2D
from keras.layers import Dropout, Activation
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def nVidiaModel():
    """    Creates nVidea Autonomous Car Group model   """
    print("######### In nVidia Model ######### ")
    model = createPreProcessingLayers()     
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

def LeNet():
    print("######### In LeNet Model #########")
    model = createPreProcessingLayers()
    model.add(Conv2D(6, (5, 5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())    
    model.add(Flatten())
    model.add(Dense(120))    
    model.add(Dense(84))
    model.add(Dropout(0.5))
    model.add(Dense(1))    
    model.summary()    
    return model



# Splitting train_data and creating generators.
from sklearn.model_selection import train_test_split

train_data = list(zip(imagePaths, steerings))
X_train, X_validation = train_test_split(train_data, test_size=0.2)
print('Train samples: {}'.format(len(X_train)))
print('Validation samples: {}'.format(len(X_validation)))

#"""
train_generator = generator(X_train, batch_size=32)
validation_generator = generator(X_validation, batch_size=32)

# Model creation
if (train_mode == "LeNet"):
    model = LeNet()
else:  #"nVidia"
    model = nVidiaModel()
    
# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
                        samples_per_epoch= len(X_train), 
                        validation_data=validation_generator, 
                        nb_val_samples=len(X_validation), 
                        nb_epoch=nepochs, 
                        verbose=1)

model.save(save_model_file)
#model.save('model_lenet.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
"""
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
"""
