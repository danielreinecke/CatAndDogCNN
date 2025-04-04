import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import os

DATADIR = r"C:\Users\danie\OneDrive\Desktop\Vs Projects\AI stuff\Convolutions\kagglecatsanddogs_5340\PetImages" #path to dataset
CATEGORIES = ['Dog', 'Cat'] #defined catagories
IMG_SIZE = 100

#split the training data into parts
def create_training_data():
    training_data = []
    for catagory in CATEGORIES:
        path = os.path.join(DATADIR, catagory)  #path to the diffrent files in the training set
        print(f"creating: {catagory}")
        class_num = CATEGORIES.index(catagory)  #sets dog as value 0 and cat as value 1
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)    #read in the images as a array in diffrent shades of grey (makes it smaller and colors is prob not important)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))     #makes the images smaller and easier to read in
                #plt.imshow(new_array, cmap = "grey")
                #plt.show()
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    #shuffle the data around to add randomness to traning
    import random
    print(len(training_data))
    random.shuffle(training_data)

    X = []  #features
    y = []  #labels

    for features, label in training_data:
        X.append(features)
        y.append(label)

    x = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #the one is for its grey scale and is splitting the data apart
    return X, y

#saves data using pickle
def save_data(X,y):
    pickle_out = open('X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()

#loads created data using pickle
def load_data():
    pickle_in = open('X.pickle', 'rb')
    X = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open('y.pickle', 'rb')
    y = pickle.load(pickle_in)
    pickle_in.close()

    return X,y

if __name__ == '__main__':
    X,y = create_training_data()
    #save_data(X,y)
    x_train,y_train = load_data()

    #actual CNN
    import keras