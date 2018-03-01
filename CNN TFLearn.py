#import modules
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d
import os
import cv2
from random import shuffle
import numpy as np


#put the location of train and test data here
train_image_loc= r'C:\Users\Gebruiker\Documents\Github\1 CNN with TFLearn Cat vs Dog Dataset\train'
test_image_loc = r'C:\Users\Gebruiker\Documents\Github\1 CNN with TFLearn Cat vs Dog Dataset\test1'

#parameters
image_size = 60
LR = 0.001

#model name
MODEL_NAME = 'Dog_V_Cat_Classifier'
#'''
#Data Preprocessing
#Turn data into a one-hot array
def one_hot(image):
    label = image.split('.')
    label = label[0]
    if label == 'cat':
        return [1,0]
    elif label == 'dog':
        return [0,1]

#prepare the training data
def prep_train():
    train_data = []
    nr = 1
    for image in os.listdir(train_image_loc):
        label = one_hot(image)
        path = os.path.join(train_image_loc,image)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size,image_size))
        train_data.append([np.array(img),np.array(label)])
        print('Preparing training image ',nr)
        nr += 1
        
    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data

#prepare the testing data
def prep_test():
    test_data = []
    nr = 1
    for image in os.listdir(test_image_loc):
        path = os.path.join(test_image_loc,image)
        img_num = image.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size,image_size))
        test_data.append([np.array(img), img_num])
        print('Preparing testing image ',nr)
        nr += 1
        
    shuffle(test_data)
    np.save('test_data.npy', test_data)
    return test_data


#'''


'''


#Convolutional Neural Network
cnn = input_data(shape=[None, image_size, image_size, 1], name='input')
cnn = conv_2d(cnn, 40, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 80, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 160, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 80, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 40, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = fully_connected(cnn, 1024, activation='relu')
cnn = dropout(cnn, 0.9)
cnn = fully_connected(cnn, 2, activation='softmax')
cnn = regression(cnn, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn, tensorboard_dir='log')





#prepare the dataset for training
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')
#train_data = prep_train()
#test_data = prep_test()

train = train_data[:-1000]
test = train_data[-1000:]

X = []
for image in train:
    X.append(image[0])

Y = []
for image in train:
    Y.append(image[1])

test_x = []
for image in test:
    test_x.append(image[0])

test_y = []
for image in test:
    test_y.append(image[1])

X      = np.array(X).reshape(-1,image_size,image_size,1)
test_x = np.array(test_x).reshape(-1,image_size,image_size,1)


#Fit the model
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#save the model
model.save(MODEL_NAME)
'''

#Writing the Kaggle submission file
'''
with open('submission_file.csv','w') as f:
    f.write('id,label\n')
            
with open('submission_file.csv','a') as f:
    for data in test_data:
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(image_size,image_size,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))
'''

#model.load(MODEL_NAME)
#score = model.evaluate(test_x, test_y)



















