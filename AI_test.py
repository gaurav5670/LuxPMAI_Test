import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report

labels=['apple','banana']

img_size=224

def train_cls_model(x_train,y_train,x_val,y_val):
        #Create Model
        model = Sequential()
        model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(img_size, img_size, 3)))
        model.add(MaxPool2D())

        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(64, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        model.summary()
        opt = Adam(lr=0.001)
        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(x_train,y_train,epochs=100,validation_data=(x_val,y_val))

        predictions = model.predict_classes(x_val)
        predictions = predictions.reshape(1, -1)[0]
        print(classification_report(y_val, predictions, target_names=['apple (Class 0)','unknown (Class 1)']))


def read_data(dir_path):
    train_dir=os.path.join(dir_path,'train')
    test_dir=os.path.join(dir_path,'test')
    train_data=[]
    test_data=[]
    for label in labels:
        class_id = labels.index(label)
        label_train=os.path.join(train_dir,label)
        label_test=os.path.join(test_dir,label)

        #read training images

        for img in os.listdir(label_train):
            img_path=os.path.join(label_train,img)
            frame=cv2.imread(img_path)
            frame=cv2.resize(frame,(img_size,img_size)) #resize image to desire input size
            train_data.append([frame,class_id])

        for img in os.listdir(label_test):
            img_path = os.path.join(label_test, img)
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, (img_size, img_size))  # resize image to desire input size
            test_data.append([frame,class_id])

    return np.array(train_data),np.array(test_data)

if __name__=="__main__":
    train_data,test_data=read_data("./data/")
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for feature, label in train_data:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test_data:
        x_val.append(feature)
        y_val.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)
    train_cls_model(x_train,y_train,x_val,y_val)

