import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import image_utils

#openCV character segmentation 來切字元並且盼獨字元數
#make a constraint to slip 驗證碼 如 1 and I

epochs = 20       #訓練的次數
img_rows = None   #驗證碼影像檔的高
img_cols = None   #驗證碼影像檔的寬
digits_in_img = 5 #驗證碼影像檔中有幾位數
x_list = list()   #存所有驗證碼數字影像檔的array
y_list = list()   #存所有的驗證碼數字影像檔array代表的正確數字
x_train = list()  #存訓練用驗證碼數字影像檔的array
y_train = list()  #存訓練用驗證碼數字影像檔array代表的正確數字
x_test = list()   #存測試用驗證碼數字影像檔的array
y_test = list()   #存測試用驗證碼數字影像檔array代表的正確數字
    
def split_digits_in_img(img_array, x_list, y_list):
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(img_filename[i])

img_filenames = os.listdir('training')
 
for img_filename in img_filenames:
    if '.jpg' not in img_filename:
        continue
    img = image_utils.load_img('training/{0}'.format(img_filename), color_mode='grayscale')
    img_array = image_utils.img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, x_list, y_list)   

y_list = pd.get_dummies(y_list)
 
len_y = len(y_list)
len_x = len(x_list)
print([len_x,len_y])

print(y_list)
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list)
#print([x_train, x_test, y_train, y_test ])

#print( len(x_train) )
#print( len(y_train) )
#print(img_rows, img_cols // digits_in_img)


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols//digits_in_img,1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=30, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), batch_size=digits_in_img, epochs=epochs, validation_data=(np.array(x_test), np.array(y_test)))
 
train_loss, train_accuracy = model.evaluate(np.array(x_train), np.array(y_train), batch_size=digits_in_img)
test_loss, test_accuracy = model.evaluate(np.array(x_test), np.array(y_test), batch_size=digits_in_img)

print('train loss:', train_loss)
print('train accuracy:', train_accuracy)

print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
