import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing import image

print("TF Version : ", tf.__version__)  # 2.4.0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import argparse

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--csv',
                    type=str,
                    help='path to csv file')
parser.add_argument('--img_dir',
                    type=str,
                    help='path to image directory')

args = parser.parse_args()


def main(__unused__):
	img_width   = 350
	img_height  = 350
	X = []
	data = pd.read_csv(args.csv)
	
	for i in tqdm(range(data.shape[0])):
		path = args.img_dir + "/" + data['Id'][i] + '.jpg'
		img = image.load_img(path, target_size=(img_width, img_height, 3))
		img = image.img_to_array(img)
		img = img / 255.0
		X.append(img)
	
	X = np.array(X)
	y = data.drop(['Id', 'Genre'], axis=1)
	y = y.to_numpy()
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.15)
	
	# Define CNN model
	model = Sequential()
	model.add(Conv2D(16, (3, 3), activation='relu', input_shape=X_train[0].shape))
	model.add(BatchNormalization())
	model.add(MaxPool2D(2, 2))
	model.add(Dropout(0.3))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(2, 2))
	model.add(Dropout(0.3))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(2, 2))
	model.add(Dropout(0.4))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D(2, 2))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(25, activation='sigmoid'))
	
	model.summary()
	
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))
	model.save('model.h5')


if __name__ == "__main__":
	tf.compat.v1.app.run()
