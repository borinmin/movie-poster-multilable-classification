from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--img',
                    type=str,
                    default='sample/bloodshit.jpg',
                    help='Give image path for inferencing')
args = parser.parse_args()


if __name__=="__main__":
  
  img_width   =   350
  img_height  =   350
  data = pd.read_csv('classes.csv')
  img = image.load_img(args.img, target_size=(img_width, img_height, 3))
  
  img = image.img_to_array(img)
  img = img/255.0
  img = img.reshape(1,img_width,img_height,3)
  
  classes = data.columns[2:]
  
  model = tf.keras.models.load_model('model.h5',compile=False)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
  y_prob = model.predict(img)
  top3 = np.argsort(y_prob[0])[:-4:-1] # get top 3
  
  
  print("Possible Movie Genres are: ")
  for i in range(3):
    print("\tGenre:",classes[top3[i]])