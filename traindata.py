from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.metrics import categorical_accuracy
import io
import Augmentor
import os

def perbanyak_(ini, sebanyak_ini):
  source_dir = ini
  output_dir = "."
  p = Augmentor.Pipeline(source_directory=source_dir, output_directory=output_dir)
  p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=1)
  p.rotate(probability=0.7, max_left_rotation=13, max_right_rotation=13)
  p.zoom_random(probability=0.5, percentage_area=0.9)
  p.crop_random(probability=0.6, percentage_area=0.9)
  p.resize(probability=1.0, width=64, height=64)

  p.sample(sebanyak_ini)

# perbanyak_("dataset/train/sapi", 100)
# perbanyak_("dataset/train/belut", 100)
# perbanyak_("dataset/train/salmon", 100)
# perbanyak_("dataset/test/sapi", 10)
# perbanyak_("dataset/test/belut", 10)
# perbanyak_("dataset/test/salmon", 10)

def teslayer(teszz):
  size_img = 64

  model_gambar = Sequential()
  model_gambar.add(Conv2D(32, (3, 3), activation='relu', input_shape=(size_img,size_img,3)))
  model_gambar.add(Conv2D(32, (3, 3), activation='relu'))
  model_gambar.add(MaxPooling2D(pool_size=(2,2)))

  model_gambar.add(Conv2D(64, (3, 3), activation='relu', input_shape=(size_img,size_img,3)))
  model_gambar.add(Conv2D(64, (3, 3), activation='relu'))
  model_gambar.add(MaxPooling2D(pool_size=(2,2)))

  model_gambar.add(Conv2D(64, (3, 3), activation='relu', input_shape=(size_img,size_img,3)))
  model_gambar.add(Conv2D(64, (3, 3), activation='relu'))
  model_gambar.add(MaxPooling2D(pool_size=(2,2)))
  model_gambar.add(Flatten())
  model_gambar.add(Dense(128, activation='relu'))
  model_gambar.add(Dense(9, activation='softmax'))

  model_gambar.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['categorical_accuracy'])

  train = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
  test = ImageDataGenerator(rescale=1./255)

  train_data= train.flow_from_directory('dataset/train', target_size= (size_img,size_img), batch_size=32 , class_mode = 'categorical')
  test_data= test.flow_from_directory('dataset/test', target_size= (size_img,size_img), batch_size=32 , class_mode = 'categorical')

  # try:
  #   model_gambar.fit_generator(train_data, steps_per_epoch=5, epochs=200, validation_data=test_data, validation_steps=10)
  # except:
  #   pass
  # model_gambar.save_weights('azz.h5')

  model_gambar.load_weights('azz.h5')

  import matplotlib.pyplot as plt
  def switch_dict_key_values(this_dict):
    return dict((v,k) for k,v in this_dict.items())

  nama_train_data = switch_dict_key_values(train_data.class_indices)
  print(nama_train_data)

  imge = image.load_img(a,target_size=(size_img,size_img))
  imge = image.img_to_array(imge)
  imge = np.expand_dims(imge, axis=0)

  hasil = model_gambar.predict_classes(imge)
  hasil = nama_train_data[hasil[0]]
  
  return hasil

a = 'images15.jpg'
hasil = teslayer(a)
print(hasil)