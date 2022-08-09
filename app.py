from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.metrics import categorical_accuracy

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

model_gambar.load_weights('azz.h5')

import matplotlib.pyplot as plt

def switch_dict_key_values(this_dict):
  return dict((v,k) for k,v in this_dict.items())

nama_train_data = switch_dict_key_values(train_data.class_indices)
print(nama_train_data)

app = Flask(__name__, template_folder='template') 
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#rumus untuk memanggil library flask
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('Tidak ada file')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada gambar untuk diupload')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        test = 'static/uploads/'+filename
        imge = image.load_img(test,target_size=(size_img,size_img))
        imge = image.img_to_array(imge)
        imge = np.expand_dims(imge, axis=0)

        hasil = model_gambar.predict_classes(imge)

        predict = nama_train_data[hasil[0]]

        return render_template('predict.html', filename= filename, predict_makanan=predict)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/'+filename), code=301)
    
if __name__ == "__main__":
    app.run(debug=True) #debug=True untuk saat ada penggantian file tidak perlu run ulang


        