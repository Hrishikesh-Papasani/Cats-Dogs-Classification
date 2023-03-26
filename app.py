#Main flask application file.


# Import required modules
import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

import tensorflow as tf
# from keras.preprocessing.image import load_img, img_to_array
from keras.utils import load_img, img_to_array


# Initialize Flask application
app = Flask(__name__,template_folder='/Users/hrishi/Desktop/Python/DA (imp)/Course final projects/flask_cnn/templater')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load trained model
model = load_model('final_model_cats_dogs.h5', compile=False)
#Compiling the model
model.compile(loss='categorical_crossentropy',
  optimizer='rmsprop',metrics=['accuracy'])

# Define helper function to check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define main route for index.html
@app.route('/')
def index():
    return render_template('index.html')

# Define route for file upload
@app.route('/upload', methods=['POST'])
def upload():
        file = request.files['file']

        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='Please upload a file')

        # Check if file has a valid extension
        if not allowed_file(file.filename):
            return render_template('index.html', message='Invalid file type')

        # Save file to disk
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Predict the class of the uploaded image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        img = image.load_img(img_path, target_size=(128, 128))
        im=np.expand_dims(im,axis=0)
        im=np.array(im)
        im=im/255
        prediction= np.argmax(model.predict([im])[0], axis=-1)

        # Display the prediction result
        if prediction == 1:
            prediction = 'Dog'
        else:
            prediction = 'Cat'

        # Return the prediction by redirecting to the predict route
        return redirect(url_for('predict', filename=filename, prediction=prediction))


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']

    # Save the file to the uploads folder
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Load the image
    im = load_img(file_path, target_size=(128, 128))

    # Preprocess the image
    im=np.expand_dims(im,axis=0)
    im=np.array(im)
    im=im/255
    prediction= np.argmax(model.predict([im])[0], axis=-1)

    # Convert the prediction to a string label
    prediction = "Dog" if prediction == 1 else "Cat"

    # Pass the filename and prediction variables to the template
    return render_template('predict.html', filename=file.filename, prediction=prediction)



# Run the application
if __name__ == '__main__':
    app.run(debug=True)
