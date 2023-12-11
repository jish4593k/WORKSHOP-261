import urllib.request
import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from PIL import Image
import requests
from io import BytesIO


execution_path = os.getcwd()




def download_and_extract_dataset():
    if not os.path.exists(ZIP_FILE_PATH):
        print("Downloading trafficnet_dataset_v1.zip")
        with urllib.request.urlopen(SOURCE_PATH) as response, open(ZIP_FILE_PATH, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(execution_path)
        print("Dataset downloaded and extracted successfully.")

# Function to train the model
def train_traffic_net():
    download_and_extract_dataset()

    # Build a simple model using ResNet50 as a base
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # You need to adapt the training code based on your dataset structure and requirements
    # Here is a placeholder for demonstration purposes:
    # train_datagen = ...
    # train_generator = ...
    # model.fit_generator(train_generator, epochs=10, steps_per_epoch=100)

    print("Model training completed.")


def run_predict():

    model = tf.keras.models.load_model("trafficnet_resnet_model_ex-055_acc-0.913750.h5")


    img_path = "images/1.jpg"
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0

   
    predictions = model.predict(img_array)
    classes = ['class1', 'class2', 'class3', 'class4']  # Update with your actual class labels

    for i in range(len(classes)):
        print(f"{classes[i]}: {predictions[0][i]}")
