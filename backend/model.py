import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import scipy
import pickle

class ImageClassifier:
    def __init__(self):
        pass

    def build_model(self):
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True)
        self.training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

        test_datagen = ImageDataGenerator(rescale = 1./255)
        self.test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

        self.cnn = tf.keras.models.Sequential()
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.cnn.add(tf.keras.layers.Flatten())
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        self.cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.cnn.fit(x = self.training_set, validation_data = self.test_set, epochs = 25)

        self.pickle_model()
            
    def pickle_model(self):
        self.cnn.save('models/dog_cat_model.h5')
    
    def unpickle_model(self):
        return load_model('models/dog_cat_model.h5')
    
    def classifyImage(self, filepath):
        model = self.unpickle_model()
        test_image = image.load_img(filepath, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
        return prediction

if __name__ == '__main__':
    classifier = ImageClassifier()
    classifier.build_model()
    # print(classifier.classifyImage('dataset/test/1.jpg'))
