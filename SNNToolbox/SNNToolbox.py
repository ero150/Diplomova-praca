import os
import time
import numpy as np
#zmena
import keras
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, \
    BatchNormalization, Activation
from keras.datasets import mnist
from keras.utils import np_utils

from snntoolbox.bin.run import main
from snntoolbox.bin import gui
from snntoolbox.utils.utils import import_configparser
#------------------------Import kniznic-------------------------------

#--------------------Definicia a vytvorenie precinku---------------------------

# Definicia priecinku kde sa ulozia model a vystupy
directory = "SnnToolbox_CNN"

# Cesta k priecinku
parent_dir = "Cesta"

# Vytvorenie plnej cesty
path_wd = os.path.join(parent_dir, directory)

# Vytvorenie priecinku
os.makedirs(path_wd)

# Priprava databazy #

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Nacitanie MNIST

# Normalizacia vzoriek - zrychluje vypocet
x_train = x_train / 255
x_test = x_test / 255


axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
x_train = np.expand_dims(x_train, axis)
x_test = np.expand_dims(x_test, axis)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Ulozenie databazy
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test)
np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train[::10])

# Vytvorenie ANN
input_shape = x_train.shape[1:]
input_layer = Input(input_shape)

layer = Conv2D(filters=32,
               kernel_size=(5, 5))(input_layer)
layer = Activation('relu')(layer)

layer = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation='relu')(layer)
layer = MaxPooling2D()(layer)
layer = Dropout(0.25)(layer)
layer = Flatten()(layer)
layer = Dense(units=128, activation='relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(units=10, activation='softmax')(layer)

model = Model(input_layer, layer)

model.summary()

model.compile('adam', 'categorical_crossentropy', ['accuracy'])

# Natrenovanie modelu
model.fit(x_train, y_train, batch_size=8, epochs=100, verbose=2,
          validation_data=(x_test, y_test))

# Ulozenie modelu
model_name = 'mnist_cnn'
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

# SNNToolbox Konfiguracia
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,             # Cesta k modelu
    'dataset_path': path_wd,        # Cesta k datasetu.
    'filename_ann': model_name      # Nazov modelu
}

config['tools'] = {
    'evaluate_ann': True,           # Otestovanie ANN na datasete pred konverziou
    'normalize': True,              # Normalizacia vah
}

config['simulation'] = {
    'simulator': 'brian2',          # Vyber vystupu
    'duration': 50,                 # Cas simulacie
    'num_to_test': 5,               # Pocet vzoriek
    'batch_size': 1,                # Velkost vzorky
    'dt': 0.1
}

config['input'] = {
    'poisson_input': False
}

config['output'] = {
    'plot_vars': {                  #Vykreslenie grafov
        'spiketrains',
        'spikerates',
        'activations',
        'correlation',
        'v_mem',
        'error_t'}
}

# Ulozenie konfiguracneho suboru
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# Spustenie SNNToolbox
main(config_filepath)
