# !python train.py -p "3" -e "1" -lr '0.0005'


from skimage import io
import numpy as np
import keras
import tensorflow as tf
from keras.layers import *
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import gc
import tqdm
import dataloader
import blocks

import argparse

parser = argparse.ArgumentParser(' Parameters for training')
parser.add_argument("-p", "--patchSize", required=True, type=int, default=7)
parser.add_argument("-e", "--epochs", required=True, type=int, default=1000)
parser.add_argument("-lr", "--learning_rate", required=True, type=float, default=0.00005)
parser.add_argument("-emb", "--embedding", required=True, type=int, default=128)

args = parser.parse_args()


print('loading data')
training_patch, y_train, testing_patch, y_test = dataloader.gen_data(args.patchSize)
print('dataset shape', training_patch.shape)
print('running for ', args.epochs, ' epochs ! with LR ===> ', args.learning_rate)

classes = len(np.unique(y_train))

h = Input((args.patchSize,args.patchSize,144))
l = Input((args.patchSize,args.patchSize,1))

h1f = blocks.FilterBlock(args.embedding)(h)
l1f = blocks.FilterBlock(args.embedding)(l)

concat = concatenate([h1f,l1f])

for i in range(3):
  h1 = blocks.SelfAttention(args.embedding)(h1f)
  l1 = blocks.SelfAttention(args.embedding)(l1f)

  h1 = blocks.CrossAttn.cal(h1,l1,args.embedding)
  l1 = blocks.CrossAttn.cal(l1,h1,args.embedding)

  h1_co = blocks.CrossOut(args.embedding)(h1)
  l1_co = blocks.CrossOut(args.embedding)(l1)

  h1f = Add()([h1f,h1_co])
  l1f = Add()([l1f,l1_co])
  concat = concatenate([concat,h1f,l1f])

classifier = GlobalAveragePooling2D()(concat)
classifier = Dense(512, activation='relu')(classifier)
classifier = Dense(classes, activation = 'softmax')(classifier)


model = keras.Model((h,l),(classifier))

model.compile(optimizer = tf.keras.optimizers.Adam(args.learning_rate), loss = 'sparse_categorical_crossentropy', metrics = 'acc')

for i in range(args.epochs):
  gc.collect()
  history = model.fit((training_patch[:,:,:,:-1],training_patch[:,:,:,-1]) , y_train, batch_size=32, epochs = 1, validation_data=((testing_patch[:,:,:,:-1],testing_patch[:,:,:,-1]), y_test))




# save model fuctions can be added as per own wish
