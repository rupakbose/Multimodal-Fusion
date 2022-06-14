from tensorflow.keras import layers
from keras.layers import *
import tensorflow as tf
import keras


class FilterBlock(keras.layers.Layer):
    def __init__(self, embedding):
        super(FilterBlock, self).__init__()
        self.embedding = embedding
        self.conv1 = Conv2D(self.embedding,3,padding = 'same')
        self.layer_norm = LayerNormalization()
        

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.layer_norm(x)
        x = tf.nn.relu(x)
        return x


class SelfAttention(keras.layers.Layer):
    def __init__(self, embedding):
        super(SelfAttention, self).__init__()
        self.embedding = embedding
        self.conv1 = Conv2D(self.embedding,3,padding = 'same')
        

    def call(self, inputs):
        x = self.conv1(inputs)
        x1 = Reshape((-1,self.embedding))(x)
        x = tf.matmul(x1,x1,transpose_b=True)
        x = tf.nn.softmax(x,axis = -1)/(self.embedding**0.5)
        x = tf.matmul(x,x1)
        x = Reshape((inputs.shape[-2], inputs.shape[-3], -1))(x)
        x = Multiply()([x,inputs])
        return x


class CrossOut(keras.layers.Layer):
    def __init__(self, embedding):
        super(CrossOut, self).__init__()
        self.embedding = embedding
        self.conv1 = Conv2D(self.embedding,3,padding = 'same')
        self.layer_norm = LayerNormalization()
        

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.layer_norm(x)
        x = tf.nn.relu(x)
        return x

        
class CrossAttn():
  def __init__(self, key, value, emb):
    self.key = key
    self.value = value
    self.emb = emb

  def cal( key, value, emb):
    x = tf.matmul(key,key,transpose_b=True)
    x = tf.nn.softmax(x,axis = -1)
    x = x /(emb**0.5)
    x = tf.matmul(x,value)
    x = Reshape((value.shape[-2], value.shape[-3], -1))(x)
    x = Multiply()([x, key])
    return x

