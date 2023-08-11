import numpy as np
from tensorflow import math
from keras.layers import Layer
import tensorflow as tf
import seaborn as sns
from matplotlib.pyplot import savefig

class Cross_Attention_Layer(Layer):
    
    def __init__(self, source_size = 768, target_size=768, d_k = 128, d_v = 64, **kwargs):
        self.source_size = source_size
        self.target_size = target_size
        self.d_k = d_k
        self.d_v = d_v
        super(Cross_Attention_Layer,self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.w_q = self.add_weight(name = "Wq",shape=(self.d_k,self.source_size),initializer=tf.keras.initializers.RandomNormal(stddev=0.01),trainable=True)
        self.w_k = self.add_weight(name = "Wk",shape=(self.d_k,self.target_size),initializer=tf.keras.initializers.RandomNormal(stddev=0.03),trainable=True)
        self.w_v = self.add_weight(name = "Wv",shape=(self.d_v,self.target_size),initializer=tf.keras.initializers.RandomNormal(stddev=0.5),trainable=True)        
        self.w_output = self.add_weight(name = "Woutput",shape=(1,self.d_v),initializer=tf.keras.initializers.RandomNormal(stddev=0.1),trainable=True)
        
    def call(self,inputs):
        [input1,input2] = inputs
        
        print("input_shape",input1,input2)
        if tf.reduce_sum(input2) == 0:
            return input1[0,:,:]
        
        Query = input1[0,:,:]@tf.transpose(self.w_q)
        Key = input2[0,:,:]@tf.transpose(self.w_k)
        Value = input2[0,:,:]@tf.transpose(self.w_v)
        
        '''Q = inputs[0,:,:]@tf.transpose(self.w_q)
        K = inputs[1,:,:]@tf.transpose(self.w_k)
        V = inputs[1,:,:]@tf.transpose(self.w_v)'''
        
        dot_prod = tf.tensordot(Query,tf.transpose(Key),axes=1)/np.sqrt(128)
        softmax_prod = math.softmax(dot_prod,axis = 1)
        output = (softmax_prod@Value)@tf.transpose(self.w_output)

        
        return (input1[0,:,:]*(1-output))


if __name__ == "__main__":
    word_emb1 = tf.zeros(shape=(1,150,768))
    '''word_emb2 = np.random.normal(0,0.2,size=(1,70,768))
    print(type(word_emb1[0:0]))
    word_emb1 = tf.convert_to_tensor(word_emb1)
    word_emb2 = tf.convert_to_tensor(word_emb2)
    #print(crossattn(word_emb1,word_emb2).shape)
    
    input = tf.keras.layers.Input(shape = (1,70,768))
    input_action = tf.keras.layers.Input(shape = (1,70,768))
    
    concated = tf.concat(values=[input,input_action],axis = 0)
    
    cal = Cross_Attention_Layer(768,768,128,64)(concated)
    
    dns = tf.keras.layers.LSTM(100,input_shape = (70,768),activation = "sigmoid",return_sequences=True)(cal)
    dns = tf.cast(dns,dtype="float64")
    print(dns.shape)
    dns = tf.reshape(dns,shape=(1,)+dns.shape[1:])
    
    print(concated.shape)
    dns1 = tf.keras.layers.Dense(1,activation = "sigmoid")()
    model = tf.keras.Model([input,input_action],outputs = dns1)
    print(model.predict(word_emb1).shape)'''
    