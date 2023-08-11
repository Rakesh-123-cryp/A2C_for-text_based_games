import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from transformers import DistilBertTokenizer, TFDistilBertModel
import pytreebank
import spacy
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class bertlayer(tf.keras.layers.Layer):
    def __init__(self,size,**kwargs):
        self.size = size
        self.model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
        super(bertlayer,self).__init__(**kwargs)
    
    def call(self,inputs):
        inputs[0] = tf.convert_to_tensor(inputs[0],dtype=tf.int32)
        inputs[1] = tf.convert_to_tensor(inputs[1],dtype=tf.int32)
        

        distilbert = self.model(inputs[0],attention_mask = inputs[1])[0]
           
        return distilbert
           
class sentiment:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tokeniser = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
    def get_lines(self):
        data = pytreebank.import_tree_corpus("archive (5)/SST2-Data/SST2-Data/trainDevTestTrees_PTB/trees/train.txt")
        sentences = [item.to_labeled_lines()[0][1].replace("``",'""').replace("''",'""') for item in data]
        words = [self.tokeniser(sentence,return_tensors="np",padding="max_length",max_length=100) for sentence in sentences]
            
        words_id_attn = [tf.constant([sentence.get("input_ids"),sentence.get("attention_mask")]) for sentence in words]
        words_id_attn = tf.convert_to_tensor(words_id_attn)

        sentiments = [item.to_labeled_lines()[0][0] for item in data]
        sentiments = tf.constant(sentiments)
        self.max_size = len(words)
        
        cross_val = pytreebank.import_tree_corpus("archive (5)/SST2-Data/SST2-Data/trainDevTestTrees_PTB/trees/dev.txt")
        cv_1 = [item.to_labeled_lines()[0][1].replace("``",'""').replace("''",'""') for item in cross_val]
        cv_2 = [item.to_labeled_lines()[0][0] for item in cross_val]
        cv_2 = tf.constant(cv_2)
        
        cv_1 = [self.tokeniser(sentence,return_tensors="np",padding="max_length",max_length=100) for sentence in cv_1]#self.tokeniser.encode_plus(cv_1,padding='max_length',max_length=50,return_tensors="tf")
        cv_1 = tf.convert_to_tensor([tf.constant([sentence.get("input_ids"),sentence.get("attention_mask")]) for sentence in cv_1])
        cross_validation = []
        cross_validation.append(cv_1)
        cross_validation.append(cv_2)
        
        test = pytreebank.import_tree_corpus("archive (5)/SST2-Data/SST2-Data/trainDevTestTrees_PTB/trees/test.txt")
        t_1 = [item.to_labeled_lines()[0][1].replace("``",'""').replace("''",'""') for item in test]
        t_2 = [item.to_labeled_lines()[0][0] for item in test]
        t_2 = tf.constant(t_2)
        
        t_1 = [self.tokeniser(sentence,return_tensors="np",padding="max_length",max_length=100) for sentence in t_1]#self.tokeniser.encode_plus(t_1,padding='max_length',max_length=50,return_tensors="tf")
        t_1 = tf.convert_to_tensor([tf.constant([sentence.get("input_ids"),sentence.get("attention_mask")]) for sentence in t_1])
        test_set = []
        test_set.append(t_1)
        test_set.append(t_2)
        
        self.test_size = len(t_1)
        self.cv_size = len(cv_1)

        return words_id_attn,sentiments,cross_validation,test_set
    
    def build_model(self):
        reciever_ids = tf.keras.layers.Input(shape=(100),dtype=tf.int32)
        reciever_attn = tf.keras.layers.Input(shape=(100),dtype=tf.int32)
        
        distilbert = bertlayer(size=(self.max_size,100,768))([reciever_ids,reciever_attn])
        
        lstm_1 = tf.keras.layers.Bidirectional(LSTM(100,return_sequences=True),input_shape=(100,768))(distilbert)
        lstm_1 = tf.reduce_mean(lstm_1,axis = 1)
        print(lstm_1.shape)
        
        drop = tf.keras.layers.Dropout(0.01)(lstm_1)
        feed_forward_1 = tf.keras.layers.Dense(128,activation="sigmoid")(drop)
        feed_forward_2 = tf.keras.layers.Dense(32,activation="sigmoid")(feed_forward_1)
        feed_forward_3 = tf.keras.layers.Dense(5,activation="softmax")(feed_forward_2)
        model = tf.keras.Model([reciever_ids,reciever_attn],feed_forward_3)
        model.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics="accuracy")
        
        return model
    
    def train(self):
        sentences,sentiments,cross_validation,test_set = self.get_lines()
        
        train_id = tf.reshape(sentences[:,0,:],shape=(self.max_size,100))
        train_attn = tf.reshape(sentences[:,1,:],shape=(self.max_size,100))
        sentiments = tf.reshape(sentiments,shape=(sentiments.shape[0],1))
        
        cross_validation[1] = tf.reshape(cross_validation[1],shape=(cross_validation[1].shape[0],1))
        test_set[1] = tf.reshape(test_set[1],shape=(test_set[1].shape[0],1))
        
        model = self.build_model()
        print("\n cv : ",cross_validation[1].shape)
        model.fit([train_id,train_attn],sentiments,epochs=100,validation_data=([tf.reshape(cross_validation[0][:,0,:],shape=(self.cv_size,100)),tf.reshape(cross_validation[0][:,1,:],shape=(self.cv_size,100))],cross_validation[1]))

        model.save("Sentiment_Model.keras")
        
        return None
    
#model = TFRobertaModel.from_pretrained("roberta-base-uncased")
#tokeniser = RobertaTokenizer.from_pretrained("roberta-base-uncased")

#dependecy parsing
'''dependency = nlp(text)

for i in dependency:
    print("{:<20} | {:<7} | {:<20} | {:<10} | {:<5}".format(i.text,i.dep_,i.head.text,i.pos_,i.tag_))

analyzer = SentimentIntensityAnalyzer()

polarity = analyzer.polarity_scores(text)
print(polarity)
'''


if __name__ == "__main__":
    sentiment_model = sentiment()
    sentiment_model.train()


