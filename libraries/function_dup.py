import numpy as np
from pysentimiento import create_analyzer
from spacytextblob.spacytextblob import SpacyTextBlob
from gensim.models import Word2Vec
from nltk.corpus import brown,verbnet,propbank
from keras.losses import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import pandas as pd
from transformers import DistilBertConfig,TFDistilBertModel,DistilBertTokenizer
import tensorflow as tf
import flair
import regex as re

#nltk.download("vader_lexicon")
nlp = spacy.load('en_core_web_sm')
    
def epsilon_greedy(output,epsilon,empty_inventory=1):
    #print("\nEpsilon output shape : \n",output.shape)
    if np.random.random()<(1-epsilon):
        if empty_inventory > 1:
            #print("sorted ARRAY : ",np.sort(output[0]))
            return np.argmax(output == np.sort(output[0])[-empty_inventory])
        #print("Max index : \n",np.argmax(output,axis=1))
        return np.argmax(output,axis=1).item()
    else:
        return np.random.choice(np.arange(output.shape[1])).item()

class Reward:
    
    def __init__(self,score):
        self.score = score
        self.locations = ("West Of House",)
    
    def reward_function(self,state_info,score = 0):
        
        if score>self.score:
            return 10
        
        if state_info["position"] not in self.locations:
            self.locations+=(state_info["position"],)
            return 5
        
        if state_info["description"] == "Aaaarrrrgggghhhh!":
            return -1
        
        flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
        s = flair.data.Sentence(state_info["description"])
        flair_sentiment.predict(s)
        total_sentiment = s.labels
        if "NEGATIVE" in total_sentiment[0].value:
            return -5
        else:
            return -1
            
            
        '''analyzer = create_analyzer(task="sentiment", lang="en")
        if analyzer.predict(state_info["description"]).output == "NEG":
            return -1
        else:
            return 0'''
        
        '''vocab = spacy.load('en_core_web_sm')
        vocab.add_pipe("spacytextblob")
        doc = vocab(state_info["description"])
        if(doc._.blob.polarity<0):
            print("\nNegative Reward")
            return -1
        else:
            print("\nPositive reward")
            return 0'''

'''def reward_function(state_info,visited=[None],score = 0):
    if state_info["position"] not in visited[:-1]:
        return 10
    
    analyzer = SentimentIntensityAnalyzer()
    #analyzer = create_analyzer(task="sentiment",lang="en")
    #output = analyzer.predict(state_info)
    #output = analyzer.polarity_scores(state_info)
    #print(output)
    vocab = spacy.load('en_core_web_sm')
    vocab.add_pipe("spacytextblob")
    doc = vocab(state_info["description"])
    if(doc._.blob.polarity<0):
        print("\nNegative Reward")
        return -1
    else:
        print("\nPositive reward")
        return 0'''

def get_simiarity_score(command,type,objects=None,items=None):
    
    # Code for getting the tf-idf score to imporve the cosine similarity score
    '''counter = CountVectorizer(analyzer="word")
    doc = [" ".join(sent).replace("``",'"').replace("`","'").replace("''",'"') for sent in brown.sents()]
    print(doc[0])
    word_count = counter.fit_transform(doc)
    #tf = pd.DataFrame(word_count.toarray(),columns=counter.get_feature_names_out())
    idf_vector = TfidfTransformer()
    x = idf_vector.fit_transform(word_count)
    print(idf_vector.idf_,)'''
    
    #Making command syntax into a list to facilitate similarity calculation
    #print("\nChosen command Syntax : ",command)
    command_syntax = command.split()
    command_count = [0]
    action_scores = pd.DataFrame(columns=["similarity","command"])
    
    #distilbert model
    #config = DistilBertConfig(dim = 100,n_heads=10)
    tokeniser = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")#(config=config)
    
    string = "^(([a-z]+)+ *([a-z]+)*) *\[*"
    pattern = re.compile(string)
    first = re.findall(pattern,command.lower())[0][0].strip()
    
    #print("First : ",first.lower())
    # if the format is action [object] preposition [object]
    if items is None:

        word = tokeniser(first.lower(),return_tensors="tf")
        output = model(word)
        #print("output embedding : ",output[0])
        output = output[0][:,1:-1,:]
        output = tf.reduce_mean(output,axis=1)
        
        for i in objects:
            #print("THE INITIAL OBJECT CONSIDERED : ",i)
            word = tokeniser(i,return_tensors="tf")
            output1 = model(word)
            output1 = output1[0][:,1:-1,:]
            output1 = tf.reduce_mean(output1,axis=1)
            
            if type == "simple":
                
                cosine_sim = cosine_similarity(output,output1)
                
                #print("\nConsine Similarities :", cosine_sim)
                if cosine_sim<-0.65:
                    command_replacement = command.replace("[object]",i)
                    temp_df = pd.DataFrame({"similarity" : cosine_sim.numpy().item(),"command" : command_replacement},index=command_count)
                    action_scores = pd.concat([action_scores,temp_df],axis=0,ignore_index=True)
                    #print("\nDataframe for each command : ",action_scores)
                    command_count[0]+=1
            else:
                word = tokeniser(command_syntax[2].lower(),return_tensors="tf")
                output_ = model(word)
                output_ = output_[0][:,1:-1,:]
                output_ = tf.reduce_mean(output_,axis=1)
                
                for j in objects:
                    #print("THE OBJECT CONSIDERED : ",j)
                    if i != j:
                        word = tokeniser(j,return_tensors="tf")
                        output2 = model(word)
                        output2 = output2[0][:,1:-1,:]
                        output2 = tf.reduce_mean(output2,axis=1)
                            
                        cosine_sim = cosine_similarity((output + output1)/2,(output_+output2)/2)
                            
                        #print("\nConsine Similarities :", cosine_sim)
                        
                        if cosine_sim<-0.65:
                            command_replacement = command.replace("[object]",i,1).replace("[object]",j)
                            #print("Altered command : ",command)
                            temp_df = pd.DataFrame({"similarity" : cosine_sim.numpy().item(),"command" : command_replacement},index=[command_count])
                            action_scores = pd.concat([action_scores,temp_df],axis=0,ignore_index=True)
                            #print("\nDataframe for each command : ",action_scores)
                            command_count[0]+=1
    
    # if the format is action [item]
    elif objects is None:
        word = tokeniser(first.lower(),return_tensors="tf")
        output = model(word)
        #print("output embedding : ",output[0])
        output = output[0][:,1:-1,:]
        output = tf.reduce_mean(output,axis=1)
        
        for i in items:
            word = tokeniser(i,return_tensors="tf")
            output1 = model(word)
            #print("output1 embedding : ",output1[0])
            output1 = output1[0][:,1:-1,:]
            output1 = tf.reduce_mean(output1,axis=1)
            
            cosine_sim = cosine_similarity(output,output1)
            
            #print("\nConsine Similarities :", cosine_sim)
            if cosine_sim<-0.65:
                command_replacement = command.replace("[item]",i)
                temp_df = pd.DataFrame({"similarity" : cosine_sim.numpy().item(),"command" : command_replacement},index=command_count)
                action_scores = pd.concat([action_scores,temp_df],axis=0,ignore_index=True)
                #print("\nDataframe for each command : ",action_scores)
                command_count[0]+=1
    
    else:
        word = tokeniser(first.lower(),return_tensors="tf")
        output = model(word)
        #print("\noutput embedding : ",output[0])
        output = output[0][:,1:-1,:]
        output = tf.reduce_mean(output,axis=1)
        
        word = tokeniser(command_syntax[2].lower(),return_tensors="tf")
        output_ = model(word)
        #print("\noutput_ embedding : ",output_[0])
        output_ = output_[0][:,1:-1,:]
        output_ = tf.reduce_mean(output,axis=1)
        
        for i in objects:
            word = tokeniser(i,return_tensors="tf")
            output1 = model(word)
            #print("\noutput1 embedding : ",output1[0])
            output1 = output1[0][:,1:-1,:]
            output1 = tf.reduce_mean(output1,axis=1)
            
            for j in items:
                if i != j:
                    word = tokeniser(i,return_tensors="tf")
                    output2 = model(word)
                    output2 = output2[0][:,1:-1,:]
                    output2 = tf.reduce_mean(output2,axis=1)
                    
                    cosine_sim = cosine_similarity((output + output1)/2,(output_+output2)/2)
                    
                    #print("\nConsine Similarities :", cosine_sim)
                    if cosine_sim<-0.65:
                        command_replacement = command.replace("[object]",i).replace("[item]",j)
                        temp_df = pd.DataFrame({"similarity" : cosine_sim.numpy().item(),"command" : command_replacement},index=command_count)
                        #print("\nDataframe for each command : ",temp_df)
                        action_scores = pd.concat([action_scores,temp_df],axis=0,ignore_index=True)
                        #print("\nDataframe for each command : ",action_scores)
                        command_count[0]+=1
                    
    '''if len(action_scores) == 0:
        temp_df = pd.DataFrame({"similarity" : 0.7,"command" : "n"},index=command_count)
        action_scores = pd.concat([action_scores,temp_df],axis=0,ignore_index=True)'''
    return action_scores

def get_objects(description):
    result = nlp(description)
    objects = []
    for i in result:
        if i.dep_ in ("pobj","obj","attr","dobj","iobj","obl","obj2"):
            if i.tag_ in  ("NN", "NNS", "NNP", "NNPS"):
                if i.text not in objects:
                    #print("{:<10} | {:<7} | {:<10} |".format(i.text,i.dep_,i.head.text))
                    objects.append(i.lemma_)
    return objects

if __name__ == "__main__":

    '''flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    s = flair.data.Sentence("You are facing the north side of a white house. There is no door here, and all  the windows are boarded up. To the north a narrow path winds through the trees.")
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    print(total_sentiment)'''
    
    #total_sentiment.value
    #get_simiarity_score("open [object] with [item]",type=None,items=["key","leaf","sword"],objects=["field","door","window"])
    #reward_function("You are not worth existing")
    #print(get_objects("You are standing in an open field west of a white house, with a boarded front door. There is a small mailbox here."))
    #get_objects("there is a thief in the house")
    #loaded_model = Word2Vec.load("word_vec")
    #oaded_model.train(["deflate"],total_words=1,epochs=50)
    #print(len(loaded_model.wv["deflate"]))
    #nltk.download('propbank')
    '''doc = [" ".join(sent).replace("``",'"').replace("`","'").replace("''",'"') for sent in reuters.sents()]
    wv = Word2Vec(reuters.sents(),vector_size=100,min_count=1)
    wv.train(reuters.sents(),total_examples=1,epochs=50)
    wv.save("word_vec")'''
    #print(help(propbank))