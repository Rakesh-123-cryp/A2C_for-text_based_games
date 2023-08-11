from keras.losses import cosine_similarity
from striprtf.striprtf import rtf_to_text
import regex as re
from gensim.models import Word2Vec
import numpy as np
from transformers import TFDistilBertModel, DistilBertTokenizer,DistilBertConfig
import tensorflow as tf

loaded_model = Word2Vec.load("word_vec")

def epsilon_greedy(similarity,epsilon = 0.8,empty_inventory = 1):
    r = np.random.random()
    
    if r>(1-epsilon):
        if empty_inventory>1:
            return np.argmax(similarity == np.sort(similarity)[-empty_inventory])
        return np.argmax(similarity)
    else:
        return np.random.choice(np.arange(len(similarity))).item()
    
def get_command_verb():
    string = "^(([a-z]+)+ *([a-z]+)*) *\[*"
    pattern = re.compile(string)

    f1 = open("commands.rtf","r")
    lines = rtf_to_text(f1.read())
    lines = lines.split(sep = "\n")

    commands = [re.findall(pattern,line.lower())[0][0].strip() for line in lines]
    
    for i in commands:
        try:
            loaded_model.wv[i.split()[0]]
        except:
            ind = commands.index(i)
            print("is not present : ",i)
            
    return commands

def _remove_special_tokens(output):
    return output[:,1:-1,:]

class embedding:
    def __init__(self):
        self.verb_embedding = []
        
    def get_embddeing_similarity(self,embedding):
        
        if self.verb_embedding == []:
            verbs = get_command_verb()
            config = DistilBertConfig(dim = 100)
            tokeniser = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            for i in verbs:
                print(tokeniser.tokenize(i))
                words = tokeniser(i,return_tensors="tf")
                print(words)
                model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")#(config)
                output = model(words)
                output = _remove_special_tokens(output[0])
                
                output = tf.reduce_mean(output,axis=1)
                self.verb_embedding.append(output)
                print(output.shape)
        
        similarity = [cosine_similarity(self.verb_embedding[trial],embedding) for trial in range(len(self.verb_embedding))]
        
        return similarity

#if __name__ == "__main__":
    #_get_command_verb()
    #print(loaded_model.wv[""])
    #print(get_embddeing_similarity(np.random.normal(size=(1,786))))