import tensorflow as tf
import numpy as np
import copy
import regex as re
from transformers import TFBertModel,TFEncoderDecoderModel,BertTokenizer,EncoderDecoderConfig,TransfoXLTokenizer, TFAutoModel
import pandas as pd

import json
import itertools

'''f = open("commands_disc_data/train_dataset.json")
df = pd.DataFrame(columns=["observation","previous_action","previous_triplets","target_commands"])
for line in itertools.islice(f,5):
  line = line.strip()
  if not line: 
      continue
  print(json.loads(line))

f.close()'''
class encoderdecoder:
    
    def __init__(self) -> None:
        pass
    def get_data(self):
        nrows = 2001
        location_dict = {}
        file = open("commands_disc_data/train_dataset.json")
        data = pd.read_json("commands_disc_data/train_dataset.json",lines=True,nrows=nrows)
        desc = data.loc[:,"observation"].values[:nrows-1].tolist()
        new_desc = copy.deepcopy(desc)
        action = data.loc[:,"previous_action"].values[1:nrows].tolist()
        location=""
        inventory = ["salt"]
        iter=0
        while(iter<10):
            print("action : ",action[iter])
            
            if action[iter] == "inventory":
                action.remove("inventory")
                print("description popped : ",desc[iter+1])
                desc.pop(iter+1)
                continue
            elif action[iter].startswith("take"):
                inventory.append(re.search("[a-z]* (.+)( from)*",action[iter]).group(1))
            else:
                 for item in inventory:
                    if item in action[iter]:
                        inventory.remove(item)
            #elif action[iter].startswith("drop"):
            #    inventory.remove(" ".join(action[iter].split()[1:]))
                
            if desc[iter].startswith(("-="," -=")):
                pat = re.compile("-= [A-Z]{1}[a-z]* =")
                pat1 = re.compile("[A-Z]{1}[a-z]*")
                location = re.findall(pat1,re.findall(pat,desc[iter])[0])[0]
                if location not in location_dict:
                    location_dict[location] = desc[iter][re.search(pat,desc[iter]).span()[1]+3:]
                desc[iter] = location_dict[location]
                print("location : ",location)
                
            else:
                desc[iter] = desc[iter-1]+desc[iter]
                location_dict[location] = desc[iter]
                
            items = ",".join(inventory)
            new_desc[iter] = desc[iter]
            if inventory != []:
                new_desc[iter] += "inventory contains " + items
            print("location_disc : ",new_desc[iter])
            
            iter+=1
        #print("desc : ",desc.shape,"action : ",action.shape)
        return new_desc,action
    
    def _shift(self,xs, n, val):
        
        return np.concatenate((xs[:,:,-n:], np.full((xs.shape[0],1,-n), val)),axis=-1)
    
    def encoder_decoder(self,ids1,attn1,ids2,attn2,dec_input):
        bert = TFBertModel.from_pretrained("bert-large-uncased")
        base_bert = TFBertModel.from_pretrained("bert-base-uncased")
        output = base_bert(input_ids=tf.squeeze(dec_input,axis=1),attention_mask=tf.squeeze(tf.convert_to_tensor(attn2),axis=1))[0]
        print(output.shape)
        transfoxl = TFAutoModel.from_pretrained("transfo-xl-wt103")
        self.model = TFEncoderDecoderModel(encoder = bert,decoder=transfoxl)
        self.model.config.decoder.is_decoder = True
        self.model.config.decoder.add_cross_attention = True
        output = self.model(input_ids = ids1,attention_mask = attn1, labels=ids2, decoder_inputs_embeds=output, decoder_attention_mask = attn2,training=True)
        print(type(output.logits))
        print(output.logits.shape)
        print(output.loss)
        
    def tokeniser_func(self,data1,data2):
        bert_tokeniser = BertTokenizer.from_pretrained("bert-large-uncased")
        bert_token = [bert_tokeniser(data,padding="max_length",max_length=250,return_tensors="tf") for data in data1]
        print(type(bert_token[0].input_ids))
        bert_info = [tf.concat([x.input_ids,x.attention_mask],axis=0) for x in bert_token]
        bert_info = tf.convert_to_tensor(bert_info)
        bert_attn = bert_info[:,1,:]
        bert_ids = bert_info[:,0,:]
        print(bert_info.shape)
        
        base_bert_tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
        #self.decoder_input = base_bert_tokeniser.cls_token_id
        xl_token = [base_bert_tokeniser(data,padding="max_length",max_length=10,return_tensors="tf") for data in data2]
        xl_ids = [x.input_ids for x in xl_token]
        xl_ids = np.array(xl_ids,ndmin=3)
        decoder_input = xl_ids
        xl_ids = self._shift(xl_ids,n=-1,val=0)
        xl_attn = np.ones_like(xl_ids)
        xl_attn = np.where((decoder_input == base_bert_tokeniser.pad_token_id),decoder_input,xl_attn)
        print(xl_attn[0])
        #xl_attn[xl_attn == base_bert_tokeniser.pad_token_id] = 1
        print(xl_ids.shape)
        
        return [bert_ids,bert_attn,xl_ids,xl_attn,decoder_input]
    
    def forward(self):
        bert,xl = self.get_data()
        tokens = self.tokeniser_func(bert,xl)
        self.encoder_decoder(tokens[0],tokens[1],tokens[2],tokens[3],tokens[4])
        
if __name__ == "__main__":
    obj = encoderdecoder()
    obj.forward()