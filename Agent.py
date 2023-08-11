'''DOC-STRING
    -> 
   DEVELOPER-NOTES
    -> This code is the developed version of running.py with added cross-attention layer in the NN.
    -> No improvments after this model
    ->it has a change in command-None code the latest version from 3 and 2
   '''

import pexpect
import regex
import os
import nltk
import numpy as np
from datetime import datetime
from nltk.corpus import stopwords
#from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import spacy
from transformers import BertTokenizer, TFBertModel,TFDistilBertModel,DistilBertTokenizer
import matplotlib.pyplot as plt
import tensorflow as tf
from libraries.Experience_replay import experience_replay
from libraries.graph import graph_node
import libraries.function_dup as fns
from keras.losses import Loss
from striprtf.striprtf import rtf_to_text
import pandas as  pd
from libraries.cross_attention import Cross_Attention_Layer

nlp = spacy.load('en_core_web_sm')
commands_dict = {}
#Reading the file to get commands
f1 = open("commands.rtf","r")
commands = rtf_to_text(f1.read())
commands = commands.split(sep = "\n")
#commands = f1.readlines()
#print(len(commands))
# Loading the game
game = pexpect.spawn("frotz game/zork1.z5")

#ALL Regex patterns for retrieving the texts
pattern = r"(\r.*(\[){1}m)"
pattern2 = r"(\r.*[0-9]d)"
#pattern3 = r"\x1b.*H"
initial = r"\x1b.*(\[){1}m"
#pos = "[A-Z][a-z].*[a-z]{2,}"
only_pat = regex.compile(r"((\x9B|\x1B\[)[0-?]*[ -\/]*[@-~])|(\x1b\(B)")
#pos_pattern = regex.compile(pos)
initial_pattern = regex.compile(initial)
r = regex.compile(pattern)
r2 = regex.compile(pattern2)
#r3 = regex.compile(pattern3)

#Getting the attributes for the syntax specified
def get_command_attr(command_syntax,items,objects):
    parts = command_syntax.split()
    if "[item]" in parts:
        if "[object]" in parts:
            df = fns.get_simiarity_score(command_syntax,type=None,items = items,objects = objects)
            #print("\nDataFrame : ",df)
            #getting the command with the highest similarity
            try:
                command = df[df["similarity"] == df["similarity"].min()]["command"].values.item()
            except:
                command = None
                
            #print("\nSelected command : ", type(command),"\n")
            return command,df
        else:
            df = fns.get_simiarity_score(command_syntax,type=None,items = items,objects = None)
            #print("\nDataFrame : ",df)
            #getting the command with the highest similarity
            try:
                command = df[df["similarity"] == df["similarity"].min()]["command"].values.item()
            except:
                command = None
                
            #print("\nSelected command : ", command,"\n")
            return command,df
        
    elif "[object]" in parts:
        if parts.count("[object]") == 2:
            df = fns.get_simiarity_score(command_syntax,type="double",items = None,objects = objects)
            #print("\nDataFrame : ",df)
            #getting the command with the highest similarity
            try:
                command = df[df["similarity"] == df["similarity"].min()]["command"].values.item()
            except:
                command = None
             
            #print("\nSelected command : ", command,"\n")
            return command,df

        else:
            df = fns.get_simiarity_score(command_syntax,type="simple",items = None,objects = objects)
            #print("\nDataFrame : ",df)
            #getting the command with the highest similarity
            try:
                command = df[df["similarity"] == df["similarity"].min()]["command"].values.item()
            except:
                command = None
             
            #print("\nSelected command : ", type(command),"\n")
            return command,df

    else:
        df = pd.DataFrame({"similarity" : 1, "command" : command_syntax},index=[0])
        return command_syntax,df

#Custom loss function
def logloss(y_expec, y_pred):
        y_expec = tf.clip_by_value(y_expec,clip_value_min=0.001,clip_value_max=1)
        log_prob = -1*tf.math.log(y_pred)*y_expec
        return log_prob
        
#get X anf Y traninig data
def get_training(batches,discount_factor = 0.6):
    X = np.zeros(shape=(batches[0].shape[0],150,768))
    X_action = np.zeros(shape=(batches[0].shape[0],12,768))
    Y_statevalue = np.zeros(shape=(batches[0].shape[0],1,1))
    Y_qvalue = np.zeros(shape = (batches[0].shape[0],1,75))
    
    dis_reward = 0
    for i in range(batches[0].shape[0]-1,-1,-1):
        #Setting the X values for training
        X[i,:,:] =  batches[0][i,:,:]
        X_action[i,:,:] = batches[3][i,:,:]
            
        #Calculating Cumilative reward with given discount factor
        dis_reward = batches[2][i] + discount_factor*dis_reward
        
        
        Y_statevalue[i,0,0] = dis_reward
        Y_qvalue[i,0,:].fill(dis_reward - batches[-1][i])
    

    '''for iter in batches:
        dis_reward = 0
        #print("Each batch shape : ",iter[0].shape[0])
        for j in range(iter[0].shape[0]):
            #print("iter shape : ",iter[0].shape,end="\n")
            
            #Calculating Cumilative reward with given discount factor
            dis_reward = iter[2][j] + discount_factor*dis_reward
            
            #Setting the X values for training
            X[j,:,:] =  iter[0][j,:,:]
            X_action[j,:,:] = iter[3][j,:,:]
            
            #The result of the value function is the cumilative reward Gi
            Y_statevalue[j,0] = dis_reward
            
            #Advantage value and training result for Q value
            delta = dis_reward - iter[-1][j]
            Y_qvalue[j,:] = delta
            #np.log(iter[1][j])*delta'''
            
    return X,X_action,Y_qvalue,Y_statevalue
            
#Function to get the first output
def get_initial_output():
    state_info = {}
    
    #output for the first iterstion is treated differently as it has unwanted information
    game.expect(">")
    output = game.before.decode("utf-8")
    position1 = "West of House"
    get_rid = "ZORK I: The Great Underground Empire Copyright (c) 1981, 1982, 1983 Infocom, Inc. All rights reserved. ZORK is a registered trademark of Infocom, Inc. Revision 88 / Serial number 840726 West of House "
    output = regex.sub(initial_pattern,"",output)
    output_first = output.split(" ")
    for i,j in zip(output_first,range(len(output_first))):
        output_first[j] = regex.sub(r2," ",i)

    output_first = " ".join(output_first)
    output_first = output_first.replace(get_rid,"")
    state_info["position"] = position1
    state_info["description"] = output_first
    
    #print(state_info)
    return state_info

#Function to get the state info for the given command
def get_state_variables(command,prev_pos):
    state_info = {}
    
    # if the game is restarted then the intial output function is called
    if command == "restart":
        global game
        #game.timeout = 1
        #game.expect([">",pexpect.TIMEOUT])
        #game.sendline("restart" + "\n")
        game.terminate(force=True)
        game = pexpect.spawn("frotz game/zork1.z5")
        
        return None,None#get_initial_output()
    
    #Sending the first input and getting the output to and from the game
    game.timeout = 1
    game.expect([">",pexpect.TIMEOUT])
    game.sendline(command + "\n")
    game.expect([">",pexpect.TIMEOUT])
    output1 = game.before.decode("utf-8")
    
    #print("OUTPUT RECIEVED : ",output1.encode("utf-8"))
    #Getting the position from the output
    lines = only_pat.sub(" ",output1).split(sep="\r")
    #print("LINES : ",lines)
    
    score=0
    
    #If the word entered is not recognised then there is a spcial if statement
    if 'know the word' in lines[1].strip() or not(lines[1][-1].isdigit()):
        output1 = lines[1].strip()
        state_info["position"] = prev_pos
    
    #If the output is time passes
    elif lines[2].strip() == "Time passes...":
        state_info["position"] = prev_pos
        position = prev_pos
        lines = "Time passes..."
        #Getting the score
        score = lines[1][-3] if lines[1][-3].isdigit() else 0
    
    #elif statement to get the name of the place otherwise 
    elif (len(lines[2].split())>3) or lines[2].strip() in ("Taken.","Aaaarrrrgggghhhh!") or lines[2].endswith((".","!","?")):
        #print("entered here")
        state_info["position"] = prev_pos
        position = prev_pos
        output1 = " ".join(lines[2:])
        #Getting the score
        score = lines[1][-3] if lines[1][-3].isdigit() else 0
    
    #When entering new place name is taken
    else:
        position = lines[2]
        state_info["position"] = position.strip()
        output1 = " ".join(lines[3:])
        #Getting the score
        score = lines[1][-3] if lines[1][-3].isdigit() else 0
        
    '''if (len(lines[2].split())>3): position = '0' 
    else: position = lines[2]
    
    if position == '0':
        state_info["position"] = prev_pos
        position = prev_pos
        output1 = " ".join(lines[2:])
    else:
        state_info["position"] = position.strip()
        output1 = " ".join(lines[3:])'''
        
    #moves = lines[1]
    state_info["description"] = output1.strip()
    #if position in lines[1]:
    #    moves = moves[-1]
    
    #print(state_info)
    return state_info,score

#Function to add the missing values in the input
def add_empty(container,type="description"):
    if type == "description":
        zeros = tf.zeros(shape=(1,150-container.shape[1],container.shape[-1]))
        container = tf.concat([container,zeros],axis=1)
        #print("Container shape : ",container.shape,end="\n")
        return container
    else:
        zeros = tf.zeros(shape=(1,12-container.shape[1],container.shape[-1]))
        container = tf.concat([container,zeros],axis=1)
        #print("Container shape : ",container.shape,end="\n")
        return container

#Function to get the command embddings - Not sure whether to be used
def get_command_embedding(command):
    if command is None:
        return tf.zeros(shape=(1,12,768))
    tokeniser = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    words = tokeniser(command,return_tensors = "tf")
    output = model(words)
    emb = output[0][:,1:-1,:]
    emb = add_empty(emb,type="command")
    
    return emb

#Fuction to create the keras lstm model (triggered only once) - ADD NORMALISATION LAYER IF NECESSARY
def create_tf_model():
    #Input layer
    input = tf.keras.layers.Input(shape=(150,768))
    input_action = tf.keras.layers.Input(shape=(12,768))
    
    #Cross-Attention Layer to get sentence sense
    attn = Cross_Attention_Layer(768,768,128,64)([input,input_action])#(tf.concat([input,input_action],axis = 0))
    #print(attn.shape)
    
    #LSTM Layer with 100 cells
    lstm = tf.keras.layers.LSTM(100,input_shape = (150,768),recurrent_activation = "sigmoid")(tf.expand_dims(attn,axis=0))
    
    print("\ncrossed LSTM : \n",lstm.shape,end="\n")

    norm = tf.keras.layers.Normalization(axis=-1)(lstm)
    #ANN for Q-value for actions
    norm = tf.keras.layers.Dropout(0.01)(norm)
    dense1_action = tf.keras.layers.Dense(512,activation='elu')(norm)
    dense2_action = tf.keras.layers.Dense(128,activation='elu')(dense1_action)
    dense3_action = tf.keras.layers.Dense(75,activation='softmax',name = "q_value")(dense2_action)
    
    #ANN for value funtion for state
    dense1_state = tf.keras.layers.Dense(64,activation='elu')(norm)
    dense2_state = tf.keras.layers.Dense(16,activation='sigmoid')(dense1_state)
    dense3_state = tf.keras.layers.Dense(1,activation='relu',name = "state_value")(dense2_state)
    
    print("\npassed the model creation\n")
    #Model compilation
    model = tf.keras.Model(inputs = [input,input_action],outputs = [dense3_action,dense3_state])
    model.compile(optimizer = "Adam",loss={"q_value" : logloss, "state_value" : "mean_squared_error"}, metrics="accuracy")
    return model

# Function to run the bert and pass through lstm
def NLP_model(descp,action_taken,model):
    #Removing Stop words
    
    # Initializing bert tokenizer and bert pre-trained model
    tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    bert  = TFBertModel.from_pretrained("bert-base-uncased")
    
    #For description
    #Tokenising the sentence and turning it into ids with masking
    words = tokenizer.encode(descp)
    #stp_wrds = set(stopwords.words("english")) - set(("not","no","never","hasn't","can't","won't","hadn't","couldn't","wouldn't","shouldn't","haven't","mustn't","isn't"))
    #words = [i for i in words if i not in stp_wrds]
    
    #ids = tokenizer.convert_tokens_to_ids(words)
    #ids = tf.expand_dims(ids, 0)
    words = tf.expand_dims(words, 0)
    mask = tf.ones_like(words)
    
    #Getting the output from the bert model
    outputs = bert(words, attention_mask=mask)
    final = outputs[0]
    #Zeros added to match the dimension of the lstm input
    final = add_empty(final,type="description")
    
    if action_taken is not None:
        #For command
        #tokenising the sentence and turining into ids
        '''command_words = tokenizer.encode(descp)
        
        #ids = tokenizer.convert_tokens_to_ids(words)
        #ids = tf.expand_dims(ids, 0)
        command_words = tf.expand_dims(command_words, 0)
        mask = tf.ones_like(command_words)
        
        #Getting the output from the bert model
        command_output = bert(command_words, attention_mask=mask)
        command_emb = command_output[0]
        #Zeros added to match the dimension of the lstm input
        print(command_emb.shape,"\n\n")
        command_emb = add_empty(command_emb,type="command")'''
        
        action_taken = get_command_embedding(action_taken)#command_emb
    else:
        action_taken = tf.zeros(shape=(1,12,768))
        
    #Check GPU and CPU
    '''devices = tf.config.list_physical_devices("GPU")
    for i in devices:
        print("\ndevice type : ",i.device_type)
        print("\nfreq : ",tf.config.experimental.get_device_details(i))'''
        
    #getting states from the model with the data
    #print(final.shape)
    action_prob = model.predict([final,action_taken])
    print(action_prob[1].shape)
    return [final,action_prob]

#Function to update the minimap
def record_minimap(dest_name,head_node=None,direction=None):
    if head_node == None:
        head_node = graph_node(dest_name)
        return head_node
    else:
        head_node.set_n_node(dest_name,direction)
        return head_node.n_nodes[-1][0]

#Main function that controls the flow of the program
def propogate(**kwargs):
    episode_count = 0
    #Object and tuple creation for model and experience replay
    model = create_tf_model()
    exp = experience_replay()
    score_dict = {"West of House" : {}}
    
    #NLP_model(info["description"],model=model)
    epsilon_value = 0.8
    if len(kwargs)!=0:
        model = kwargs["model"]
        exp = kwargs["exp"]
        score_dict = kwargs["score_dict"]
        episode_count = kwargs["episode"]+1
        
    #Episode loop each episode consists of 100 iterations
    while(episode_count<10):
        i=1
        #Getting the initial state information and minimap
        info = get_initial_output()
        start_location = record_minimap(info["position"])
        current_location = start_location
        object_dict = {}
        
        #required items, objects and environment related details
        items = []
        visited = ()
        area_look = {}
        area_look[info["position"]] = info["description"]
        action_taken = None
        prev_action_taken = None
        reward_object = fns.Reward(0)
        
        
        while(i<=1000):
            
            print("\nIteration : ",i,"\n")
            
            #print("\nState_Info : ",info,end="\n")
            
            #Passing it through the nlp model
            [state,[prob,value]] = NLP_model(info["description"],action_taken,model=model)

            #print("VALUE FUNCTION RESULT : ",value)
            #print("Prob shape of output : ",prob.shape)
            
            #Getting the index of the action to be performed using epsilon greedy
            index = fns.epsilon_greedy(prob,epsilon=epsilon_value)
            
            #Getting the valid action and if inventory is emoty all the commands with [item] tag is removed
            if items == []:
                prob_copy = prob
                for index in range(len(commands)):
                    if "[item]" in commands[index]:
                        #print(commands[index])
                        prob = np.delete(prob,index,axis=1)
                #print(prob)
                index = fns.epsilon_greedy(prob,epsilon=epsilon_value,empty_inventory=1)
            action_syntax = commands[index]
            
            
            objects = fns.get_objects(info["description"])
            
            if info["position"] in object_dict:
                if object_dict[info["position"]] == 1:
                    score_dict[info["position"]] = {}
            
            object_dict[info["position"]] = 0
        
                
            #If actions have not been taken in that position
            if info["position"] not in score_dict:
                action_taken,df = get_command_attr(action_syntax,items,objects+items)
                print("at first : ",action_taken)
                #Getting the dataframe with some command to execute while the preferences are none
                if action_taken is None:
                    num = 2
                    while len(df) == 0 or (action_taken is None):
                        if num>len(prob[0]):
                                return {"model":model,"exp":exp,"score_dict":score_dict,"episode":episode_count}
                            
                        index = fns.epsilon_greedy(prob,epsilon=epsilon_value,empty_inventory=num)
                        action_syntax = commands[index]
                        action_taken,df = get_command_attr(action_syntax,items,objects+items)
                        print("after : ",action_taken,df)
                        num+=1
                
                #The action syntax is stored for that position
                if len(df) == 0:
                    break
                score_dict[info["position"]] = {action_syntax : df}
                print("ACTION_DICT : ",score_dict[info["position"]].keys(),action_taken)
                
            #If actions have been already taken in that state
            elif info["position"] in score_dict:

                #print("ACTION : ",action_syntax ,"DICT : ",score_dict[info["position"]].keys())
                #if the action has already been taken before it takes the old dataframe 
                if action_syntax in score_dict[info["position"]]:
                    
                    df = score_dict[info["position"]][action_syntax]
                    num = 2
                    #If action_syntax in score_dict but the df is empty - meaning the pbjects that existed earlier did not provide positive results
                    while len(df) == 0:
                        if num>len(prob[0]):
                                return {"model":model,"exp":exp,"score_dict":score_dict,"episode":episode_count}
                            
                        index = fns.epsilon_greedy(prob,epsilon=1,empty_inventory=num)
                        action_syntax = commands[index]
                            
                        if action_syntax not in score_dict[info["position"]]:
                            action_taken,df = get_command_attr(action_syntax,items,objects+items)

                        else:
                            df = score_dict[info["position"]][action_syntax]
                        
                    if action_syntax not in score_dict[info["position"]]:
                        score_dict[info["position"]][action_syntax] = df
                    else:
                        action_taken = df[df["similarity"] == df["similarity"].min()]["command"].values.item()
                    
                #If the action is new the attributes are decided
                else:
                    action_taken,df = get_command_attr(action_syntax,items,objects+items)
                    
                    #Getting the dataframe with some command to execute while the preferences are none
                    if action_taken is None:
                        num=2
                        while len(df) == 0 or action_taken == None:
                            if num>len(prob[0]):
                                return {"model":model,"exp":exp,"score_dict":score_dict,"episode":episode_count}
                            
                            index = fns.epsilon_greedy(prob,epsilon=epsilon_value,empty_inventory=num)
                            action_syntax = commands[index]
                            
                            #If the command is already present in the dictionary then the record is considered
                            if action_syntax in score_dict[info["position"]]:
                                df = score_dict[info["position"]][action_syntax]
                                try:
                                    action_taken = df[df["similarity"] == df["similarity"].min()]["command"].values.item()
                                except:
                                    action_syntax = None
                            else:
                                action_taken,df = get_command_attr(action_syntax,items,objects+items)
                                num+=1
                        
                        if action_syntax not in score_dict[info["position"]]:
                            score_dict[info["position"]][action_syntax] = df
                            
                    else:
                        score_dict[info["position"]][action_syntax] = df
            
            
            print("\nAction Taken : ",action_taken)
                
                
            #Recording the minimap
            if info["position"] in visited:
                pass
            else:
                visited+=(info["position"],)
                current_location = record_minimap(info["position"],current_location,action_taken)
                
                
            #Adding items to items
            if action_syntax.startswith("take"):
                items.append(action_taken.split()[-1])
                #print("\nItems : ",items,"\n")
            elif action_syntax.startswith("drop"):
                items.remove(action_taken.split()[-1])
                #print("\nItems : ",items,"\n")
                
            #Executing the action taken and resukt is stored for the next state
            next_info,score = get_state_variables(action_taken,current_location.name)
            next_info["description"] = next_info["description"].replace('"',"").replace("can't","cannot")
            
            print("RESPONSE : ",next_info,end="\n")
            
            #Getting the reward for the action
            if next_info["description"] == "":
                #reward = reward_object.reward_function({"position" : next_info["position"], "description" : area_look[next_info["position"]]},score)
                reward = 0
            else:
                reward = reward_object.reward_function(next_info,int(score))
                
            if reward<0:
                print("NEGATIVE")
            else:
                print("POSITIVE")
                
            #changing the value for each action based on the reward and storing it in a dictionary
            table = score_dict[info["position"]][action_syntax]
            #print("dataframe to be considered : \n",table,"action_taken : ",action_taken)
            sim_score = table[table["command"] == action_taken].iloc[-1,0]
            #print("Score of the chosen action : ",sim_score)
            if reward<=0:
                table = table[table["similarity"] != sim_score]
                score_dict[info["position"]][action_syntax] = table
            
            info = next_info

            #Adding the sequence to the experience replay
            exp.add_sars(np.reshape(state,(150,768)),prob[0,index],reward,get_command_embedding(prev_action_taken),False,value)
            prev_action_taken = action_taken
            
            #To concatenate the descriptions for the next iteration
            if info["position"] in area_look:
                
                #If the reward is negative then the original state description is taken
                if reward<0:
                    info["description"] = area_look[info["position"]]
                    
                #Else if reward is positive the description is concatenated to the original one
                else:
                    area_look[info["position"]] += " " + info["description"]
                    info["description"] = area_look[info["position"]]
                    object_dict[info["position"]] = 1
            #If the place being visited is a new place then the description is stored
            else:
                area_look[info["position"]] = info["description"]
            
            #getting the training value for every 10 iterations
            if(i%100 == 0):
                #print("y=iterations : ",i)
                batches = exp.return_seq()#exp.sample(num1=3,num2=5)
                X_d,X_action,Y_qvalue,Y_statevalue = get_training(batches)
                #print("X : {}, X_action : {}, Y_q : {}, Y_s : {}".format(X_d.shape,X_action.shape,Y_qvalue.shape,Y_statevalue.shape))
                log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                model.fit(x = [X_d,X_action],y = [Y_qvalue,Y_statevalue],epochs=50,verbose=1,callbacks=[tensorboard_callback])
                if i<1000 and epsilon_value>0.2:
                    epsilon_value-=0.2
            # if reward is too less the episode terminates
            if exp.get_cum_reward()<-50:
                info = get_state_variables("restart",None)
                i = 0
                episode_count+=1
                print("\nEpisode ",episode_count," Terminated\n")
                epsilon_value = 0.8
            i+=1
        next_info,score = get_state_variables("restart",current_location.name)
        epsilon_value = 0.8
        print("\nEpisode ",episode_count," Terminated\n")
        episode_count+=1
        
    return None
if __name__ == '__main__':
    graph = propogate()
    while isinstance(graph,dict):
        graph = propogate(model=graph["model"],exp=graph["exp"],score_dict=graph["score_dict"],episode=graph["episode"])

#Accessing elements of the graph
'''print("\nNodes : ",graph.n_nodes,"\n")
for i in graph.n_nodes:
    print(i[1])'''