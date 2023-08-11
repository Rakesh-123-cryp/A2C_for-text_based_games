import numpy as np

class experience_replay:
    def __init__(self):
        self.counter = 0
        self.buffer_size = 100
        self.state = np.zeros(shape=(self.buffer_size,150,768))
        self.prev_action = np.zeros(shape=(self.buffer_size,12,768))
        self.action = np.zeros(shape=(self.buffer_size,1))
        self.terminal = np.zeros(shape=(self.buffer_size,1))
        self.reward = np.zeros(shape=(self.buffer_size,1))
        self.state_value = np.zeros(shape=(self.buffer_size,1))
    
    #Function to add set of state,action,reward,new_state    
    def add_sars(self,state,action,reward,prev_action,terminal,value_func):
        self.counter = self.counter%self.buffer_size
        self.prev_action[self.counter,:,:] = prev_action
        self.state[self.counter,:,:] = state
        self.action[self.counter,0] = action
        self.reward[self.counter,0] = reward
        self.terminal[self.counter,0] = terminal
        self.state_value[self.counter,0] = value_func
        self.counter+=1
        d = {"STATE" : state.shape ,"ACTION" : action.shape,"REWARD" : reward,"PREV_ACTION" : prev_action,"VALUE" :value_func}
        print(d,end="\n")
    
    #Function to get the sample set    
    def sample(self,num1=3,num2=None):
        replays = ()
        max_index = min(self.counter,self.buffer_size-1)
        
        #retrieving random indexes 
        i = np.random.choice(np.arange(max_index-1),size=(num2,1),replace=False)
        
        for iter in i:
            replays +=  ((self.state[iter[0]:iter[0]+num1,:,:], self.action[iter[0]:iter[0]+num1,0], \
                self.reward[iter[0]:iter[0]+num1,0], self.prev_action[iter[0]:iter[0]+num1,:,:], \
                    self.terminal[iter[0]:iter[0]+num1,0], self.state_value[iter[0]:iter[0]+num1,0]),)
        print("REPLAY SHAPE : ",len(replays),end="\n")
        return replays
    
    #Function to get cumilative reward to get 
    def get_cum_reward(self):
        summation = 0
        for i in range(10):
            index = self.counter%self.buffer_size
            summation += self.reward[index,0]
            index+=1
        return summation

    def return_seq(self):
        max_index = min(self.counter-1,self.buffer_size-1)
        replays = (self.state[:max_index+1,:,:], self.action[:max_index+1,0], \
                self.reward[:max_index+1,0], self.prev_action[:max_index+1,:,:], \
                    self.terminal[:max_index+1,0], self.state_value[:max_index+1,0])
        return replays
        
        