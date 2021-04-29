import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


#Standard in pytorch is that each class taht extends the functionality of the base NN is derived from nn.module
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        #calls constructor for base class then save appropriate variable in class
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam (self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    actions = self.fc3(x)
    
    return actions



class Agent():
    
    # hyperparameters:
        # gamma = determines weighting of future rewards
        # epsilon = solution to explore/exploit dilemma (how often does agent spend exploring vs using best known action)
        # eps_dec = parameter stating by what amount to decrement the epsilon with each time step
        # eps_end = once the agent has explored enough it should reach a very low number (0.01 here) - once it has reached this stage it has essentially 'almost' stopped learning
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        # list comprehension to represent the list of available actions as a integers
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        # play around with the sizes
        self.Q_eval = DeepQNetwork(self.lr, n_actions = n_actions, input_dims=input_dims,
                                   fc1_dims = 256, fc2_dims = 256)
        
        # mechanism for storing memory
        
        self.state_memory = np.zeroes((self.mem_size, *input_dims), 
                                      dtype = np.float32)
        
        self.new_state_memory = np.zeroes((self.mem_size, *input_dims), 
                                          dtype = np.float32)
        
        self.action_memory = np.zeres(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # value of terminal state is always 0: if terminal state is encountered then the game is done
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        


    def store_transitions(self, state, action, reward, state_, done):
        
        # position of the first unoccupied memory - wrap around so that once the agent has reached 100000 memories it will start overwriting its oldest memories
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index]
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
    
    
    #                       observation of the current state of environment
    def choose_action(self, observation):
        
        # Pick a random number and if it is greater than epsilon then take the best known action 
        
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)                # pass the state of the environment through the nn        
            action = T.argmax(actions).item()                   # use argmax to specify the maximal action for the state
        
        # otherwise do a random action from the environment
        
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.G_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arrange(self.batch_size, dtype=np.int32)
        
        # convert all numpy arrays to pytorch tensors
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_memory[batch]
        
        # forward pass on nn                     slicing here to get only the values of the actions taken
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        
        #----------if using a target network do it here----------
        q_next = self.Q_eval.forward(new_state_batch)
        
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dims=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        # decrement epsilon unless it has already reached minimum
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min
        
    
    
    
    







