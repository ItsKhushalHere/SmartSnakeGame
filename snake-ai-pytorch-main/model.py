import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """
    Linear Q-Network for Snake.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.outut_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)   # First Linear Layer
        self.linear2 = nn.Linear(hidden_size, output_size)  # Second Linear Layer

    def forward(self, x):
        """
        Forward passing
        """
        x = F.relu(self.linear1(x)) # Input Tensor passed to first linear layer with it's output passed to Relu Activation function
        x = self.linear2(x)         # Output of activation function passed to second linear layer with it's output (Raw numbers) used directly.
        return x

    def save(self, file_name='snake_default.pth.tar'):
        
        model_folder_path = './model/MAY14'
        
        # If model saving folder doesn't exists, then create it
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        
        torch.save(self.state_dict(), file_name)
        
    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))
        self.eval()
        print("MODEL LOADED !!!")

    
class QTrainer:
    """
    Q Trainer for Training the network.
    """
    def __init__(self, model, lr, gamma):
        self.lr = lr        # Learning Rate
        self.gamma = gamma  # Bellman equation's gamma parameter
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # Optimizer
        self.criterion = nn.MSELoss()                               # Mean squared error loss function

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # Dimension (n, x) :- "n" is number of batches

        ## If only a single game info is passed
        if len(state.shape) == 1:
            # Dimension (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
    
    def optimizer_save(self, file_name='snake_default_opt.pth.tar'):
        
        model_folder_path = './model/MAY14'
        
        # If model saving folder doesn't exists, then create it
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        
        torch.save(self.optimizer.state_dict(), file_name)


    def optimizer_load(self, file_name):
        self.optimizer.load_state_dict(torch.load(file_name))
        print("OPTIMIZER LOADED !!!")

