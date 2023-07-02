import torch
import random
import numpy as np
from collections import deque   # Doubly Ended Queue
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import time
from datetime import datetime

MAX_MEMORY = 100_000 # 1,00,000 items can be stored in hte deque
BATCH_SIZE = 1000    # 
LR = 0.001           # Learning Rate


class Agent:

    def __init__(self):
        """
        Initialize the Agent
        """
        self.n_games = 0                        # number of games played so far
        self.epsilon = 0                        # Parameter to control the randomness of moves
        self.gamma = 0.9                        # discount rate used in bellman equation (loss function) (MUST BE SMALLER THAN 1)
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() automatically if we exceed the memory
        self.model = Linear_QNet(11, 256, 3)    # Linear Q-Network with (Input size, Hidden Size, Output Size)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        """
        Getting the state information (11 values) based on the game information
        (e.g, Snake information, moving direction information, etc)"""
        head = game.snake[0]    # Point of Head of the snake
        
        point_l = Point(head.x - 20, head.y) # Point from left of head
        point_r = Point(head.x + 20, head.y) # Point from right of head
        point_u = Point(head.x, head.y - 20) # Point from upside of head
        point_d = Point(head.x, head.y + 20) # Point from downside of head
        
        dir_l = game.direction == Direction.LEFT    # Is Sanke moving in left
        dir_r = game.direction == Direction.RIGHT   # Is Sanke moving in right
        dir_u = game.direction == Direction.UP      # Is Sanke moving to upside
        dir_d = game.direction == Direction.DOWN    # Is Sanke moving to downside

        state = [
            # Danger straight   ## 1st State ##
            (dir_r and game.is_collision(point_r)) or # Is snake moving in right & there is wall on right side
            (dir_l and game.is_collision(point_l)) or # Is snake moving in left & there is wall on left side
            (dir_u and game.is_collision(point_u)) or # Is snake moving in upper & there is wall on upper side
            (dir_d and game.is_collision(point_d)),   # Is snake moving in lower & there is wall on lower side

            # Danger right     ## 2nd State ##
            (dir_u and game.is_collision(point_r)) or # Same as mentioned in Snake Danger Straight movement logic
            (dir_d and game.is_collision(point_l)) or # Same as mentioned in Snake Danger Straight movement logic
            (dir_l and game.is_collision(point_u)) or # Same as mentioned in Snake Danger Straight movement logic
            (dir_r and game.is_collision(point_d)),   # Same as mentioned in Snake Danger Straight movement logic

            # Danger left    ## 3nrd State ##
            (dir_d and game.is_collision(point_r)) or # Same as mentioned in Snake Danger Straight movement logic
            (dir_u and game.is_collision(point_l)) or # Same as mentioned in Snake Danger Straight movement logic
            (dir_r and game.is_collision(point_u)) or # Same as mentioned in Snake Danger Straight movement logic
            (dir_l and game.is_collision(point_d)),   # Same as mentioned in Snake Danger Straight movement logic
            
            # Move direction
            dir_l,          ## 4th State ##
            dir_r,          ## 5th State ##
            dir_u,          ## 6th State ##
            dir_d,          ## 7th State ##
            
            # Food location compared to snake head location
            game.food.x < game.head.x,  # food in left
            game.food.x > game.head.x,  # food in right
            game.food.y < game.head.y,  # food on upside
            game.food.y > game.head.y  # food in downside
            ]

        return np.array(state, dtype=int)   # A trick to convert the boolean values of food location to 0 & 1


    def remember(self, state, action, reward, next_state, done):
        """
        Storing the Information of th games being played in the deque (memory)
        """
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        """
        Training the model on a multiples games information. 
        """

        # Fetch a mini sample of games with no. of games = BATCH_SIZE from deque (memory), 
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        
        # else all the info about all the games played so far
        else:
            mini_sample = self.memory

        # Multiple lists of different informations regarding a game
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Training the model on a single game information
        """
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        """
        Get the next action based on the current state 
        """
        # Random moves: tradeoff between exploration / exploitation
        self.epsilon = 80 - self.n_games
        
        final_move = [0,0,0]
        
        ## Selecting whether to play a ramdom move or not, the more games played (uptil 80 games), the lesser the chances of a random move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)        # (1 X 3) with any raw float values, as it executes the forward() by default
            move = torch.argmax(prediction).item() # torch.argmax(prediction) gives the index as a tensor, and item() converts the tensor to an integer
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        
        # Get Old State
        state_old = agent.get_state(game)

        # Get Move
        final_move = agent.get_action(state_old)

        # Perform the move and get new state
        reward, done, score, load_model_optimizer = game.play_step(final_move)
        
        # Load model if input is given by the user
        if load_model_optimizer:
            model_filename = './model/MAY12_2/record_game_116.pth.tar'
            optimizer_filename = './model/MAY12_2/record_game_opt_116.pth.tar'
            agent.n_games = int(model_filename.split("/")[-1].split(".")[0].split("_")[-1])

            agent.model.load(model_filename)
            agent.trainer.optimizer_load(optimizer_filename)
        
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember all information---
        agent.remember(state_old, final_move, reward, state_new, done)

        # If Game over
        if done:
            # train long memory, and plot result
            
            game.reset()                # Reset the game conditions
            agent.n_games += 1          # Increase no. of games played
            agent.train_long_memory()

            # If snake performance has improved compared to all previous games
            if score > record:
                record = score
                
                agent.model.save("record_game_" + str(agent.n_games) + ".pth.tar")  # Saving the model when record is reached
                agent.trainer.optimizer_save("record_game_opt_" + str(agent.n_games) + ".pth.tar")  # Saving the optimizer when record is reached
                
            elif int(agent.n_games%20) == 0:
                agent.model.save(file_name = "game_" + str(agent.n_games) + ".pth.tar")   # Saving the model after every 20 games
                agent.trainer.optimizer_save(file_name = "game_opt_" + str(agent.n_games) + ".pth.tar")   # Saving the optimizer after every 20 games
                
            print('Game', agent.n_games, 'Score', score, 'Record Score:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Ploting the graph of scores and mean_scores after each game
            plot(plot_scores, plot_mean_scores)
            

if __name__ == '__main__':
    train()