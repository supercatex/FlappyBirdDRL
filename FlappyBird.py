import pygame
from pygame.constants import K_SPACE
import random
import Agent
import numpy as np


class Game:
    
    def __init__(self):
        pygame.init()
        self.FPS = 60
        self.clock = pygame.time.Clock()
        self.screen_width = 288
        self.screen_height = 512
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Flappy Bird')
        
        self.images = {}
        self.images['background'] = (pygame.image.load('images/background.png').convert())
        self.images['base'] = (pygame.image.load('images/base.png').convert())
        self.images['player'] = (pygame.image.load('images/player.png').convert_alpha())
        self.images['pipe'] = (pygame.image.load('images/pipe.png').convert_alpha())
        
        self.gameover = False
        
        self.player_x = int(self.screen_width * 0.2)
        self.player_y = int((self.screen_height - self.images['player'].get_height()) / 2)
        self.player_vx = -2
        self.player_vy = 0
        self.player_acc = -6
        self.flap_acc = 9.8 / 30
        
        self.upper_pipes = []
        self.lower_pipes = []
        self.t = 0
    
    def update(self, action):
        if self.gameover:
            self.__init__()
        
        # default reward.
        reward = 0
        
        # update position.
        self.player_vy = min(10, self.player_vy + self.flap_acc)
        if action == 1:
            self.player_vy = self.player_acc
        self.player_y = max(0, self.player_y + self.player_vy)
        
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            upper_pipe['x'] += self.player_vx
            lower_pipe['x'] += self.player_vx
        
        # add new pipe.
        if self.t % 80 == 0:
            new_pipe = self.createPipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])
        
        if self.upper_pipes[0]['x'] < -self.images['pipe'].get_width():
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)
        
        # checking.
        if self.player_y >= self.screen_height * 0.79:
            self.gameover = True
        
        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            if self.player_x <= upper_pipe['x'] + self.images['pipe'].get_width() and \
            self.player_x + self.images['player'].get_width() >= upper_pipe['x'] and \
            self.player_y <= upper_pipe['y'] + self.images['pipe'].get_height() and \
            self.player_y + self.images['player'].get_height() >= upper_pipe['y']:
                self.gameover = True
                break
            if self.player_x <= lower_pipe['x'] + self.images['pipe'].get_width() and \
            self.player_x + self.images['player'].get_width() >= lower_pipe['x'] and \
            self.player_y <= lower_pipe['y'] + self.images['pipe'].get_height() and \
            self.player_y + self.images['player'].get_height() >= lower_pipe['y']:
                self.gameover = True
                break
        
        if self.gameover == False:
            if self.player_x + self.images['player'].get_width() > self.upper_pipes[0]['x'] and \
            self.player_x + self.images['player'].get_width() < self.upper_pipes[0]['x'] - self.player_vx:
                reward = 1
            if self.player_x < self.upper_pipes[0]['x'] + self.images['pipe'].get_width() and \
            self.player_x > self.upper_pipes[0]['x'] + self.images['pipe'].get_width() + self.player_vx:
                reward = 1
        else:
            reward = -999999999
        
        # render screen.
        if self.gameover == False:
            self.screen.blit(self.images['background'], (0, 0))
            self.screen.blit(self.images['player'], (self.player_x, self.player_y))
            for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
                self.screen.blit(self.images['pipe'], (upper_pipe['x'], upper_pipe['y']))
                self.screen.blit(self.images['pipe'], (lower_pipe['x'], lower_pipe['y']))
            self.screen.blit(self.images['base'], (0, self.screen_height * 0.79))
            
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.FPS)
        
        self.t += 1
        
        for i in range(len(self.lower_pipes)):
            if self.lower_pipes[i]['x'] + self.images['pipe'].get_width() > self.player_x:
                return [np.reshape([int(self.player_x - self.lower_pipes[i]['x']),
                        int(self.player_y - self.lower_pipes[i]['y'])], [1, 2]),
                        reward,
                        self.gameover]
        return [np.reshape([0, 0], [1, 2]), reward, self.gameover]
    
    def createPipe(self):
        r = random.randint(80, int(self.screen_height * 0.79) - 80 - 100)
        return [
            {'x': self.screen_width, 'y': r - self.images['pipe'].get_height()},
            {'x': self.screen_width, 'y': r + 100}
        ]
        

if __name__ == '__main__':
    game = Game()
    agent = Agent.DeepQLearningAgent(2, 2)
    agent.epsilon = 0
    agent.batch_size = 32
    agent.load_data()
    
    prev_state = np.reshape([0, 0], [1, 2])
    curr_state = np.reshape([0, 0], [1, 2])
    prev_action = 0
    curr_action = 0
    reward = 0
    terminal = False
    
    t = 0
    r = 0
    s = 0
    while True:
        if r % 100 in range(1) or s > 10:
            game.FPS = 60
        else:
            game.FPS = 60
            
        #action = pygame.key.get_pressed()[K_SPACE]
        prev_action = curr_action
        curr_action, q = agent.get_action(curr_state)
        
        prev_state = curr_state
        curr_state, reward, terminal = game.update(curr_action)
        
        agent.remember(prev_state, prev_action, reward, curr_state, terminal)

        if terminal:
            r = r + 1
            s = 0
            #agent.replay()
        
        if reward == 1:
            s = s + 1
        
        t = t + 1
        print(r, '\t', t, '\t', s, '\t', reward, terminal, '\t', q, '\t', agent.loss, '\t', agent.epsilon, curr_state)
        
        #if t % 1000 == 0:
            #agent.save_data()
