import numpy as np
import matplotlib.pyplot as plt
import random
import pygame
import sys 
import time
# Hyperparameters
ALPHA = 0.08
GAMMA = 0.9
EPSILON = 1
EPSILON_DECAY = 0.99995 
EPOCHS = 2000

STEP_SIZE = 10
BOB_VELX = 5
BOB_VELY = 5
# Environment dimensions
WIDTH = 400
HEIGHT = 400

FPS = 1500
old_slope = 0



STEPS = 1000
NUM_BINS = [20, 20]  

height = 10
width = 50




# Initialize Q-table
try:
    Q_table = np.load('Q_table_for_test2.npy')
    print('Loaded')
except:
    Q_table = np.zeros((NUM_BINS[0], NUM_BINS[0], NUM_BINS[1], NUM_BINS[0], NUM_BINS[1], 2))
class MyEnv():
    def __init__(self):
        self.x, self.y = None, None 
        self.bob_x, self.bob_y = None, None 
        self.threshold = 20
        self.vx, self.vy = BOB_VELX, BOB_VELY

    def reset(self):

        self.x, self.y = random.randint(0, WIDTH), HEIGHT - 50
        self.bob_x, self.bob_y = random.randint(0, WIDTH), 0
        return [self.x, self.bob_x, self.bob_y, self.vx, self.vy]

    def choose_action(self, discrete_state):

        if random.random() < EPSILON:
            return random.randint(0, 1) 
        else:
            return np.argmax(Q_table[discrete_state])

    def get_reward(self, state):
        d_x = abs(self.x - self.bob_x)
        rew = 0
        #rew = 0
        if abs(self.bob_x - self.x) < width/2 and abs(self.bob_y - self.y) < 5:
            rew += 100
        if self.bob_y > self.y + 10:
            rew -= 100
        return rew

    def step(self, action):
        if action == 0:  # move left
            self.x = max(0, self.x - STEP_SIZE)
        elif action == 1:  # move right
            self.x = min(WIDTH, self.x + STEP_SIZE)

        
        self.bob_x += self.vx
        self.bob_y += self.vy 

        if self.bob_x > WIDTH or self.bob_x < 0:
            self.vx *= -1
        if self.bob_y > HEIGHT or self.bob_y < 0:
            self.vy *= -1
        if abs(self.bob_x - self.x) < width/2 and abs(self.bob_y - self.y) < 1:
            self.vy *= -1
        
        new_state = [self.x, self.bob_x, self.bob_y, self.vx, self.vy]
        reward = self.get_reward(new_state)

        if self.bob_y > WIDTH - 5:
            done = True
        else:
            done = False

        return new_state, reward, done

def create_bins():
    x_bins = np.linspace(0, WIDTH, NUM_BINS[0] - 1)
    #y_bins = np.linspace(0, HEIGHT, NUM_BINS[1] - 1)
    bobx_bins = np.linspace(0, WIDTH, NUM_BINS[0] - 1)
    boby_bins = np.linspace(0, HEIGHT, NUM_BINS[1] - 1)
    vx_bins  =  np.linspace(0, WIDTH, NUM_BINS[0] - 1)
    vy_bins = np.linspace(0, HEIGHT, NUM_BINS[1] - 1)
    return [x_bins, bobx_bins, boby_bins, vx_bins, vy_bins]

def discretize_state(state):
    BINS = create_bins()
    x, bob_x, bob_y, vx, vy = state 
    idx_x = np.digitize(x, BINS[0])
    idx_bobx = np.digitize(bob_x, BINS[1])
    idx_boby = np.digitize(bob_y, BINS[2])
    idx_vx = np.digitize(vx, BINS[3])
    idx_vy = np.digitize(vy, BINS[4])

    idx_x = min(max(idx_x, 0), NUM_BINS[0] - 1)
    idx_bobx = min(max(idx_bobx, 0), NUM_BINS[0] - 1)
    idx_boby = min(max(idx_boby, 0), NUM_BINS[1] - 1)
    idx_vx = min(max(idx_vx, 0), NUM_BINS[0] - 1)
    idx_vy = min(max(idx_vy, 0), NUM_BINS[1] - 1)



    return (idx_x, idx_bobx, idx_boby, idx_vx, idx_vy)
toggle = True

def training(env: MyEnv):
    global EPSILON, FPS, toggle, height, width  

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    for epoch in range(1, EPOCHS + 1):
        step = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                np.save('Q_table_for_test2.npy', Q_table)
                print('Saved')
                sys.exit()
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    print(toggle)
                    toggle = not toggle
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    if toggle:
                        EPSILON += 0.001
                    else:
                        FPS += 10
                elif event.button == 5:
                    if toggle:
                        EPSILON -= 0.001
                    else:
                        FPS -= 10
                
        if not running:
            break
        state = env.reset()
        discrete_state = discretize_state(state)
        total_rewards = 0
        if toggle:
            color = (255, 50, 0)
        else:
            color = (0, 50, 250)
        
        done = False

        while not done and step < STEPS:
            step += 1
            action = env.choose_action(discrete_state)
            next_state, reward, done = env.step(action)
            new_discrete_state = discretize_state(next_state)


            max_future_q = np.max(Q_table[new_discrete_state])
            current_q = Q_table[discrete_state + (action,)]
            Q_table[discrete_state + (action,)] = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)

            total_rewards += reward
            discrete_state = new_discrete_state


            screen.fill((255, 255, 255))
            agent = pygame.Rect(state[0] - width/2, env.y - height/2, width, height)
            pygame.draw.rect(screen, (0, 250, 50), agent, 5)
            pygame.draw.circle(screen, color, (state[1], state[2]), 5)
            pygame.display.set_caption(f"E: {epoch},  R: {total_rewards:.1f} Ep: {EPSILON:.2f} F: {FPS} T: {20 + env.threshold:.2f}")
            pygame.display.flip()
            clock.tick(FPS)

            state = next_state  # update current state

        # Decay epsilon after each episode, with a floor of 0.05
        EPSILON = EPSILON * EPSILON_DECAY

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Total Reward: {total_rewards}')
            

        if EPSILON < 0.01 or epoch == EPOCHS:
            np.save('Q_table_for_test2.npy', Q_table)
            print('Saved')
            break

    pygame.quit()


if __name__ == "__main__":
    env = MyEnv()
    training(env)
    

