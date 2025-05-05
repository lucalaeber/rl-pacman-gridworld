import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import pygame
import random
import os

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.grid_size = (10, 10) #y,x

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Four actions: Move Up, Down, Left or Right
        self.observation_space = spaces.Box(low=0, high=9, shape=(2,), dtype=np.float32)

        # Define walls using coordinate pairs (start, end)
        # Walls: Designed to force narrow corridors and block shortcuts.
        self.walls = {
            # Bottom section (near agent start)
            ((8, 0), (8, 1)), ((8, 1), (8, 2)), ((8, 2), (8, 3)),
            ((7, 0), (6, 0)), ((6, 0), (6, 1)), ((6, 1), (6, 2)),
            ((7, 1), (7, 2)), ((7, 2), (7, 3)), ((7, 3), (7, 4)),
            ((7, 4), (7, 5)), ((6, 5), (6, 6)),
        
            # Middle section
            ((6, 3), (5, 3)), ((5, 3), (5, 4)), ((5, 4), (4, 4)), ((4, 4), (4, 5)),
            ((4, 5), (4, 6)), ((4, 6), (4, 7)), ((4, 7), (3, 7)), ((3, 6), (3, 7)),
            ((3, 5), (3, 6)), ((3, 4), (3, 5)), ((3, 3), (3, 4)), ((4, 3), (4, 4)),
            ((5, 5), (5, 6)), ((5, 6), (4, 6)),
        
            # Upper-left section
            ((2, 2), (2, 3)), ((1, 2), (2, 2)), ((1, 2), (1, 3)), ((1, 3), (0, 3)),
            ((0, 3), (0, 4)), ((0, 4), (0, 5)), ((0, 5), (0, 6)), ((0, 6), (0, 7)),
            ((0, 2), (0, 3)), ((0, 1), (0, 2)), ((0, 0), (1, 0)), ((1, 0), (1, 1)),
            ((1, 1), (1, 2)),
        
            # Near goal (top-right)
            ((1, 6), (1, 7)), ((1, 7), (1, 8)), ((1, 8), (1, 9)), ((2, 6), (2, 7)),
            ((2, 7), (2, 8)), ((2, 8), (2, 9)), ((3, 8), (3, 9)), ((4, 8), (4, 9)),
            ((6, 8), (6, 9)), ((7, 8), (7, 9)), ((8, 8), (7, 8)), ((9, 2), (9, 3)),
            ((9, 1), (8, 1))
                    }


        # Monsters:
        self.monsters = {
            (3, 7),  
            (8, 2),  
            (5, 4),  
            (1, 6),  
            (6, 8),  
            (2, 4),  
            (4, 6),  
            (7, 5)   
        }
        self.monster_appearances = {}
        for pos in self.monsters:
            monster_num = random.randint(1, 4)
            self.monster_appearances[pos] = monster_num
        
        # Endpoint top right (0,9)
        self.endpoint = (0, 9)

        # Initialize state bottom left
        self.state = (9, 0)

        self.cell_size = 70

        # Pygame setup
        pygame.init()
        self.width, self.height = self.grid_size[1] * self.cell_size, self.grid_size[0] * self.cell_size #1920, 1080 # Window size
        self.window = pygame.display.set_mode((self.width, self.height))
        self.window = pygame.display.set_mode((self.width, self.height))  # This must come first
        self.pacman_img = pygame.image.load("pacman.png").convert_alpha()
        self.pacman_img = pygame.transform.scale(self.pacman_img, (40, 40))
        self.monster_img1 = pygame.image.load("monster1.png").convert_alpha()
        self.monster_img1 = pygame.transform.scale(self.monster_img1, (40, 40))
        self.monster_img2 = pygame.image.load("monster2.png").convert_alpha()
        self.monster_img2 = pygame.transform.scale(self.monster_img2, (40, 40))
        self.monster_img3 = pygame.image.load("monster3.png").convert_alpha()
        self.monster_img3 = pygame.transform.scale(self.monster_img3, (40, 40))
        self.monster_img4 = pygame.image.load("monster4.png").convert_alpha()
        self.monster_img4 = pygame.transform.scale(self.monster_img4, (40, 40))
        pygame.display.set_caption("Pac-Man Environment")
        self.clock = pygame.time.Clock()
        self.reset()
        
        
        # Build the transition model P for all states and actions.
        self.P = {}
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                s = (y, x)
                self.P[s] = {}
                for a in range(self.action_space.n):
                    self.P[s][a] = []
                    prob, next_state, reward, done = self.get_transition(s, a)
                    self.P[s][a].append((prob, next_state, reward, done))
        
    
    def get_transition(self, s, action):
        """
        Given a state (as a tuple (y, x)) and an action,
        compute the deterministic outcome: probability, next_state, reward, and done.
        """
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dy, dx = moves[action]
        y, x = s
        new_y = y + dy
        new_x = x + dx
    
        # Check for walls (if the move is blocked, remain in the same state)
        if ((y, x), (new_y, new_x)) in self.walls or ((new_y, new_x), (y, x)) in self.walls:
            next_state = s
            reward = -2  # Penalty for hitting a wall
            done = False
        else:
            # Check for out-of-bounds: if so, stay in the same state
            if new_y < 0 or new_y >= self.grid_size[0] or new_x < 0 or new_x >= self.grid_size[1]:
                next_state = s
                reward = -1
                done = False
            else:
                next_state = (new_y, new_x)
                reward = -1  # Base movement cost
                done = False
                if next_state in self.monsters:
                    reward = -5
                    done = False
                elif next_state == self.endpoint:
                    reward = 100
                    done = True
        return 1.0, next_state, reward, done
        

    def reset(self, seed=None, options=None):
        """ Reset the environment to an initial state. """
        self.state = np.array([9.0, 0.0])
        self.done = False
        
        self.dots = {(y, x) for y in range(self.grid_size[0]) 
                              for x in range(self.grid_size[1]) 
                              if (y, x) not in self.monsters and (y, x) != self.endpoint and (y, x) != (9, 0)}
        return np.array(self.state), {}


    def step(self, action):
        """ Apply an action and return results. """

        # Define movement directions
        moves = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        dy, dx = moves[action]
        # Store current position before movement
        prev_state = self.state.copy()
        new_y = self.state[0] + dy
        new_x = self.state[1] + dx

        # Check if movement is blocked by a wall
        if ((self.state[0], self.state[1]), (new_y, new_x)) in self.walls or ((new_y, new_x), (self.state[0], self.state[1])) in self.walls:
            reward = -2  # Large penalty for hitting a wall
            new_y, new_x = prev_state  # Stay in place
        else:
            reward = -1  # Small penalty for normal movement
            self.state = np.array([new_y, new_x])  # Move if not blocked
            
        current_cell = (int(self.state[0]), int(self.state[1]))
        if current_cell in self.dots:
            self.dots.remove(current_cell)  # Remove dot when eaten


        # Check if Pacman encounters a monster (Game Over)
        if (new_y, new_x) in self.monsters:
            reward = -5  # Large penalty for hitting a monster
            done = False
        # Check if Pacman reaches the endpoint (Win Condition)
        elif (new_y, new_x) == self.endpoint:
            reward = 100  # Large reward for reaching goal
            done = True
        elif new_y < 0 or new_y > self.grid_size[0]-1 or new_x < 0 or new_x > self.grid_size[1]-1:
            reward = -1
            self.state = prev_state
            done = False
        else:
            done = False
            

        return np.array(self.state), reward, done, False, {}


    def render(self):
        """ Render the environment using Pygame. """
        # Draw an agent (circle)
        
        BG = (25, 25, 25)
        self.window.fill(BG)
        

        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                
                if (y, x) == self.endpoint:
                    color = (0, 255, 0)  # Endpoint (Green)
                else:
                    color = (50, 50, 50)  # Empty spaces (Dark Gray)
                
                pygame.draw.rect(self.window, color, rect)  # Empty grid cells
                pygame.draw.rect(self.window, (255, 255, 255), rect, 1)  # White grid lines


        # Only draw dots that remain in self.dots
        dot_radius = 3
        dot_color = (255, 165, 0)
        for y, x in self.dots:
            dot_center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.window, dot_color, dot_center, dot_radius)
                

        
        # Draw walls as lines
        for (y1, x1), (y2, x2) in self.walls:
            # Rotate the walls for correct representation
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            new_x1 = mid_x - (y1 - mid_y)
            new_y1 = mid_y + (x1 - mid_x)
            new_x2 = mid_x - (y2 - mid_y)
            new_y2 = mid_y + (x2 - mid_x)

            start_pos = (int(new_x1 * self.cell_size + self.cell_size // 2), int(new_y1 * self.cell_size + self.cell_size // 2))
            end_pos = (int(new_x2 * self.cell_size + self.cell_size // 2), int(new_y2 * self.cell_size + self.cell_size // 2))

            pygame.draw.line(self.window, (0, 0, 255), start_pos, end_pos, 5)  # Blue rotated wall

            
        # Draw monsters 
        for y, x in self.monsters:
            monster_x = int(x * self.cell_size + self.cell_size // 2)
            monster_y = int(y * self.cell_size + self.cell_size // 2)

            monster_num = self.monster_appearances[(y, x)]
            if monster_num == 1:
                image_rect = self.monster_img1.get_rect(center=(monster_x, monster_y))
                self.window.blit(self.monster_img1, image_rect)
            elif monster_num == 2:
                image_rect = self.monster_img2.get_rect(center=(monster_x, monster_y))
                self.window.blit(self.monster_img2, image_rect)
            elif monster_num == 3:
                image_rect = self.monster_img3.get_rect(center=(monster_x, monster_y))
                self.window.blit(self.monster_img3, image_rect)
            elif monster_num == 4:
                image_rect = self.monster_img4.get_rect(center=(monster_x, monster_y))
                self.window.blit(self.monster_img4, image_rect)
            else:
                raise Exception("Invalid monster image index")

        
        agent_x = int(self.state[1] * self.cell_size + self.cell_size // 2)
        agent_y = int(self.state[0] * self.cell_size + self.cell_size // 2)
        image_rect = self.pacman_img.get_rect(center=(agent_x, agent_y))
        self.window.blit(self.pacman_img, image_rect)



        # Update display
        pygame.display.flip()
        pygame.time.wait(1000)
        self.clock.tick(30)  # Limit FPS to 30

    def close(self):
        """ Clean up Pygame resources. """
        pygame.quit()


