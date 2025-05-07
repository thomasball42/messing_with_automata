import numpy as np
import pygame
import matplotlib.cm as cm
from typing import Callable, Tuple, Optional
import random

class AdvancedCellularAutomaton:
    def __init__(self, 
                 width: int = 800, 
                 height: int = 600, 
                 cell_size: int = 10,
                 state_range: Tuple[float, float] = (0.0, 1.0),
                 cmap_name: str = 'viridis',
                 fps: int = 10):

        # Display parameters
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.fps = fps
        
        self.state_min, self.state_max = state_range
        self.current_grid = np.random.uniform(
            low=self.state_min, 
            high=self.state_max, 
            size=(self.rows, self.cols))
        self.next_grid = np.zeros_like(self.current_grid)
        
        self.cmap = cm.get_cmap(cmap_name)
        self.cmap.set_bad(color=(0, 0, 0))  # Color for values outside range
        
        self.neighborhood_func = self.get_neighborhood
        self.rule_func = self.default_diffusion_rule
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Advanced Cellular Automaton")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
    
    def initialize_grid(self, mode: str = 'random', **kwargs):
        """Initialize the grid state"""
        if mode == 'random':
            self.current_grid = np.random.uniform(
                low=self.state_min,
                high=self.state_max,
                size=(self.rows, self.cols))
        elif mode == 'uniform':
            value = kwargs.get('value', self.state_min)
            self.current_grid = np.full((self.rows, self.cols), value)
        # elif mode == 'gradient':
        #     # Horizontal gradient
        #     self.current_grid = np.linspace(
        #     self.state_min, 
        #     self.state_max, 
        #     self.cols).reshape(1, -1).repeat(self.rows, 0)
        self.next_grid = np.zeros_like(self.current_grid)
    
    def get_neighborhood(self, row: int, col: int, radius: int = 1) -> np.ndarray:
        neighbors = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i == 0 and j == 0:
                    continue  # Skip center cell
                r, c = row + i, col + j
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    neighbors.append(self.current_grid[r, c])
        return np.array(neighbors)
    
    def default_diffusion_rule(self, cell_value: float, neighbors: np.ndarray) -> float:
        ret_val = 0.9 * cell_value + 0.1 * np.mean(neighbors)
        return ret_val 
    
    def X_diffusion_rule(self, cell_value: float, neighbors: np.ndarray, attenuation: float = 0.03) -> float:
        live_neighbors = sum(neighbors)
        if 0.5 < cell_value < 1.5:
            valx = np.random.poisson(4)/4 if 1.5 < live_neighbors < 3.5 else 0
        else:
            valx = np.random.poisson(9)/9 if live_neighbors > 3 else 0
        ret_val = (1-attenuation) * valx
        
        return ret_val 

    
    def set_rule(self, rule_func: Callable[[float, np.ndarray], float]):
        """Set the rule function"""
        self.rule_func = rule_func
    
    def set_colormap(self, cmap_name: str):
        """Change the colormap"""
        self.cmap = cm.get_cmap(cmap_name)
    
    def update(self):
        """Update the grid according to the current rule"""
        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = self.current_grid[row, col]
                neighbors = self.neighborhood_func(row, col)
                self.next_grid[row, col] = self.rule_func(cell_value, neighbors)
        
        self.current_grid, self.next_grid = self.next_grid, self.current_grid
    
    def draw(self):
        """Draw the current grid state with colormap"""
        # Normalize values to [0,1] for colormap
        normalized = (self.current_grid - self.state_min) / (self.state_max - self.state_min)
        normalized = np.clip(normalized, 0, 1)
        
        # Convert to RGB using colormap
        rgba_colors = (self.cmap(normalized) * 255).astype(np.uint8)
        
        surf = pygame.surfarray.make_surface(rgba_colors[..., :3])
        surf = pygame.transform.scale(surf, (self.width, self.height))
        self.screen.blit(surf, (0, 0))
        
        # Display info
        info = f"Rule: {self.rule_func.__name__} | States: {self.state_min:.2f}-{self.state_max:.2f}"
        text = self.font.render(info, True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        
        pygame.display.flip()
    
    def run(self):
        """Run the simulation"""
        running = True
        paused = False
        
        self.initialize_grid(mode='random')
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        self.initialize_grid(mode='random')
                    elif event.key == pygame.K_c:
                        self.initialize_grid(mode='uniform', value=self.state_min)
                    # elif event.key == pygame.K_g:
                    #     self.initialize_grid(mode='gradient')
                        
                    elif event.key == pygame.K_m:
                        current_cmap = self.cmap.name
                        cmaps = ['viridis', 'plasma', 'magma', 'coolwarm', 'twilight']
                        next_cmap = cmaps[(cmaps.index(current_cmap) + 1) % len(cmaps)] if current_cmap in cmaps else cmaps[0]
                        self.set_colormap(next_cmap)
            
            if not paused:
                self.update()
            
            self.draw()
            self.clock.tick(self.fps)
        
        pygame.quit()
    
    def _resize_grid(self):
        """Resize grid when cell size changes"""
        new_cols = self.width // self.cell_size
        new_rows = self.height // self.cell_size
        
        # Create new grid with interpolated values
        from scipy.ndimage import zoom
        zoom_factor = (new_rows / self.rows, new_cols / self.cols)
        self.current_grid = zoom(self.current_grid, zoom_factor, order=1)
        self.next_grid = np.zeros_like(self.current_grid)
        
        self.cols, self.rows = new_cols, new_rows

if __name__ == "__main__":
    automaton = AdvancedCellularAutomaton(
        width=800,
        height=600,
        cell_size=7,
        state_range=(0.0, 1.0),
        cmap_name='inferno',
        fps=60
    )
    
    automaton.set_rule(automaton.X_diffusion_rule)
    automaton.run()