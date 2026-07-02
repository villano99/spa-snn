import numpy as np
import random
import torch

class SaccadeAugmenter:
    """
    Motor bio-plausible de Aumento de Datos para Eventos DVS.
    Simula Microsácados (twitches oculares) y oclusión natural sin romper la
    esparsidad matemática de la nube de eventos neuromórficos.
    """
    def __init__(self, W_target=160, H_target=120, max_shift_x=10, max_shift_y=10, flip_prob=0.5, current_ds=2):
        self.W = W_target
        self.H = H_target
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y
        self.flip_prob = flip_prob
        self.current_ds = current_ds # El downsample actual
        
    def __call__(self, x_coords, y_coords, t, p):
        """
        Interviene la nube de puntos cruda (x, y, t, p) antes de que se digitalice en bins.
        """
        # --- 1. Sácados Aleatorios (Translación Espacial Jitter) ---
        # Calculamos cuánto "temblará" el ojo en este video en específico
        shift_x = random.randint(-self.max_shift_x, self.max_shift_x) * self.current_ds
        shift_y = random.randint(-self.max_shift_y, self.max_shift_y) * self.current_ds
        
        # Como los coords crudos vienen en la resolución Original (ds=1), el shift se aplica 
        # asumiendo esa escala para que luego el downsample lo comprima suavemente.
        
        x_aug = x_coords + shift_x
        y_aug = y_coords + shift_y
        
        # --- 2. Inversión Estructural (Mirroring Biológico) ---
        if random.random() < self.flip_prob:
            # W_orig es asumido como W * ds
            W_orig = self.W * self.current_ds
            x_aug = W_orig - 1 - x_aug
            
        # Retornamos los tensores sin filtrar bordes todavía. 
        # El Dataset se encargará del clipping en la función __getitem__ clásica.
        return x_aug, y_aug, t, p
