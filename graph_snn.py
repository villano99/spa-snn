
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F

class GraphSNN(nn.Module):
    def __init__(self, input_size=(2, 128, 128), num_classes=4, 
                 l1_per_cell=2, l2_size=128, l3_size=64,
                 beta=0.9, r1=5.0, r2=8.0, r3=12.0, device='cuda'):
        super().__init__()
        self.input_size = input_size
        self.C, self.H, self.W = input_size
        self.num_classes = num_classes
        self.device = device
        
        # Parametros de las neuronas
        spike_grad = surrogate.atan()
        
        # --- Capa 1: Conectividad Local (Entrada -> L1) ---
        # Tamano de cuadricula: Dividimos la imagen en celdas.
        # Si la entrada es 128x128 y queremos que L1 la cubra.
        # Original logic: Cell grid was implied by `generate_input_masks`.
        # Aqui lo hacemos agnostico a la resolucion.
        # Asumimos que L1 tiene N neuronas distribuidas espacialmente.
        # L1 no es fully connected para ahorrar memoria.
        # Enfoque: Grafo Aleatorio con Mascaras Locales.
        
        # Numero de neuronas L1
        # Or just fixed number?
        # Original: 1000 input cells -> 2000 L1 neurons. Input was 160x120. cells were 4x4.
        # So 40x30 = 1200 cells.
        # Let's define grid size dynamically.
        self.grid_size = 4 # Downsample input by 4
        self.rows = self.H // self.grid_size
        self.cols = self.W // self.grid_size
        self.num_input_cells = self.rows * self.cols
        
        self.num_l1 = self.num_input_cells * l1_per_cell
        self.num_l2 = l2_size
        self.num_l3 = l3_size
        
        print(f"Building GraphSNN: In={self.H}x{self.W} -> Grid={self.rows}x{self.cols} -> L1={self.num_l1}")

        # Layers
        self.fc1 = nn.Linear(self.num_input_cells * self.C, self.num_l1) # Sparse masked later
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(self.num_l1, self.num_l2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc3 = nn.Linear(self.num_l2, self.num_l3)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_out = nn.Linear(self.num_l3, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True) # Reset mechanism?
        
        # Connectivity Masks
        self.mask1 = self._create_l1_mask(r1)
        self.mask2 = self._create_geometric_mask(self.num_l1, self.num_l2, r2, level=1)
        self.mask3 = self._create_geometric_mask(self.num_l2, self.num_l3, r3, level=2)
        
        # Apply masks
        with torch.no_grad():
            self.fc1.weight.data *= self.mask1.to(self.fc1.weight.device)
            self.fc2.weight.data *= self.mask2.to(self.fc2.weight.device)
            self.fc3.weight.data *= self.mask3.to(self.fc3.weight.device)

    def _create_l1_mask(self, radius):
        # Map Input Cells (Row, Col) to L1 Neurons
        # Input: (Rows * Cols) * Channels
        # Output: (Rows * Cols) * L1_Cell_Density
        mask = torch.zeros(self.num_l1, self.num_input_cells * self.C)
        
        # Assign L1 neurons to spatial positions
        # Simple strategy: L1 neuron i corresponds to Cell (i // density)
        # It connects to neighbors of that cell.
        
        inp_h, inp_w = self.rows, self.cols
        l1_per_cell = self.num_l1 // self.num_input_cells
        
        for i in range(self.num_l1):
            cell_idx = i // l1_per_cell
            cy, cx = divmod(cell_idx, inp_w)
            
            # Connect to inputs within radius (in terms of cells)
            # If radius=5.0 (pixels), and grid=4, radius_cells ~ 1.25
            r_cells = radius / self.grid_size
            
            # Iterate input cells
            # Optimization: only check nearby
            y_min = max(0, int(cy - r_cells - 1))
            y_max = min(inp_h, int(cy + r_cells + 2))
            x_min = max(0, int(cx - r_cells - 1))
            x_max = min(inp_w, int(cx + r_cells + 2))
            
            for r in range(y_min, y_max):
                for c in range(x_min, x_max):
                    dist = ((r - cy)**2 + (c - cx)**2)**0.5
                    if dist <= r_cells:
                        inp_idx = r * inp_w + c
                        # Connect both polarities (2 channels)
                        mask[i, inp_idx*2] = 1.0
                        mask[i, inp_idx*2+1] = 1.0
        return mask

    def _create_geometric_mask(self, n_in, n_out, radius, level=1):
        """
        Crea máscara sparse basada en coordenadas espaciales aleatorias.
        Neuronas se conectan solo si la distancia normalizada < radius_norm.
        Bio-plausible: preserva localidad topológica.
        """
        import numpy as np
        
        # Asignar coordenadas 2D a neuronas de entrada y salida
        # Fijar semilla para reproducibilidad por nivel
        rng = np.random.RandomState(42 + level)
        
        pos_in = rng.rand(n_in, 2)   # Posiciones en [0,1]²
        pos_out = rng.rand(n_out, 2)
        
        # Radio normalizado: radius / max_grid_dim
        max_dim = max(self.rows, self.cols)
        r_norm = radius / max_dim
        
        # Calcular distancias y crear máscara
        mask = torch.zeros(n_out, n_in)
        for i in range(n_out):
            dx = pos_in[:, 0] - pos_out[i, 0]
            dy = pos_in[:, 1] - pos_out[i, 1]
            dist = np.sqrt(dx**2 + dy**2)
            connected = dist <= r_norm
            # Asegurar mínimo de conexiones (al menos 5% de n_in)
            if connected.sum() < max(1, n_in // 20):
                topk = min(n_in // 20 + 1, n_in)
                closest = np.argsort(dist)[:topk]
                connected[closest] = True
            mask[i, torch.from_numpy(connected)] = 1.0
        
        sparsity = 1.0 - mask.sum().item() / (n_out * n_in)
        print(f"  Mask L{level+1}: {n_in}->{n_out}, sparsity={sparsity:.1%}")
        return mask

    def preprocess_input(self, x):
        """
        Pools and flattens input for the network.
        x: (Time, Batch, 2, H, W) — binary spikes (0/1)
        Returns: (Time, Batch, N_input) — binary spikes preservados
        """
        T, B, C, H, W = x.shape
        x_flat = x.reshape(T*B, C, H, W)
        # max_pool2d preserva spikes: si hay spike en la región → 1
        x_pooled = F.max_pool2d(x_flat, self.grid_size)
        x_vec = x_pooled.reshape(T, B, -1)
        return x_vec

    def forward(self, x, return_internals=False):
        # x: (Time, Batch, 2, H, W)
        
        T, B, C, H, W = x.shape
        # Max pooling: preserva spikes binarios en la región
        x_flat = x.reshape(T*B, C, H, W)
        x_pooled = F.max_pool2d(x_flat, self.grid_size)
        x_vec = x_pooled.reshape(T, B, -1)
        
        # Initialize
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        spk_out_rec = []
        spk1_rec = []     # For STDP
        spk3_rec = []     # For SPA Classifier
        x_vec_rec = []    # For STDP (Pre-synaptic L1)
        
        for step in range(T):
            cur_in = x_vec[step]
            cur1 = self.fc1(cur_in)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            cur_out = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)
            
            if return_internals:
                spk1_rec.append(spk1)
                spk3_rec.append(spk3)
                x_vec_rec.append(cur_in)
            
        out = torch.stack(spk_out_rec, dim=0)
        
        if return_internals:
             # Return stacks for STDP/SPA processing
             return out, torch.stack(spk1_rec, dim=0), torch.stack(spk3_rec, dim=0), torch.stack(x_vec_rec, dim=0)
             
        return out

