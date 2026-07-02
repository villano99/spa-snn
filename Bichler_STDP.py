import torch
import torch.nn as nn
from tqdm import tqdm
import math

class BichlerSTDP:
    """
    Implementación del algoritmo Unsupervised STDP de Bichler et al. 2012.
    Reglas clave:
    1. Si una neurona dispara (Post-Spike), las sinapsis con un Pre-Spike reciente ganan LTP (+alpha_plus).
    2. ABSOLUTAMENTE TODAS LAS DEMÁS SINAPSIS reciben LTD (-alpha_minus), logrando selectividad extrema.
    3. Inhibición Lateral Local (WTA): Si una neurona dispara, desactiva a sus vecinas espaciales por un tiempo.
    """
    def __init__(self, M1, N_in, mask, l1_per_cell, W_cells, H_cells, dt_ms=25.0):
        self.M1 = M1
        self.N_in = N_in
        self.mask = mask
        self.dt_ms = dt_ms
        
        # Parámetros Bichler
        self.w_min = 0.0
        self.w_max = 1.0
        self.alpha_plus = 0.05
        self.alpha_minus = 0.01  # Relación empírica ~1/5 de alfa plus
        
        self.T_LTP_steps = 3     # Tiempo de LTP (ej. 3 bins temporales)
        self.T_inhibit_steps = 2 # Tiempo que las vecinas se quedan "apagadas"
        
        # Estructura espacial para Inhibición Lateral (Winner-Take-All)
        self.l1_per_cell = l1_per_cell
        self.W_cells = W_cells
        self.H_cells = H_cells
        
        if self.W_cells * self.H_cells * self.l1_per_cell != self.M1:
            print("[WARNING] Mismatch en células espaciales para WTA, la inhibición será global por neurona.")
            self.spatial_wta = False
        else:
            self.spatial_wta = True

    def pretrain(self, model, dataloader, device, epochs=15):
        """
        Realiza el entrenamiento 100% no supervisado de la Capa 1 de la red (W1)
        alimentando todo el loader secuencialmente.
        """
        model.eval() # La red en modo inferencia, ajustamos pesos manualmente a bajo nivel
        
        # Inicializar W1 uniformemente en [0.8, 1.0] (Neuronas muy receptivas al inicio como dice Bichler)
        # Solo las sinapsis disponibles (en la máscara)
        with torch.no_grad():
            model.W1.data = torch.rand_like(model.W1.data) * 0.2 + 0.8
            model.W1.data *= self.mask
        
        print(f"[BICHLER-STDP] Iniciando Pre-Entrenamiento Unsupervised ({epochs} Epochs)")
        print(f"               Alpha+: {self.alpha_plus} | Alpha-: {self.alpha_minus} | T_LTP: {self.T_LTP_steps} bins")
        
        for ep in range(epochs):
            pbar = tqdm(dataloader, desc=f"STDP Epoch {ep+1}/{epochs}")
            total_spikes = 0
            
            for batch_idx, (x_batch, _, dims) in enumerate(pbar):
                # x_batch: (Batch, Time, N_in)
                x_batch = x_batch.to(device)
                B, T, N = x_batch.shape
                
                # Memoria de trazado
                # Tiempos del último pre-spike por píxel, inicializado muy atrás en el tiempo
                last_pre_spike = torch.full((B, N), -9999.0, device=device)
                
                # Estado del tiempo de inhibición lateral por neurona
                inhibition_timer = torch.zeros((B, self.M1), device=device)
                
                # Membranas de las neuronas L1 integrando bajo el capó
                mem = torch.zeros((B, self.M1), device=device)
                
                W1 = model.W1.data # (M1, N_in)
                
                # Iterar en el tiempo como un streaming biológico orgánico
                for t in range(T):
                    pre_spikes = x_batch[:, t, :] # (B, N_in)
                    
                    # Actualizar historial del prespike
                    has_spike = (pre_spikes > 0)
                    last_pre_spike[has_spike] = t
                    
                    # Decaimiento del voltaje (Leak, aprox 0.8 decaimiento constante)
                    mem = mem * 0.8 
                    
                    # Entrada sináptica instantánea: sumar W a los píxeles encendidos
                    # pre_spikes (B, N_in) * W1.T (N_in, M1) -> (B, M1)
                    input_current = torch.matmul(pre_spikes, W1.t())
                    
                    # Si la neurona está inhibida, ignora este evento (se apaga)
                    active_mask = (inhibition_timer <= 0).float()
                    mem = mem + (input_current * active_mask)
                    
                    # Chequear qué neuronas cruzaron su umbral y dispararon
                    # (Usamos el umbral orgánico que tenga el modelo)
                    threshold = getattr(model, 'threshold1', 1.0)
                    if not isinstance(threshold, float):
                        # Asumimos que model.lif1 existe y leemos the threshold desde la capa real si threshold1 no era atributo
                        if hasattr(model, 'lif1'):
                            t_val = getattr(model.lif1, 'threshold', 1.0)
                            if isinstance(t_val, torch.Tensor):
                                threshold = t_val.item()
                            else:
                                threshold = t_val
                        else:
                            threshold = 1.0
                            
                    post_spikes = (mem >= threshold).float()
                    
                    # Resetear las membranas que dispararon
                    mem[post_spikes > 0] = 0.0
                    
                    total_spikes += post_spikes.sum().item()
                    
                    # ----------------------------------------------------
                    # APLICAR APRENDIZAJE BICHLER STDP Y WTA
                    # ----------------------------------------------------
                    if post_spikes.sum() > 0:
                        # 1. LATERAL INHIBITION (WTA ESPACIAL)
                        if self.spatial_wta:
                            # Remapear spikes a la grilla [B, l1_per_cell, W_cells, H_cells]
                            ps_grid = post_spikes.view(B, self.l1_per_cell, self.W_cells, self.H_cells)
                            # Ver qué celdas espaciales tuvieron GANADORES en este batch
                            # Sumar por el eje de las neuronas que compiten (dim 1)
                            cell_activity = ps_grid.sum(dim=1, keepdim=True)
                            
                            # Si hubo actividad en esa celda espacial, silencias a TODA la celda 
                            # por 'T_inhibit_steps'
                            silence_mask = (cell_activity > 0).float().expand_as(ps_grid)
                            
                            # Achatamos de nuevo
                            silence_linear = silence_mask.reshape(B, self.M1)
                            inhibition_timer[silence_linear > 0] = self.T_inhibit_steps
                            
                            # Excepción: La neurona que realmente disparó no se inhibe hasta el *siguiente* frame
                        
                        # 2. WEIGHT UPDATES (Unsupervised Global LTD) - FULLY VECTORIZED
                        for b_idx in range(B):
                            if post_spikes[b_idx].sum() == 0:
                                continue
                                
                            # (N_in,)
                            time_diff = t - last_pre_spike[b_idx]
                            
                            # Valid causal window
                            is_ltp = (time_diff >= 0) & (time_diff <= self.T_LTP_steps)
                            is_ltd = ~is_ltp
                            
                            # Expand to (M1, N_in) virtual grid (Broadcasted on GPU)
                            # spiked_n is (M1, 1), valid_syn is (M1, N_in), is_ltp is (1, N_in)
                            spiked_n = (post_spikes[b_idx] > 0).unsqueeze(1)
                            valid_syn = self.mask > 0
                            
                            is_ltp_grid = is_ltp.unsqueeze(0)
                            is_ltd_grid = is_ltd.unsqueeze(0)
                            
                            final_ltp = is_ltp_grid & valid_syn & spiked_n
                            final_ltd = is_ltd_grid & valid_syn & spiked_n
                            
                            # Global inplace matrix scalar addition
                            W1[final_ltp] += self.alpha_plus
                            W1[final_ltd] -= self.alpha_minus
                            
                    inhibition_timer -= 1
                
                # CLIP WEIGHTS BOUNDARY
                model.W1.data.clamp_(self.w_min, self.w_max)
                model.W1.data *= self.mask
                
            pbar.set_postfix({'Total Spikes M1': f"{total_spikes}"})
            
        print("\n[BICHLER-STDP] Pre-Entrenamiento Terminado.")
