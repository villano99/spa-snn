import torch
import torch.nn as nn
import numpy as np

class R_STDP(nn.Module):
    """
    Reward-Modulated STDP (R-STDP) Robust implementation.
    
    Can function in two modes:
    1. Unsupervised (Classic STDP): Hebbian learning driven by correlation.
    2. Reward-Modulated (R-STDP): Updates scaled by a global reward signal.
       - Reward > 0: LTP (Strengthen causal connections)
       - Reward < 0: Anti-LTP (Weaken causal connections that led to the error)
       
    Features:
    - Trace-based implementation (efficient).
    - Homeostasis (Adaptive Thresholds or Weight Normalization).
    - Stochastic Re-seeding for dead neurons.
    - Sparse Mask support.
    """
    
    def __init__(self, 
                 learning_rate=1e-3,
                 w_max=1.0,
                 w_min=0.001,
                 tau_pre=20.0, 
                 tau_post=20.0,
                 a_plus=1.0, 
                 a_minus=0.8,
                 homeostasis_rate=0.001,
                 target_rate=0.05):
        super().__init__()
        
        self.lr = learning_rate
        self.w_max = w_max
        self.w_min = w_min
        
        # Constantes temporales para los traces
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.pre_decay = np.exp(-1.0 / tau_pre)
        self.post_decay = np.exp(-1.0 / tau_post)
        
        # Amplitudes de STDP
        self.a_plus = a_plus
        self.a_minus = a_minus
        
        # Homeostasis
        self.homeostasis_rate = homeostasis_rate
        self.target_rate = target_rate
        
        # Estado (Traces)
        self.trace_pre = None
        self.trace_post = None
        
        # Contadores para homeostasis
        self.activity_monitor = None
        self.steps_monitored = 0

    def reset_state(self):
        self.trace_pre = None
        self.trace_post = None
        
    def init_traces(self, batch_size, n_in, n_out, device):
        if self.trace_pre is None or self.trace_pre.shape[0] != batch_size:
            self.trace_pre = torch.zeros(batch_size, n_in, device=device)
            self.trace_post = torch.zeros(batch_size, n_out, device=device)

    def compute_update(self, pre_spikes, post_spikes, weights, mask=None, reward=None):
        """
        Calcula el delta_w con opción de Reward Modulation intra-batch.
        """
        batch_size = pre_spikes.size(0)
        device = pre_spikes.device
        
        # 1. Actualizar Trazas
        self.init_traces(batch_size, pre_spikes.size(1), post_spikes.size(1), device)
        
        # x_trace(t) = x_trace(t-1) * decay + spike(t)
        self.trace_pre = self.trace_pre * self.pre_decay + pre_spikes
        self.trace_post = self.trace_post * self.post_decay + post_spikes
        
        # 2. Modulate Traces with Reward if present
        # R-STDP: dw ~ Sum_b( Post[b] * PreTrace[b] * Reward[b] )
        tp_pre = self.trace_pre
        tp_post = self.trace_post
        
        if reward is not None:
            # Reward shape check: [B, 1] o [B]
            if reward.dim() == 1:
                reward = reward.view(-1, 1)
            
            # Modulamos las trazas (o los spikes, es equivalente matemáticamente para el producto)
            # LTP: Post * (PreTrace * Reward)
            tp_pre = tp_pre * reward
            
            # LTD: (PostTrace * Reward) * Pre
            tp_post = tp_post * reward

        # 3. Compute DW (Sum over batch via Matrix Mult)
        # LTP: [Out, Batch] @ [Batch, In] -> [Out, In]
        dw_ltp = torch.matmul(post_spikes.t(), tp_pre)
        
        # LTD: [Out, Batch] @ [Batch, In]
        dw_ltd = torch.matmul(tp_post.t(), pre_spikes)
        
        dw = (self.a_plus * dw_ltp) - (self.a_minus * dw_ltd)
        
        # Aplicar máscara
        if mask is not None:
            dw *= mask
            
        return dw / batch_size # Promedio

    def step(self, weights, pre_spikes, post_spikes, reward=1.0, mask=None):
        """
        Paso de aprendizaje R-STDP.
        """
        with torch.no_grad():
            # Si reward es escalar (1.0), lo pasamos tal cual (None iteraría si tratamos diferente)
            # Pero nuestra lógica soporta tensores. 
            # Si es 1.0 (float), compute_update lo ignora (None/Check) o multiplicamos.
            
            rw_tensor = None
            scalar_rw = 1.0
            
            if isinstance(reward, torch.Tensor):
                rw_tensor = reward
            elif isinstance(reward, (float, int)) and reward != 1.0:
                 # Si es un escalar global != 1.0, lo aplicamos al final
                 scalar_rw = reward
            
            delta_w = self.compute_update(pre_spikes, post_spikes, weights, mask, reward=rw_tensor)
            
            # Update
            update = delta_w * self.lr * scalar_rw
            
            # [DEBUG] Check if we are learning
            if self.steps_monitored % 100 == 0: 
                 # Print every ~2 batches (steps_monitored increments by batch size)
                 # Actually steps_monitored accumulates samples. 
                 # Let's use specific counter or just print if random.
                 # Better: Print stats
                 spikes_count = post_spikes.sum().item()
                 dw_mag = update.abs().mean().item()
                 print(f"[R-STDP DEBUG] Batch Spikes: {spikes_count} | Mean dW: {dw_mag:.8f} | W range: {weights.min():.4f}-{weights.max():.4f}")

            weights.add_(update)
            weights.clamp_(self.w_min, self.w_max)
            
            # Monitoring
            if self.activity_monitor is None:
                self.activity_monitor = torch.zeros(weights.size(0), device=weights.device)
            self.activity_monitor += post_spikes.sum(dim=0)
            self.steps_monitored += post_spikes.size(0)
            
            # [LAZY NORMALIZATION] Enforce competition every 50 steps (approx 1 batch)
            # This prevents all weights from saturating to w_max
            if (self.steps_monitored // post_spikes.size(0)) % 50 == 0:
                 if mask is not None:
                    W_masked = weights * mask.float()
                    norms = torch.norm(W_masked, p=2, dim=1, keepdim=True)
                    weights.copy_(W_masked / (norms + 1e-8))  # FIX: Only keep masked weights
                 else:
                    norms = torch.norm(weights, p=2, dim=1, keepdim=True)
                    weights.div_(norms + 1e-8)

    def apply_homeostasis(self, weights, mask=None):
        """
        Debe llamarse periódicamente (ej: fin de epoch).
        1. Normaliza pesos (L2 norm).
        2. Resiembra neuronas muertas.
        """
        with torch.no_grad():
            # 1. Weight Normalization (Competencia sináptica)
            # Cada neurona post-sináptica tiene un límite de recursos
            if mask is not None:
                # W * Mask para asegurar ceros
                W_masked = weights * mask.float()
                norms = torch.norm(W_masked, p=2, dim=1, keepdim=True)
                weights.copy_(W_masked / (norms + 1e-8))  # FIX: Only keep masked weights
                
            # 2. Re-seeding (Stochastic)
            if self.steps_monitored > 0:
                rates = self.activity_monitor / self.steps_monitored
                dead_mask = rates < (self.target_rate * 0.1) # Muy baja actividad
                
                n_dead = dead_mask.sum().item()
                if n_dead > 0:
                    print(f"[HOMEOSTASIS] Resembrando {n_dead} neuronas muertas.")
                    # Reinicializar pesos de neuronas muertas
                    # Ruido + Normalización
                    new_w = torch.rand_like(weights[dead_mask]) * 0.1
                    if mask is not None:
                        new_w *= mask[dead_mask]
                        new_w = new_w / (torch.norm(new_w, dim=1, keepdim=True) + 1e-8)
                        
                    weights[dead_mask] = new_w
                
                # Reset monitors
                self.activity_monitor.zero_()
                self.steps_monitored = 0
