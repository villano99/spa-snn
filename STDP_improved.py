#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STDP Mejorado con:
1. Inhibición lateral (Winner-Take-All)
2. Normalización L2 estricta
3. Homeostasis (evita neuronas muertas)
4. Visualización de receptive fields
5. FIX: Soporte para resoluciones no cuadradas

VERSIÓN CORREGIDA PARA FALL DETECTION
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class STDP_W1_Improved:
    """STDP con mecanismos bio-inspirados para aprender features espaciales."""
    
    def __init__(self, 
                 a_plus=1.0, 
                 a_minus=0.5,
                 tau_pre=10.0,
                 tau_post=10.0, 
                 eta=2e-3,
                 normalize=True,
                 lateral_inhib=True,
                 homeostasis=True,
                 target_rate=0.08,
                 chunk_size=128,
                 weight_decay=1e-5):
        
        self.pre_decay = math.exp(-1 / tau_pre)
        self.post_decay = math.exp(-1 / tau_post)
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.eta = eta
        self.normalize = normalize
        self.lateral_inhib = lateral_inhib
        self.homeostasis = homeostasis
        self.target_rate = target_rate
        self.chunk_size = chunk_size
        self.weight_decay = weight_decay
        self.last_dw_mean = 0.0
        
        # Para homeostasis
        self.spike_counts = None
        self.total_steps = 0
    
    @torch.no_grad()
    def step_sparse(self, x_pre, spk_post, pre_trace, post_trace, W, mask):
        """Un paso de STDP con mejoras bio-inspiradas.
        
        CORRECCIÓN CLAVE: La actualización STDP está CONDICIONADA a que haya
        actividad real en el batch actual. Sin actividad = sin aprendizaje.
        """
        B, N = x_pre.shape
        M1 = spk_post.shape[1]
        
        # Inicializar contador de spikes
        if self.spike_counts is None:
            self.spike_counts = torch.zeros(M1, device=spk_post.device)
        
        # Actualizar contadores (para homeostasis)
        if self.homeostasis:
            self.spike_counts += spk_post.sum(dim=0)
            self.total_steps += B
        
        # Actualizar trazas con actividad del batch actual
        pre_trace.mul_(self.pre_decay).add_(x_pre)
        post_trace.mul_(self.post_decay).add_(spk_post)
        
        # Si no hay actividad post-sináptica en absoluto, aplicamos olvido y salimos
        if spk_post.sum() == 0:
            if self.weight_decay > 0:
                W.mul_(1.0 - self.weight_decay)
            self.last_dw_mean = 0.0
            return
        
        # LTP: Un post-spike sigue a un pre-trace (pre disparó antes → reforzar)
        # (M1, B) @ (B, N_in) -> (M1, N_in)
        # La multiplicación matricial asegura que SOLO se refuercen los pares donde
        # ESA neurona en ESE batch disparó (spk_post > 0)
        ltp = torch.matmul(spk_post.t(), pre_trace)
        
        # LTD: Un pre-spike llega DESPUÉS del post-spike (inversión causal → debilitar)
        # (M1, B) @ (B, N_in) -> (M1, N_in)
        ltd = torch.matmul(post_trace.t(), x_pre)
        
        # Delta W aplicado sólo dentro de la máscara local
        dw = (self.a_plus * ltp - self.a_minus * ltd) * mask
        
        # --- k-Winners-Take-All: Solo top 10% de neuronas activas aprende a tasa completa ---
        if self.lateral_inhib:
            dw_norm = dw.norm(dim=1)     # (M1,)
            if dw_norm.sum() > 0:
                n_active_neurons = (spk_post.sum(dim=0) > 0).float().sum().item()
                k = max(1, int(n_active_neurons * 0.10))
                k = min(k, M1)
                threshold_val = torch.topk(dw_norm, k).values[-1]
                scale = torch.where(dw_norm < threshold_val,
                                    torch.full_like(dw_norm, 0.05),
                                    torch.ones_like(dw_norm)).unsqueeze(1)
                dw = dw * scale
        
        # Track dW magnitude for diagnostics
        self.last_dw_mean = dw.abs().mean().item()
        
        # Actualización de pesos (PESOS POSITIVOS: Dale's Law)
        W.add_(dw, alpha=self.eta)
        
        # Olvido global (Bio-decay): limpia marcas accidentales en zonas sin refuerzo
        if self.weight_decay > 0:
            W.mul_(1.0 - self.weight_decay)
        
        # Clamping estricto: Neuronas excitatorias se mantienen excitatorias
        W.clamp_(0.0, 4.0)
        
        # Normalización L2 Vectorizada (Cada 50 pasos)
        if self.normalize and (self.total_steps % 50 == 0):
            norms = torch.norm(W, p=2, dim=1, keepdim=True).clamp_min_(1e-8)
            W.div_(norms)
            
    # def apply_homeostasis(self, W, mask):
    #     """
    #     Homeostasis: ajustar pesos según tasa de disparo.
        
    #     CORRECCIÓN DEFINITIVA: 
    #     Re-siembra neuronas muertas (norma=0) con ruido aleatorio
    #     antes de aplicar el boost.
    #     """
    #     if not self.homeostasis or self.total_steps == 0:
    #         return
        
    #     M1 = W.shape[0]
    #     rates = self.spike_counts / max(1, self.total_steps)
    #     boost = torch.sqrt(self.target_rate / (rates + 1e-6))
    #     boost = torch.clamp(boost, 0.1, 5.0) 
        
    #     with torch.no_grad():
    #         for i in range(M1):
    #             cols = mask[i].nonzero(as_tuple=False).squeeze(1)
    #             if cols.numel() == 0:
    #                 continue
                
    #             # --- INICIO DE LÓGICA MODIFICADA ---
                
    #             # 1. Calcular la norma actual
    #             # norm = torch.norm(W[i, cols], p=2)
                
    #             # 2. Si la neurona está muerta (norma < 1e-6),
    #             #    reinicializarla con ruido aleatorio.
    #             # if norm < 1e-6:
    #                 # (Asegúrate de que tus pesos sean positivos)
    #                 # W[i, cols] = torch.rand_like(W[i, cols]) * 0.1 
    #                 # norm = torch.norm(W[i, cols], p=2)
                
    #             # 3. Normalizar a 1.0 (dividiendo por la norma)
    #             # W[i, cols].div_(norm + 1e-8)
                
    #             # 4. Escalar (multiplicar) por el boost
    #             W[i, cols].mul_(boost[i])
                
    #             # 5. Asegurarnos de que sigan positivos
    #             W[i, cols].clamp_min_(1e-6)
                
    #             # --- FIN DE LÓGICA MODIFICADA ---
        
    #     print(f"[HOMEOSTASIS] Rate range: [{rates.min():.4f}, {rates.max():.4f}]")
    #     print(f"[HOMEOSTASIS] Boost range: [{boost.min():.2f}, {boost.max():.2f}]")
        
    #     # Resetear contadores (esto está bien)
    #     self.spike_counts.zero_()
    #     self.total_steps = 0
    def apply_homeostasis(self, W, mask):
        """
        Homeostasis mejorada (Bellec-style):
        1. Vectorized boost: √(target_rate / actual_rate) para TODAS las neuronas
        2. Resurrect dead neurons con reinicio Gabor-compatible
        3. L2 renorm después del boost para mantener estabilidad
        """
        if not self.homeostasis or self.total_steps == 0:
            return
            
        M1 = W.shape[0]
        rates = self.spike_counts / max(1, self.total_steps)
        
        with torch.no_grad():
            # === 1. Vectorized boost for ALL neurons ===
            # Boost = √(target / actual) — gentle multiplicative adjustment
            # Fast neurons (rate >> target) get dampened
            # Slow neurons (rate << target) get boosted
            boost = torch.sqrt(self.target_rate / (rates + 1e-6))
            boost = torch.clamp(boost, 0.5, 2.0)  # Conservative range
            
            # Apply boost to all neurons at once (vectorized, no loop)
            W.mul_(boost.unsqueeze(1))
            
            # === 2. Resurrect truly dead neurons ===
            dead_mask = rates < 1e-5
            dead_indices = dead_mask.nonzero(as_tuple=False).squeeze(1)
            
            if dead_indices.numel() > 0:
                n_resurrect = max(1, int(dead_indices.numel() * 0.3))
                indices_to_fix = dead_indices[torch.randperm(dead_indices.numel())[:n_resurrect]]
                
                for i in indices_to_fix:
                    cols = mask[i].nonzero(as_tuple=False).squeeze(1)
                    if cols.numel() == 0: continue
                    
                    norm = torch.norm(W[i, cols], p=2)
                    if norm < 0.05:
                        # Dead neuron: reinitialize with small random weights
                        W[i, cols] = torch.rand_like(W[i, cols]) * 0.01 + 0.01

            # === 3. L2 renorm after boost to prevent weight explosion ===
            norms = torch.norm(W, p=2, dim=1, keepdim=True).clamp_min_(1e-8)
            W.div_(norms)
            
            # === 4. Clamp to valid range ===
            W.clamp_(0.0, 4.0)

        n_boosted = (boost > 1.1).sum().item()
        n_dampened = (boost < 0.9).sum().item()
        print(f"[HOMEOSTASIS] Rate: [{rates.min():.4f}, {rates.max():.4f}] | "
              f"Boosted: {n_boosted} | Dampened: {n_dampened} | "
              f"Dead resurrected: {dead_indices.numel() if dead_indices.numel() > 0 else 0}")
        
        self.spike_counts.zero_()
        self.total_steps = 0


def visualize_receptive_fields(W, mask, W_model, H_model, P, save_dir="./rf_viz", n_neurons=16, auto_crop=False, event_density=None):
    """
    Visualiza neuronas con Muestreo Espacial Uniforme para evitar capturar solo ruido de bordes.
    auto_crop=False: Muestra el campo completo (H_model x W_model).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    W = W.detach().cpu()
    mask = mask.detach().cpu()
    M1 = W.shape[0]

    norms = torch.norm(W, p=2, dim=1)
    grid_size = int(np.ceil(np.sqrt(n_neurons)))
    
    # spatial sampling
    # Dividir la imagen en grid_size x grid_size zonas
    cell_w = W_model / grid_size
    cell_h = H_model / grid_size
    
    indices = []
    
    # Calcular promedios Y, X para cada neurona usando la mascara
    # mask_indices: (M1, N_in) bool
    for gy in range(grid_size):
        for gx in range(grid_size):
            if len(indices) >= n_neurons:
                break
            
            # Limites de esta celda
            x_min, x_max = gx * cell_w, (gx + 1) * cell_w
            y_min, y_max = gy * cell_h, (gy + 1) * cell_h
            
            best_idx = -1
            best_norm = -1.0
            
            # Buscar la neurona más fuerte cuyo centro caiga en esta celda
            # Para no hacer un loop lento, hacemos una aproximacion rapida
            # o simplemente iteramos (W.shape[0] es pequeño, =9600)
            # Pero para ser rapidos: Mapeo de neurona a su centro (gx, gy)
            # Asumimos que la lista original de mascaras ordena las celulas más o menos
            # secuencialmente. Pero mejor:
            pass
            
    # Forma vectorizada de encontrar el centro de las neuronas
    cols_per_neuron = [mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(M1)]
    centers_x = torch.zeros(M1)
    centers_y = torch.zeros(M1)
    
    for i, cols in enumerate(cols_per_neuron):
        if cols.numel() > 0:
            pixel_indices = cols // P
            xs = (pixel_indices % W_model).float()
            ys = (pixel_indices // W_model).float()
            centers_x[i] = xs.mean()
            centers_y[i] = ys.mean()
        else:
            centers_x[i] = -1
            centers_y[i] = -1

    # Elegimos la más fuerte por cuadrante
    for gy in range(grid_size):
        for gx in range(grid_size):
            if len(indices) >= n_neurons:
                break
            x_min, x_max = gx * cell_w, (gx + 1) * cell_w
            y_min, y_max = gy * cell_h, (gy + 1) * cell_h
            
            in_cell = (centers_x >= x_min) & (centers_x < x_max) & (centers_y >= y_min) & (centers_y < y_max)
            valid_norms = norms.clone()
            valid_norms[~in_cell] = -1  # Invalidamos las de afuera
            
            best_idx = torch.argmax(valid_norms).item()
            if valid_norms[best_idx] > 0:
                indices.append(best_idx)

    # Llenar huecos si no hay suficientes neuronas
    if len(indices) < n_neurons:
        rem_vals, rem_indices = torch.topk(norms, n_neurons)
        for idx in rem_indices.numpy():
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= n_neurons:
                break

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14), facecolor='black')
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if i >= len(axes): break
        ax = axes[i]
        ax.set_facecolor('black')
        
        # Get active inputs for this neuron
        cols = mask[idx].nonzero(as_tuple=False).squeeze(1)
        if cols.numel() == 0:
            ax.set_title(f"N{idx} (Dead)", fontsize=7, color='red')
            ax.axis('off')
            continue

        # Compute spatial coords
        ps = cols % P
        pixel_indices = cols // P
        xs = pixel_indices % W_model
        ys = pixel_indices // W_model
        
        # Build signed weights on FULL frame (no crop - shows spatial location)
        patch_pos = np.zeros((H_model, W_model))
        patch_neg = np.zeros((H_model, W_model))
        cx_pos = xs.float().mean().item()
        cy_pos = ys.float().mean().item()
        
        weights = W[idx, cols]
        for w_idx in range(len(cols)):
            lx = int(xs[w_idx].item())
            ly = int(ys[w_idx].item())
            lp = int(cols[w_idx].item() % P) # Polarity: 0=ON, 1=OFF
            w_val = weights[w_idx].item()
            
            if 0 <= lx < W_model and 0 <= ly < H_model:
                if lp == 0: # ON
                    patch_pos[ly, lx] += w_val
                else: # OFF
                    patch_neg[ly, lx] += w_val # Map OFF weights to negative for visualization

        # Compose RGB on full canvas using abs-max normalization
        # Green = weights connected to ON inputs, Red = connected to OFF inputs
        patch_vis = np.zeros((H_model, W_model, 3))
        
        combined = patch_pos - patch_neg  # signed map for viz
        abs_max = max(np.abs(combined).max(), 1e-8)
        combined_norm = combined / abs_max  # [-1, 1]
        
        # Map to RGB: positive (ON-dominated) → green, negative (OFF-dominated) → red
        patch_vis[..., 1] = np.clip(combined_norm, 0, 1)   # green channel
        patch_vis[..., 0] = np.clip(-combined_norm, 0, 1)  # red channel
        
        # Gamma para visibilidad (aclarar senales debiles)
        patch_vis = np.power(patch_vis.clip(0, 1), 0.45)
        
        ax.imshow(patch_vis, interpolation='nearest', aspect='equal')
        ax.set_title(f"N{idx} ({cx_pos:.0f},{cy_pos:.0f})", fontsize=7, color='white')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
    
    # Hide unused
    for j in range(len(indices), len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Campos Receptivos STDP (Verde=ON / Rojo=OFF)', fontsize=13, color='white')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rf_full_sampled.png", dpi=150, facecolor='black')
    plt.close()
    
    print(f"[VIZ] Receptive fields guardados en {save_dir}/rf_full_sampled.png")

    # ----- Heatmap: Densidad de Eventos de Entrada (Ground Truth) -----
    try:
        from scipy.ndimage import gaussian_filter
        if event_density is not None:
            density_cpu = event_density.cpu().numpy()  # (H, W)
            density_smooth = gaussian_filter(density_cpu, sigma=2.0)
            
            vmax = np.percentile(density_smooth[density_smooth > 0], 98) if density_smooth.max() > 0 else 1.0
            
            plt.figure(figsize=(10, 8))
            plt.imshow(density_smooth, cmap='inferno', vmax=vmax)
            plt.colorbar(label='Densidad de Eventos (activaciones de entrada)')
            plt.title('Mapa de Calor STDP - Densidad de Eventos de Entrada')
            plt.xlabel('Cámara X')
            plt.ylabel('Cámara Y')
            plt.savefig(f"{save_dir}/rf_heatmap.png", dpi=150)
            plt.close()
            print(f"[VIZ] Heatmap guardado en {save_dir}/rf_heatmap.png")
        else:
            print("[VIZ] No hay datos de densidad de eventos disponibles")
    except Exception as e:
        print(f"[VIZ] No se pudo generar el heatmap: {e}")

def stdp_pretrain_improved(model, loader, device, args, W, H):
    """
    Versión mejorada de stdp_pretrain() con soporte Jerárquico (L1 + L2).
    """
    model.to(device)
    model.train()
    
    # Separate STDP managers for L1 and L2
    stdp1 = STDP_W1_Improved(
        a_plus=args.a_plus, a_minus=args.a_minus,
        tau_pre=args.tau_pre, tau_post=args.tau_post,
        eta=args.eta_stdp, normalize=args.stdp_normalize,
        lateral_inhib=True, homeostasis=True,
        target_rate=args.stdp_target_rate, chunk_size=128,
        weight_decay=args.stdp_weight_decay
    )
    
    # L2 uses slightly slower traces to capture temporal motifs (sequences)
    stdp2 = STDP_W1_Improved(
        a_plus=args.a_plus * 0.8, a_minus=args.a_minus * 0.8,
        tau_pre=args.tau_pre * 1.5, tau_post=args.tau_post * 1.5,
        eta=args.eta_stdp, normalize=args.stdp_normalize,
        lateral_inhib=True, homeostasis=True,
        target_rate=args.stdp_target_rate * 0.5, chunk_size=128,
        weight_decay=args.stdp_weight_decay
    )
    
    print("\n[STDP] Iniciando pre-entrenamiento JERÁRQUICO (L1 + L2)")
    print(f"[STDP] L1: eta={args.eta_stdp}, target={args.stdp_target_rate}")
    print(f"[STDP] L2: eta={args.eta_stdp}, target={args.stdp_target_rate*0.5} (Traces Lentas)")
    print(f"[STDP] Resolución: {W}×{H}")

    # Preservar parámetros originales
    orig_t1 = model.lif1.threshold.item()
    orig_t2 = model.lif2.threshold.item()
    orig_gain = model.input_gain
    
    # Thresholds de pre-entrenamiento SELECTIVOS
    # Con W>=0, necesitamos un umbral más alto para mantener la sparseness
    model.lif1.threshold = torch.tensor(1.0, device=device) 
    model.lif2.threshold = torch.tensor(1.0, device=device)
    stdp_gain = 1.0  
    
    print(f"[STDP] Config ESTABLE: gain={stdp_gain}, thresh1=1.0, thresh2=1.0")
    
    event_density_map = torch.zeros(H, W, device=device)  
    
    # --- PHASE A: Stabilize L1 (Edge Detectors) ---
    L1_epochs = max(1, args.epochs_stdp // 2)
    print(f"\n[STDP] >>> BLOQUE 1: Estabilizando Capa 1 (W1) por {L1_epochs} épocas")
    for ep in range(L1_epochs):
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"[STDP-L1] Epoch {ep+1}/{L1_epochs}")
        for batch in pbar:
            if len(batch) == 3: x, _, _ = batch
            else: x, _ = batch
            x = x.to(device)
            B, T, N = x.shape
            pre_trace1 = torch.zeros(B, model.N_in, device=device)
            post_trace1 = torch.zeros(B, model.M1, device=device)
            mem1 = torch.zeros(B, model.M1, device=device)
            for t in range(T):
                xt = x[:, t]
                h1 = model.masked_mm(model.W1, model.mask1, xt * stdp_gain)
                spk1, mem1 = model.lif1(h1, mem1)
                stdp1.step_sparse(xt, spk1, pre_trace1, post_trace1, model.W1, model.mask1)
                # Density mapping (solo en el primer batch de la primera epoca para eficiencia)
                evt_flat = xt.detach().abs().sum(dim=0)
                n_pixels = H * W
                n_pol = evt_flat.numel() // n_pixels
                if n_pol > 0:
                    evt_by_pixel = evt_flat[:n_pixels * n_pol].view(n_pixels, n_pol).sum(dim=1)
                    event_density_map.add_(evt_by_pixel.view(H, W))
            pbar.set_postfix({"s1": f"{(spk1>0).float().mean():.3f}", "dw1": f"{getattr(stdp1,'last_dw_mean',0):.4f}"})
        stdp1.apply_homeostasis(model.W1, model.mask1)
        visualize_receptive_fields(model.W1, model.mask1, W, H, 2, save_dir=f"./rf_viz/L1_stage1_ep{ep+1}", n_neurons=16, event_density=event_density_map)

    # --- PHASE B: Stabilize L2 (Sequences/Motifs) ---
    L2_epochs = max(1, args.epochs_stdp - L1_epochs)
    print(f"\n[STDP] >>> BLOQUE 2: Estabilizando Capa 2 (W2) por {L2_epochs} épocas (W1 CONGELADA)")
    for ep in range(L2_epochs):
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"[STDP-L2] Epoch {ep+1}/{L2_epochs}")
        for batch in pbar:
            if len(batch) == 3: x, _, _ = batch
            else: x, _ = batch
            x = x.to(device)
            B, T, N = x.shape
            pre_trace2 = torch.zeros(B, model.M1, device=device)
            post_trace2 = torch.zeros(B, model.M2, device=device)
            mem1 = torch.zeros(B, model.M1, device=device)
            mem2 = torch.zeros(B, model.M2, device=device)
            for t in range(T):
                xt = x[:, t]
                h1 = model.masked_mm(model.W1, model.mask1, xt * stdp_gain)
                with torch.no_grad(): spk1, mem1 = model.lif1(h1, mem1)
                h2 = model.masked_mm(model.W2, model.mask2, spk1)
                spk2, mem2 = model.lif2(h2, mem2)
                stdp2.step_sparse(spk1, spk2, pre_trace2, post_trace2, model.W2, model.mask2)
            pbar.set_postfix({"s2": f"{(spk2>0).float().mean():.3f}", "dw2": f"{getattr(stdp2,'last_dw_mean',0):.4f}"})
        stdp2.apply_homeostasis(model.W2, model.mask2)

    # Resturación Final
    model.lif1.threshold = torch.tensor(orig_t1, device=device)
    model.lif2.threshold = torch.tensor(orig_t2, device=device)
    model.input_gain = orig_gain
    print(f"\n[STDP] Parámetros restaurados.")
    
    # Guardar
    torch.save({'W1': model.W1.detach().cpu(), 'W2': model.W2.detach().cpu()}, args.save_w1)
    print(f"[STDP] Pesos W1 y W2 guardados en {args.save_w1}")
    
    # Visualización final de L1
    print(f"[STDP] Generando visualización final...")
    visualize_receptive_fields(
        model.W1, model.mask1, W, H, 2,
        save_dir="./rf_viz/final",
        n_neurons=25,
        event_density=event_density_map
    )