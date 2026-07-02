import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import math

class StreamingFallDataset(Dataset):
    """
    Dataset tipo 'Streaming' que corta las grabaciones largas en ventanas temporales fijas.
    
    A diferencia del dataset original (que comprime todo el video en N bins),
    este dataset genera múltiples muestras por cada grabación usando una ventana deslizante.
    Mantiene la escala temporal real (dt fijo).
    
    Args:
        hdf5_path (str): Ruta al datset HDF5.
        window_size_ms (float): Tamaño de la ventana de tiempo en ms (ej: 1000ms).
        dt_ms (float): Tamaño de cada paso de tiempo en ms (ej: 50ms).
        stride_ms (float): Desplazamiento de la ventana para generar la siguiente muestra (ej: 250ms).
        fall_onset_ms (float): Tiempo estimado donde comienza la caída (para etiquetar).
                            Antes de esto se considera WALK.
        target_density (float): Para submuestreo aleatorio (ahorro de memoria/cómputo).
    """
    
    def __init__(self, hdf5_path, split='train', 
                 window_size_ms=1000.0, dt_ms=50.0, stride_ms=500.0,
                 fall_onset_ms=1200.0, # Según tu observación: caída empieza ~1.2s
                 downsample=4,  # Factor de reducción espacial
                 target_density=0.03,
                 return_metadata=False):
        
        self.hdf5_path = hdf5_path
        self.window_size_us = int(window_size_ms * 1000)
        self.dt_us = int(dt_ms * 1000)
        self.stride_us = int(stride_ms * 1000)
        self.fall_onset_us = int(fall_onset_ms * 1000)
        
        self.T_bins = int(self.window_size_us / self.dt_us)
        self.downsample = downsample
        self.target_density = target_density
        self.return_metadata = return_metadata
        self.split = split
        
        # Abrir HDF5 temporalmente para metadata
        try:
            self.f = h5py.File(hdf5_path, 'r')
        except Exception as e:
            raise FileNotFoundError(f"No se pudo abrir {hdf5_path}: {e}")
            
        if 'samples' not in self.f:
            raise ValueError("El HDF5 no tiene el grupo 'samples'")

        # Dimensiones originales
        if 'metadata/resolution' in self.f:
            res = self.f['metadata/resolution'][:]
            self.W_orig, self.H_orig = res[0], res[1]
        else:
            self.W_orig, self.H_orig = 640, 480 # Default
            
        self.W = self.W_orig // downsample
        self.H = self.H_orig // downsample
        self.P = 2
        self.N_in = self.W * self.H * self.P
        
        print(f"[STREAMING DATA] Cargando {split}...")
        print(f"  Window: {window_size_ms}ms | dt: {dt_ms}ms | Stride: {stride_ms}ms")
        print(f"  Bins por ventana: {self.T_bins}")
        print(f"  Fall Onset (Label threshold): {fall_onset_ms}ms")
        
        # --- PREPARAR ÍNDICES (VENTANAS) ---
        self.window_indices = [] # [(sample_idx, t_start_us, label_window), ...]
        
        self._prepare_windows()
        
        # [MULTIPROCESSING FIX] Cerrar el handle principal
        if self.f:
            self.f.close()
            self.f = None
        
    def _prepare_windows(self):
        """Genera ventanas centradas en la acción (la acción está en el medio de la grabación)."""
        n_samples = len(self.f['samples'])
        
        # Determinar split: prioridad split attr > persona > idx%5
        test_persons = {'ziyang', 'zhangliming'}  # Fall Detection dataset
        
        counts_per_class = {}
        skipped_short = 0
        
        for idx in range(n_samples):
            key = f"samples/{idx:04d}"
            if key not in self.f: continue
            
            grp = self.f[key]
            global_label = int(grp['label'][()])
            duration_us = int(grp['duration_us'][()])
            
            # --- Flexible split logic ---
            # 1. Atributo 'split' explícito (DVSGesture, etc.)
            sample_split = grp.attrs.get('split', '')
            if sample_split:
                if self.split != sample_split:
                    continue
            else:
                # 2. Split por persona (Fall Detection)
                person = grp.attrs.get('person', '')
                if person:
                    if self.split == 'train' and person in test_persons: continue
                    if self.split == 'test' and person not in test_persons: continue
                else:
                    # 3. Fallback a idx%5
                    if self.split == 'train' and (idx % 5 == 0): continue
                    if self.split == 'test' and (idx % 5 != 0): continue
            
            if duration_us < self.window_size_us:
                skipped_short += 1
                continue
            
            # --- Estrategia: Smart Crop + Ventanas centradas en pico de actividad ---
            # Buscar el momento de máxima actividad (pico de eventos)
            events = grp['events'][:]
            t_events = events[:, 0]
            
            # Dividir en bins de 500ms y encontrar el pico
            n_bins_search = max(1, int(duration_us / 500000))
            bin_edges = np.linspace(0, duration_us, n_bins_search + 1)
            counts_hist = np.zeros(n_bins_search)
            for b_idx in range(n_bins_search):
                counts_hist[b_idx] = np.sum(
                    (t_events >= bin_edges[b_idx]) & (t_events < bin_edges[b_idx + 1])
                )
            
            # Centro del bin con más eventos
            peak_bin = np.argmax(counts_hist)
            peak_center_us = int((bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2)
            
            # Generar ventanas centradas en el pico con jitter
            center_start = int(peak_center_us - self.window_size_us / 2)
            offsets_us = [-200000, 0, 200000]  # -200ms, centro, +200ms
            
            for off in offsets_us:
                t_s = center_start + off
                if t_s >= 0 and (t_s + self.window_size_us) <= duration_us:
                    self.window_indices.append((idx, t_s, global_label))
                    counts_per_class[global_label] = counts_per_class.get(global_label, 0) + 1
        
        print(f"  [DONE] Ventanas generadas para split='{self.split}'")
        for label, count in sorted(counts_per_class.items()):
            print(f"  Clase {label}: {count} ventanas")
        if skipped_short > 0:
            print(f"  (Ignoradas {skipped_short} grabaciones por ser < {self.window_size_us/1000:.1f}s)")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        """
        Retorna (events_tensor, label, dims)
        events_tensor: [TimeBins, N_in] (Flattened spatial)
        """
        sample_idx, t_start, label = self.window_indices[idx]
        t_end = t_start + self.window_size_us
        
        # [MULTIPROCESSING FIX] Abrir archivo localmente
        # Optimización: si os.exists check no es necesario si asumimos path correcto
        with h5py.File(self.hdf5_path, 'r') as f:
            grp = f[f"samples/{sample_idx:04d}"]
            events = grp['events'][:] # [N, 4]
            # No leer attrs aquí, es lento y propenso a errores. Usar self.W_orig si fuera necesario.

        # 1. Filtrar eventos dentro de la ventana
        # Optimización: numpy mask
        # [N, 4] -> 0:t, 1:x, 2:y, 3:p
        t_col = events[:, 0]
        mask = (t_col >= t_start) & (t_col < t_end)
        events_window = events[mask]
        
        if len(events_window) == 0:
             # Retornar vacío si no hay eventos
             empty = torch.zeros(self.T_bins, self.N_in)
             label_t = torch.tensor(label, dtype=torch.long)
             if self.return_metadata:
                 return empty, label_t, (self.W, self.H, self.P)
             return empty, label_t
        
        # 2. Shift temporal
        t_rel = events_window[:, 0] - t_start
        x = events_window[:, 1]
        y = events_window[:, 2]
        p = events_window[:, 3]
        
        # 3. Downsample espacial
        x = x // self.downsample
        y = y // self.downsample
        
        # 4. Binning
        bin_indices = (t_rel / self.dt_us).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, self.T_bins - 1)
        
        # 5. Spatial Index validation
        valid = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        
        bin_indices = bin_indices[valid]
        x = x[valid]
        y = y[valid]
        p = p[valid]
        
        # Unique Keys logic
        spatial_idx = (y * self.W + x) * self.P + p
        flat_keys = bin_indices * self.N_in + spatial_idx.astype(np.int64)
        unique_keys = np.unique(flat_keys)
        
        # Rate Limiting
        max_spikes = int(self.T_bins * self.N_in * self.target_density)
        if len(unique_keys) > max_spikes:
             unique_keys = np.random.choice(unique_keys, max_spikes, replace=False)
             
        # Reconstruir
        final_bins = unique_keys // self.N_in
        final_neurons = unique_keys % self.N_in
        
        spikes = torch.zeros(self.T_bins, self.N_in, dtype=torch.float32)
        spikes[final_bins, final_neurons] = 1.0
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.return_metadata:
            return spikes, label_tensor, (self.W, self.H, self.P)
        
        return spikes, label_tensor

if __name__ == "__main__":
    pass
