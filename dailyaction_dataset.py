import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
try:
    from saccade_augmenter import SaccadeAugmenter
except:
    pass

class DailyDVSHDF5Dataset(Dataset):
    """
    Dataset optimizado que lee desde el HDF5 pre-procesado (2s, ds2, denoise).
    Soporta downsampling adicional al vuelo (e.g. archivo ds2 -> target ds4).
    Soporta multi-processing (abre el archivo en __getitem__).
    """
    def __init__(self, h5_path, split='train', time_bins=80, target_downsample=None, return_metadata=False):
        self.h5_path = h5_path
        self.split = split
        self.return_metadata = return_metadata
        self.time_bins = time_bins
        
        # Leemos metadatos una vez para setup
        with h5py.File(h5_path, 'r') as f:
            self.classes = f.attrs.get('classes', [f"Class_{i}" for i in range(f.attrs.get('num_classes', 12))])
            
            if 'resolution' in f.attrs:
                self.file_res = f.attrs['resolution']
            elif 'sensor_size' in f.attrs:
                self.file_res = f.attrs['sensor_size'][:2] # [W, H]
            else:
                self.file_res = [128, 128] # Default for pre-processed DailyAction
                
            self.file_downsample = f.attrs.get('downsample', 1) 
            self.window_ms = f.attrs.get('window_ms', 2000.0) # Default for DailyAction
            self.sample_keys = sorted(list(f[split].keys()))

        # Calcular factor extra de downsample
        self.extra_ds = 1
        if target_downsample is not None and target_downsample > self.file_downsample:
            self.extra_ds = target_downsample // self.file_downsample
            
        self.W = self.file_res[0] // self.extra_ds
        self.H = self.file_res[1] // self.extra_ds
        self.P = 2
        self.N_in = self.W * self.H * self.P
        
        print(f"[HDF5] Cargando {split}: {len(self.sample_keys)} muestras")
        print(f"       File ds={self.file_downsample} -> Target ds={target_downsample} (Extra: {self.extra_ds})")
        
        self.augmenter = None
        if self.split == 'train':
            try:
                self.augmenter = SaccadeAugmenter(W_target=self.file_res[0], H_target=self.file_res[1], max_shift_x=15, max_shift_y=15, flip_prob=0.5, current_ds=1)
                print("       [*] Saccade (Spatial M.-Augmentation) Activada")
            except NameError:
                pass
                
    def __len__(self):
        return len(self.sample_keys)
        
    def _open_file(self):
        # Helper para abrir archivo en cada worker
        return h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        # Abrir archivo on-demand para soporte multiproceso
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[self.split][self.sample_keys[idx]]
            
            if isinstance(grp, h5py.Dataset):
                # Caso Dataset Estructurado (Compuesto)
                data = grp[:]
                t = data['t']
                x = data['x']
                y = data['y']
                p = data['p']
            else:
                # Caso Grupo con datasets individuales
                t = grp['t'][:]
                events_grp = grp.get('events')
                if events_grp is not None:
                    events = events_grp[:] 
                    x = events[:, 1]
                    y = events[:, 2]
                    p = events[:, 3]
                else:
                    x = grp['x'][:]
                    y = grp['y'][:]
                    p = grp['p'][:]
            
            if 'label_idx' in grp.attrs:
                label_idx = grp.attrs['label_idx']
            else:
                label_idx = grp.attrs.get('label', 0)
        
        # Aplicamos la aumentación sacádica (Jitter Espacial /espejo) si estamos en TRain
        if getattr(self, 'augmenter', None) is not None:
            x, y, t, p = self.augmenter(x, y, t, p)
        
        # Apply extra downsample
        if self.extra_ds > 1:
            x = x // self.extra_ds
            y = y // self.extra_ds
            
        # Clamp coords to effective resolution
        x = np.clip(x, 0, self.W - 1)
        y = np.clip(y, 0, self.H - 1)
        
        # Convert to spikes
        spikes = np.zeros((self.time_bins, self.N_in), dtype=np.float32)
        
        if len(t) > 0:
            # Normalizar t a 0..window_ms
            # El t guardado ya empieza en 0
            # Asignar a bins
            # window_ms es float, t es int (us)
            window_us = self.window_ms * 1000
            
            # Clip t just in case
            t = np.clip(t, 0, window_us - 1)
            
            time_bins = (t * self.time_bins // window_us).astype(np.int64)
            time_bins = np.clip(time_bins, 0, self.time_bins - 1)
            
            flat_idx = (y * self.W + x) * self.P + p
            flat_idx = np.clip(flat_idx, 0, self.N_in - 1)
            
            spikes[time_bins, flat_idx] = 1.0
            
        spikes_t = torch.tensor(spikes, dtype=torch.float32)
        label_t = torch.tensor(label_idx, dtype=torch.long)
        
        if self.return_metadata:
            return spikes_t, label_t, (self.W, self.H, self.P)
        return spikes_t, label_t


