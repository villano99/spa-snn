#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train_Fall_SNN.py
=================
Entrenamiento de SNN para detección de caídas con eventos DVS.
Versión: Streaming + R-STDP

Arquitectura bio-plausible:
- R-STDP para pre-entrenamiento de W1 (features discriminativos)
- GraphSNN con máscaras de conectividad local
- LIF neurons con surrogate gradients
- Streaming Dataset (Time-windows reales)
"""

import argparse
import os
import h5py
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Importar nuevos módulos
from streaming_dataset import StreamingFallDataset
from tonic_dataset import TonicSpikeDataset
from dailydvs_dataset import DailyDVS200Dataset
from R_STDP import R_STDP
from STDP_improved import STDP_W1_Improved, stdp_pretrain_improved, visualize_receptive_fields
from EProp_Classifier import EProp_Classifier

# ========================
# CLI Arguments
# ========================
def get_args():
    p = argparse.ArgumentParser(description='Train SNN for Fall Detection (Streaming + R-STDP)')
    
    # Dataset
    p.add_argument('--dataset', type=str, default='fall_detection',
                   choices=['fall_detection', 'dvs_gesture', 'nmnist', 'dailydvs', 'paf'],
                   help='Dataset a usar')
    p.add_argument('--hdf5-path', type=str, required=True,
                   help='Path al archivo HDF5 del dataset')
    p.add_argument('--downsample', type=int, default=4,
                   help='Factor de downsample espacial (4→160x120, 8→80x60)')
    p.add_argument('--time-bins', type=int, default=40,
                   help='Número de bins temporales por ventana (40x25ms = 1s)')
    p.add_argument('--dt-ms', type=float, default=25.0,
                   help='Duración de cada bin temporal en ms (25ms = Live Standard)')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Batch size (Safe: 64, Turbo: 256)')
    p.add_argument('--num-workers', type=int, default=2,
                   help='Workers para DataLoader (Safe: 2)')
    p.add_argument('--turbo', action='store_true',
                   help='Activa modo agresivo (Batch 256, Chunk 4096). CUIDADO: Puede inestabilizar el sistema.')

    
    # Arquitectura (se infieren del HDF5, pero se pueden override)
    p.add_argument('--l1-per-cell', type=int, default=2,
                   help='Neuronas L1 por celda de entrada')
    p.add_argument('--l1-stride', type=int, default=1,
                   help='Stride espacial para celulas L1 (2 = mitad de neuronas, 4x mas rapido)')
    p.add_argument('--l2-size', type=int, default=512,
                   help='Número de neuronas L2 (512 recomendado)')
    p.add_argument('--l3-size', type=int, default=256,
                   help='Número de neuronas L3 (256 recomendado)')
    
    # Radios de conectividad local
    # Radios de conectividad local (Reducidos para foco visual)
    p.add_argument('--r1', type=str, default='8.0',
                   help='Radio de conectividad local L1 (puede ser una lista separada por comas ej: "6.0,10.0,16.0")')
    p.add_argument('--r2', type=float, default=8.0,
                   help='Radio de conectividad L1->L2')
    p.add_argument('--r3', type=float, default=12.0,
                   help='Radio de conectividad L2->L3')
    
    # R-STDP pre-training
    p.add_argument('--epochs-stdp', type=int, default=0,
                   help='Épocas de pre-entrenamiento R-STDP (0=skip)')
    p.add_argument('--eta-stdp', type=float, default=1e-4,
                   help='Learning rate para STDP (Slower = more stable)')
    p.add_argument('--a-plus', type=float, default=1.0,
                   help='A+ para STDP (LTP)')
    p.add_argument('--a-minus', type=float, default=1.2,
                   help='A- para STDP (LTD) [a_minus > a_plus para incentivar competencia]')
    p.add_argument('--tau-pre', type=float, default=2.0,
                   help='Tau pre-sináptica (steps) [2.0 = 50ms @ 25ms dt]')
    p.add_argument('--tau-post', type=float, default=2.0,
                   help='Tau post-sináptica (steps) [2.0 = 50ms @ 25ms dt]')
    p.add_argument('--save-w1', type=str, default='w1_rstdp.pt',
                   help='Archivo para guardar W1 pre-entrenado')
    
    # Eval STDP-only
    p.add_argument('--eval-stdp-only', action='store_true',
                   help='Solo evaluar STDP con decoder lineal')
    p.add_argument('--freeze-w1', action='store_true',
                   help='Ablation: Congela la primera capa durante el entrenamiento supervisado')
    p.add_argument('--eval-only', action='store_true',
                   help='Skipea el entrenamiento y solo corre validacion usando --resume')
    p.add_argument('--decoder-epochs', type=int, default=10,
                   help='Épocas para entrenar decoder (eval-stdp-only)')
    p.add_argument('--decoder-lr', type=float, default=1e-3,
                   help='LR para decoder')
    
    # Supervised training
    p.add_argument('--dense-head', action='store_true',
                   help='[ARCH] Usar capa totalmente conectada (Dense) para L3 (Global Readout)')
    p.add_argument('--epochs-sup', type=int, default=50,
                   help='Épocas de entrenamiento supervisado')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Learning rate supervisado')
    p.add_argument('--pos-weight', type=float, default=2.5,
                   help='Peso para la clase positiva (Fall) en CrossEntropy')
    p.add_argument('--beta', type=float, default=0.82,
                   help='[ARCH] Factor de decaimiento LIF (0.82~125ms @ 25ms dt)')
    p.add_argument('--grad-clip', type=float, default=1.0,
                   help='Gradient clipping')
    p.add_argument('--resume', type=str, default='',
                   help='[RESUME] Path to checkpoint to resume from (e.g. ckpts_fall/sup_epoch_5.pt)')
    p.add_argument('--threshold1', type=float, default=0.8,
                   help='LIF threshold 1')
    p.add_argument('--threshold2', type=float, default=0.6,
                   help='Threshold para L2 (0.6 recomendado)')
    p.add_argument('--threshold3', type=float, default=0.5,
                   help='Threshold para L3 (0.5 recomendado)')
    p.add_argument('--input-gain', type=float, default=3.0,
                   help='Ganancia de entrada')
    p.add_argument('--train-w1', action='store_true',
                   help='Entrenar W1 en fase supervisada (default: frozen)')
    
    # Regularization
    p.add_argument('--dropout', type=float, default=0.0,
                   help='Dropout rate')
    p.add_argument('--weight-decay', type=float, default=1e-4,
                   help='Weight decay (L2 regularization)')
    p.add_argument('--stdp-target-rate', type=float, default=0.08,
                   help='Target spike rate para homeostasis STDP')
    p.add_argument('--stdp-normalize', action='store_true', default=True,
                   help='Normalizar pesos L2 durante STDP')
    p.add_argument('--stdp-weight-decay', type=float, default=1e-4,
                   help='Peso de olvido global (decay) para limpiar phantom features en STDP')
    p.add_argument('--stdp-batch-size', type=int, default=64,
                   help='Batch size exclusivo para STDP (Recomendado: 64 para estabilidad)')

    # Misc
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt-dir', type=str, default='./ckpts_fall')
    p.add_argument('--num-classes', type=int, default=4,
                   help='Número de clases (4 default, 5 dailydvs)')
    
    p.add_argument('--eprop-resume', type=str, default='',
                        help='Ruta a un checkpoint eprop_best.pt para continuar desde ese punto')
    p.add_argument('--save-sup-best', type=str, default='best_sup.pt',
                   help='Nombre del archivo para guardar el mejor modelo supervisado')
    
    return p.parse_args()


def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================
# Dataset loaders (STREAMING)
# ========================
def make_dataloaders(args, for_stdp=False):
    """Crea DataLoaders (HDF5 streaming o tonic directo)."""
    
    if args.dataset in ('dvs_gesture', 'nmnist'):
        return _make_dataloaders_tonic(args)
    elif args.dataset == 'dailydvs':
        return _make_dataloaders_dailydvs(args)
    elif args.dataset == 'paf':
        return _make_dataloaders_paf(args)
    else:
        return _make_dataloaders_hdf5(args)


def _make_dataloaders_dailydvs(args):
    """DataLoaders para DailyDVS-200 (Support raw .aedat4 or pre-processed .h5)."""
    from dailydvs_dataset import DailyDVS200Dataset, DailyDVSHDF5Dataset

    # Check if HDF5 path is provided and valid for DailyDVS
    use_hdf5 = False
    if args.hdf5_path and args.hdf5_path.endswith('.h5') and os.path.exists(args.hdf5_path):
        use_hdf5 = True
        print(f"[DATA] Cargando DailyDVS-200 desde HDF5 optimizado: {args.hdf5_path}")
        
        train_dataset = DailyDVSHDF5Dataset(
            args.hdf5_path, split='train', time_bins=args.time_bins, 
            target_downsample=args.downsample, return_metadata=True
        )
        test_dataset = DailyDVSHDF5Dataset(
            args.hdf5_path, split='test', time_bins=args.time_bins, 
            target_downsample=args.downsample, return_metadata=True
        )
    else:
        # Fallback to Raw
        window_size_ms = args.time_bins * args.dt_ms
        print(f"[DATA] Cargando DailyDVS-200 desde RAW (Lento): {args.data_dir}")
        
        train_dataset = DailyDVS200Dataset(
            args.data_dir, split='train',
            window_size_ms=window_size_ms,
            dt_ms=args.dt_ms,
            downsample=args.downsample,
            return_metadata=True
        )
        
        test_dataset = DailyDVS200Dataset(
            args.data_dir, split='test',
            window_size_ms=window_size_ms,
            dt_ms=args.dt_ms,
            downsample=args.downsample,
            return_metadata=True
        )
    
    # Auto-set num_classes from dataset
    # HDF5 dataset has 'classes' attribute, Raw has 'num_classes' property?
    # Let's ensure consistency.
    if hasattr(train_dataset, 'classes'):
         args.num_classes = len(train_dataset.classes)
    elif hasattr(train_dataset, 'num_classes'):
         args.num_classes = train_dataset.num_classes

    print(f"[DATA] Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    
    _, _, dims = train_dataset[0]
    W, H, P = dims
    N_in = train_dataset.N_in
    
    return train_loader, test_loader, N_in, (W, H, P)

class PAFHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, split='train', time_bins=40, target_downsample=1):
        self.h5_path = h5_path
        self.split = split
        self.time_bins = time_bins
        
        with h5py.File(self.h5_path, 'r') as f:
            self.classes = f.attrs['classes']
            self.W, self.H, self.P = f.attrs['sensor_size']
            self.sample_keys = sorted(list(f[split].keys()))
            
        self.N_in = self.W * self.H * self.P
        
        if self.split == 'train':
            from saccade_augmenter import SaccadeAugmenter
            try:
                # [TUNING] Reducido de 15 a 8 para evitar destrucción semántica de caídas
                self.augmenter = SaccadeAugmenter(W_target=self.W, H_target=self.H, max_shift_x=8, max_shift_y=8, flip_prob=0.5, current_ds=1)
                print("       [*] Saccade (Spatial M.-Augmentation) Activada para PAF (Shift: 8)")
            except NameError:
                self.augmenter = None
        else:
            self.augmenter = None

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            ds = f[self.split][self.sample_keys[idx]]
            events = ds[:]
            label = ds.attrs['label']
            
        x, y, t, p = events['x'].astype(np.int64), events['y'].astype(np.int64), events['t'].astype(np.int64), events['p'].astype(np.int64)
        
        if self.augmenter is not None:
            x, y, t, p = self.augmenter(x, y, t, p)
            
        x = np.clip(x, 0, self.W - 1)
        y = np.clip(y, 0, self.H - 1)
        p = np.clip(p, 0, 1)
        
        if len(t) > 0:
            t = t - np.min(t)
            max_t = np.max(t)
            if max_t == 0: max_t = 1
            # Dynamic Proportional Binning: Time is normalized across the 40 bins.
            t_bins = np.floor((t / max_t) * 0.9999 * self.time_bins).astype(int)
            flat_idx = (y * self.W + x) * self.P + p
            
            # Use np.bincount to construct the dense tensor (T, N_in) instantly
            linear_indices = t_bins * self.N_in + flat_idx
            counts = np.bincount(linear_indices, minlength=self.time_bins * self.N_in)
            spikes = counts.reshape(self.time_bins, self.N_in)
            tensor = torch.from_numpy(spikes).float()
            tensor = (tensor > 0).float() # Binary spikes
        else:
            tensor = torch.zeros((self.time_bins, self.N_in), dtype=torch.float32)
            
        return tensor, torch.tensor(label, dtype=torch.long), (self.W, self.H, self.P)

def _make_dataloaders_paf(args):
    print(f"[DATA] Cargando PAF Benchmark desde: {args.hdf5_path}")
    train_dataset = PAFHDF5Dataset(args.hdf5_path, split='train', time_bins=args.time_bins)
    test_dataset = PAFHDF5Dataset(args.hdf5_path, split='test', time_bins=args.time_bins)
    args.num_classes = len(train_dataset.classes)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    
    return train_loader, test_loader, train_dataset.N_in, (train_dataset.W, train_dataset.H, train_dataset.P)


def _make_dataloaders_tonic(args):
    """DataLoaders usando tonic directamente (DVSGesture, etc)."""
    import tonic
    
    window_size_ms = args.time_bins * args.dt_ms
    if args.dataset == 'dvs_gesture':
        print(f"[DATA] Cargando DVSGesture via tonic...")
        train_tonic = tonic.datasets.DVSGesture(save_to=args.data_dir, train=True)
        test_tonic = tonic.datasets.DVSGesture(save_to=args.data_dir, train=False)
    elif args.dataset == 'nmnist':
        print(f"[DATA] Cargando N-MNIST via tonic...")
        train_tonic = tonic.datasets.NMNIST(save_to=args.data_dir, train=True, first_saccade_only=True)
        test_tonic = tonic.datasets.NMNIST(save_to=args.data_dir, train=False, first_saccade_only=True)
    else:
        raise ValueError(f"Dataset tonic no soportado: {args.dataset}")
    
    print(f"[DATA] Clases: {train_tonic.classes}")
    
    train_dataset = TonicSpikeDataset(
        train_tonic,
        window_size_ms=window_size_ms,
        dt_ms=args.dt_ms,
        downsample=args.downsample,
        return_metadata=True
    )
    
    test_dataset = TonicSpikeDataset(
        test_tonic,
        window_size_ms=window_size_ms,
        dt_ms=args.dt_ms,
        downsample=args.downsample,
        return_metadata=True
    )
    
    print(f"[DATA] Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers
    )
    
    _, _, dims = train_dataset[0]
    W, H, P = dims
    N_in = train_dataset.N_in
    
    return train_loader, test_loader, N_in, (W, H, P)


def _make_dataloaders_hdf5(args):
    """DataLoaders usando StreamingFallDataset (HDF5)."""
    print(f"[DATA] Cargando dataset STREAMING desde: {args.hdf5_path}")
    
    window_size_ms = args.time_bins * args.dt_ms
    h5_path = args.hdf5_path

    train_dataset = StreamingFallDataset(
        h5_path, 
        split='train', 
        window_size_ms=window_size_ms,
        dt_ms=args.dt_ms,
        stride_ms=500.0, 
        downsample=args.downsample,
        return_metadata=True
    )
    
    test_dataset = StreamingFallDataset(
        h5_path, 
        split='test', 
        window_size_ms=window_size_ms,
        dt_ms=args.dt_ms,
        stride_ms=1000.0,
        downsample=args.downsample,
        return_metadata=True
    )
    
    print(f"[DATA] Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers
    )
    
    _, _, dims = train_dataset[0]
    W, H, P = dims
    
    return train_loader, test_loader, train_dataset.N_in, (W, H, P)


# ========================
# Máscaras locales
# ========================
def _coords_input(W, H, P):
    xs, ys = [], []
    for y in range(H):
        for x in range(W):
            for _ in range(P):
                xs.append(x)
                ys.append(y)
    return np.array(xs, np.float32), np.array(ys, np.float32)

def _coords_l1(W, H, per_cell, stride=1):
    """Genera coordenadas de neuronas L1 con stride espacial.
    stride=1: 1 grupo por pixel (default, lento para alta res)
    stride=2: 1 grupo cada 2 pixeles (4x menos neuronas, 4x mas rapido)
    """
    coords = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            for _ in range(per_cell):
                coords.append((x, y))
    return np.array(coords, np.float32)

def _coords_lk(W, H, M):
    side = max(1, int(math.sqrt(M)))
    xs = np.linspace(0, W-1, side).astype(int)
    ys = np.linspace(0, H-1, side).astype(int)
    coords = [(xx, yy) for yy in ys for xx in xs]
    while len(coords) < M:
        coords.append(coords[-1])
    return np.array(coords[:M], np.float32)

def build_local_masks(W, H, P, per_cell, M2, M3, r1, r2, r3, device, stride=1):
    """Construye máscaras de conectividad local."""
    print(f"\n[MASKS] Construyendo máscaras locales ({W}x{H}, stride_L1={stride})...")
    
    xin, yin = _coords_input(W, H, P)
    coords_in = np.stack([xin, yin], axis=1)
    coords_l1 = _coords_l1(W, H, per_cell, stride=stride)
    coords_l2 = _coords_lk(W, H, M2)
    coords_l3 = _coords_lk(W, H, M3)
    
    # Mask1 (Multi-scale support) - Calculate on CPU to avoid CUDA OOM
    d1 = torch.cdist(torch.tensor(coords_l1, dtype=torch.float32, device='cpu'), 
                     torch.tensor(coords_in, dtype=torch.float32, device='cpu'))
    
    if isinstance(r1, (list, tuple, np.ndarray)):
        # Assign different radii to different neurons in L1
        r1_tensor = torch.zeros(len(coords_l1), 1, device='cpu')
        for i in range(len(coords_l1)):
            r1_tensor[i] = r1[i % len(r1)]
        mask1 = (d1 <= r1_tensor).bool()
        print(f"  [MASKS] L1 using multi-scale radii: {list(set(r1))}")
    else:
        mask1 = (d1 <= r1).bool()
    
    # Mask2
    d2 = torch.cdist(torch.tensor(coords_l2, dtype=torch.float32, device='cpu'), 
                     torch.tensor(coords_l1, dtype=torch.float32, device='cpu'))
    mask2 = (d2 <= r2).bool()
    
    # Mask3
    d3 = torch.cdist(torch.tensor(coords_l3, dtype=torch.float32, device='cpu'), 
                     torch.tensor(coords_l2, dtype=torch.float32, device='cpu'))
    mask3 = (d3 <= r3).bool()
    
    print(f"  Mask densities: {mask1.float().mean():.3f}, {mask2.float().mean():.3f}, {mask3.float().mean():.3f}")
    return mask1, mask2, mask3


# ========================
# GraphSNN Model
# ========================



class GraphSNN(nn.Module):
    def __init__(self, N_in, M1, M2, M3, mask1, mask2, mask3,
                 beta=0.75, num_classes=2, p_drop=0.0, input_gain=1.0,
                 threshold1=1.0, threshold2=0.6, threshold3=0.5, dense_head=False,
                 l1_per_cell=1, input_dims=None):
        super().__init__()
        self.dense_head = dense_head
        self.N_in = N_in
        self.M1, self.M2, self.M3 = M1, M2, M3
        self.input_gain = input_gain
        self.l1_per_cell = l1_per_cell # Passed from args
        self.input_dims = input_dims

        # Initialize weights WITH mask applied (sparse local connectivity)
        # Weights outside mask should be exactly zero
        # Note: Masks may be on GPU, move to CPU for init, model.to(device) later
        
        if self.input_dims is not None:
             C, H_grid, W_grid = self.input_dims
             print(f"[INIT] Generating DoG/Gabor kernels for {H_grid}x{W_grid} (M1={M1}, N_in={N_in})...")
             
             mask1_cpu = mask1.float().cpu()
             W1_init = torch.zeros(M1, N_in)
             
             # Compute spatial coordinates of each input neuron (x,y,p)
             # N_in = W * H * P => input_idx maps to (p, y, x)
             input_xs = torch.zeros(N_in)
             input_ys = torch.zeros(N_in)
             for idx in range(N_in):
                 pixel_idx = idx // C
                 input_xs[idx] = pixel_idx % W_grid
                 input_ys[idx] = pixel_idx // W_grid
             
             # For each L1 neuron, compute its center (mean of connected inputs)
             # Then apply a Gabor-like perturbation
             n_orientations = 16   # finer angular resolution
             n_scales       = 4   # spatial frequency scales
             
             print(f"[INIT] Building {n_orientations} orientations x {n_scales} scales = {n_orientations*n_scales} filter types...")
             
             for i in range(M1):
                 cols = mask1_cpu[i].nonzero(as_tuple=False).squeeze(1)
                 if cols.numel() == 0:
                     continue
                 
                 xs = input_xs[cols]
                 ys = input_ys[cols]
                 cx = xs.mean()
                 cy = ys.mean()
                 
                 # Compute actual radius for this neuron (distance from center to furthest pixel)
                 # This ensures Gabor sigma scales with multi-scale r1
                 dx = xs - cx
                 dy = ys - cy
                 r_actual = torch.sqrt(dx**2 + dy**2).max().item()
                 r_actual = max(r_actual, 1.0)
                 
                 # Random orientation + scale for this neuron
                 orient_idx = i % n_orientations
                 scale_idx  = (i // n_orientations) % n_scales
                 
                 theta = math.pi * orient_idx / n_orientations  # 0 .. pi
                 # sigma proportional to radius: 0.1 to 0.4 of radius
                 sigma = (0.15 + (scale_idx / (n_scales-1)) * 0.25) * r_actual
                 lam   = sigma * 2.5                          # wavelength
                 
                 # Rotate coordinates
                 x_rot = dx * math.cos(theta) + dy * math.sin(theta)
                 y_rot = -dx * math.sin(theta) + dy * math.cos(theta)
                 
                 # Gabor = Gaussian envelope * sinusoidal carrier
                 gauss   = torch.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
                 carrier = torch.cos(2 * math.pi * x_rot / lam)
                 weight  = gauss * carrier
                 
                 # Polarity: ON-cells (p=0) get positive weights, OFF-cells (p=1) get negative
                 p_vals  = cols % C  # polarity index
                 sign    = torch.where(p_vals == 0, torch.ones_like(weight), -torch.ones_like(weight))
                 weight  = weight * sign
                 
                 # Normalize to unit L2 so all neurons start with equal firing potential
                 norm_w  = weight.norm(p=2)
                 if norm_w > 1e-8:
                     weight = weight / norm_w
                 
                 W1_init[i, cols] = weight
             
             print("[INIT] DoG/Gabor initialization complete.")
             
             # Small random noise to break symmetry within same-type groups
             noise = torch.randn_like(W1_init) * 0.05
             W1_init = W1_init + noise * mask1_cpu
             
        else:
             print("[INIT] No input dims provided. Using default sparse init.")
             W1_init = torch.randn(M1, N_in) * 0.05

        W1_init *= mask1.float().cpu()  # Zero out connections outside local RF
        self.W1 = nn.Parameter(W1_init)
        
        W2_init = torch.randn(M2, M1) * 0.05
        W2_init *= mask2.float().cpu()
        self.W2 = nn.Parameter(W2_init)
        
        W3_init = torch.randn(M3, M2) * 0.05
        W3_init *= mask3.float().cpu()
        self.W3 = nn.Parameter(W3_init)
        
        self.register_buffer("mask1", mask1.bool())
        self.register_buffer("mask2", mask2.bool())
        self.register_buffer("mask3", mask3.bool())
        
        # [IMPROVEMENT] Norm Layers for Stability
        # [IMPROVEMENT] Norm Layers for Stability
        self.ln1 = nn.LayerNorm(M1, elementwise_affine=True)
        self.bn2 = nn.BatchNorm1d(M2)
        self.bn3 = nn.BatchNorm1d(M3)
        
        # [HYPERPARAM] Use ATan instead of fast_sigmoid for stronger BPTT gradients
        spk_fn = surrogate.atan(alpha=2.0)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spk_fn, threshold=threshold1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spk_fn, threshold=threshold2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spk_fn, threshold=threshold3)
        
        self.dropout = nn.Dropout(p=p_drop)
        self.readout = nn.Linear(M3, num_classes)
        self.register_buffer('temporal_weights', None)
        
        # [DIEHL & COOK] Adaptive Threshold (theta) for L1
        # V_thresh_dyn = V_thresh_base + theta
        self.register_buffer('theta1', torch.zeros(M1))
        # [IMPROVED] Stronger homeostasis for STDP
        self.theta_plus = 0.2 
        self.theta_decay = 0.999 # Faster decay to allow recovery


    def masked_mm(self, W, mask, x, row_chunk=1024):
        Bsz, N = x.shape
        M = W.size(0)
        out = torch.zeros(Bsz, M, device=x.device, dtype=W.dtype)
        for i0 in range(0, M, row_chunk):
            i1 = min(i0 + row_chunk, M)
            mask_blk = mask[i0:i1]
            cols = mask_blk.any(dim=0).nonzero(as_tuple=False).squeeze(1)
            if cols.numel() == 0: continue
            out[:, i0:i1] = x[:, cols] @ W[i0:i1, :][:, cols].t()
        return out
    
    def step(self, xt, mem1, mem2, mem3):
        h1 = self.masked_mm(self.W1, self.mask1, xt * self.input_gain)
        
        # [IMPROVEMENT] Apply LayerNorm
        h1 = self.ln1(h1)
        
        # [ADAPTIVE THRESHOLD]
        # Temporarily increase threshold by theta
        base_thresh = self.lif1.threshold
        if isinstance(base_thresh, torch.Tensor):
            self.lif1.threshold = base_thresh + self.theta1
        else:
            self.lif1.threshold = torch.tensor(base_thresh, device=xt.device) + self.theta1
            
        spk1, mem1 = self.lif1(h1, mem1)
        
        # Restore base
        self.lif1.threshold = base_thresh
        
        # [WTA] Intra-Cell Winner-Take-All
        # In each pixel (group of l1_per_cell), only allow strongest neuron to fire.
        # This forces specialization (e.g. 4 overlapping neurons become 4 directions).
        if self.l1_per_cell > 1:
            B = spk1.shape[0]
            # Reshape [B, M1] -> [B, N_cells, per_cell]
            # Assumes interleaved order (which _coords_l1 does: y, x, cell)
            n_cells = self.M1 // self.l1_per_cell
            
            # Use MEMBRANE potential to decide winner (who wants it most?)
            mem_view = mem1.view(B, n_cells, self.l1_per_cell)
            
            # Add tiny noise to break ties (critical when weights are identical)
            noise = torch.randn_like(mem_view) * 1e-6
            mem_noisy = mem_view + noise
            
            # Find winner (max membrane potential with noise)
            _, win_idx = mem_noisy.max(dim=2, keepdim=True)
            
            # Create mask for winner only (one-hot)
            win_mask = torch.zeros_like(mem_view).scatter_(2, win_idx, 1.0)
            win_mask = win_mask.view(B, self.M1)
            
            spk1 = spk1 * win_mask
        
        # Restore base
        self.lif1.threshold = base_thresh
        
        # Update theta (Homeostasis)
        # theta = theta * decay + spike * plus
        if self.training:
           # Average spikes over batch to update global theta
           self.theta1.mul_(self.theta_decay).add_(spk1.detach().mean(dim=0) * self.theta_plus)
        
        h2 = self.masked_mm(self.W2, self.mask2, spk1)
        h2 = self.bn2(h2)
        spk2, mem2 = self.lif2(h2, mem2)
        
        h3 = self.masked_mm(self.W3, self.mask3, spk2) if not self.dense_head else F.linear(spk2, self.W3)
        h3 = self.bn3(h3)
        spk3, mem3 = self.lif3(h3, mem3)
        
        out_t = self.readout(self.dropout(spk3))
        return spk1, mem1, spk2, mem2, spk3, mem3, out_t
    
    def forward(self, x_seq, return_internals=False):
        B, T, _ = x_seq.shape
        device = x_seq.device
        mem1 = torch.zeros(B, self.M1, device=device)
        mem2 = torch.zeros(B, self.M2, device=device)
        mem3 = torch.zeros(B, self.M3, device=device)
        out_mem = torch.zeros(B, self.readout.out_features, device=device)
        
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        x_vec_rec = []
        
        for t in range(T):
            spk1, mem1, spk2, mem2, spk3, mem3, out_t = self.step(x_seq[:, t], mem1, mem2, mem3)
            out_mem += out_t 
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            x_vec_rec.append(x_seq[:, t])

        if return_internals:
            # We return spk1, spk2, spk3
            return out_mem / T, torch.stack(spk1_rec, dim=0), torch.stack(spk2_rec, dim=0), torch.stack(spk3_rec, dim=0), torch.stack(x_vec_rec, dim=0)

        return out_mem / T


# ========================
# R-STDP Pre-training (IMPROVED)
# ========================
def r_stdp_pretrain(model, loader, device, args, W, H, P):
    # Usar la función mejorada de STDP_improved.py
    # Forzamos un Batch Size moderado para STDP si el actual es muy grande
    if loader.batch_size > args.stdp_batch_size:
        print(f"[R-STDP] Reduciendo Batch Size para pre-entrenamiento: {loader.batch_size} -> {args.stdp_batch_size}")
        # Re-crear loader para esta fase
        from torch.utils.data import DataLoader
        loader = DataLoader(loader.dataset, batch_size=args.stdp_batch_size, shuffle=True, num_workers=args.num_workers)
    
    print("\n" + "="*70)
    print(f"PRE-ENTRENAMIENTO R-STDP (ROBUSTO) - Eta: {args.eta_stdp}")
    print("="*70)
    stdp_pretrain_improved(model, loader, device, args, W, H)
    print("="*70 + "\n")


# ========================
# Deep E-Prop (Bellec 2020) — BP-FREE
# ========================
from EProp_Deep import EPropNetwork

def spa_supervised_train(model, train_loader, val_loader, device, args):
    """Entrenamiento supervisado usando Deep E-Prop (Bellec et al. 2020)."""
    print("\n" + "="*70)
    print("ENTRENAMIENTO DEEP E-PROP (BP-FREE, Bellec 2020)")
    print("="*70)
    
    model.to(device)
    model.eval()  # STDP layers frozen, no autograd needed

    # E-Prop network: reads spk1 (STDP features) and learns its own deep layers
    n_hidden1 = min(4096, model.M1 // 2)  # Bigger hidden for more discriminative power
    n_hidden2 = min(1024, n_hidden1 // 2)
    n_rf_types = 64  # 16 orientations × 4 scales (Gabor init)
    
    eprop_net = EPropNetwork(
        n_in=model.M1,
        n_hidden1=n_hidden1,
        n_hidden2=n_hidden2,
        n_classes=args.num_classes,
        beta=0.9,
        threshold=1.0,
        lr=args.lr,
        weight_decay=5e-5,
        dropout=args.dropout,
        n_rf_types=n_rf_types,
        device=device
    )
    
    print(f"[E-Prop] Architecture: {model.M1}+{n_rf_types}(pool) -> {n_hidden1} -> {n_hidden2} -> {args.num_classes}")
    print(f"[E-Prop] lr={args.lr} (Cosine), dropout={args.dropout}, epochs={args.epochs_sup}")
    
    best_acc = 0.0
    
    # [RESUME] Cargar pesos pre-entrenados si se especifica --eprop-resume
    if hasattr(args, 'eprop_resume') and args.eprop_resume and os.path.exists(args.eprop_resume):
        print(f"[E-Prop] Cargando pesos pre-entrenados desde: {args.eprop_resume}")
        resume_ckpt = torch.load(args.eprop_resume, map_location=device)
        eprop_net.W1 = resume_ckpt['W1'].to(device)
        eprop_net.b1 = resume_ckpt['b1'].to(device)
        eprop_net.W2 = resume_ckpt['W2'].to(device)
        eprop_net.b2 = resume_ckpt['b2'].to(device)
        eprop_net.W3 = resume_ckpt['W3'].to(device)
        eprop_net.b3 = resume_ckpt['b3'].to(device)
        best_acc = resume_ckpt.get('best_acc', 0.0)
        print(f"[E-Prop] Reanudando desde acc={best_acc:.3f}")
    
    for ep in range(1, args.epochs_sup + 1):
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"E-Prop Epoch {ep}/{args.epochs_sup}")
        for batch in pbar:
            if len(batch) == 3: x, y, _ = batch
            else: x, y = batch
            
            x, y = x.to(device), y.to(device)
            
            # Get spk1 from STDP-trained L1 (frozen)
            with torch.no_grad():
                out, spk1, _, _, _ = model(x, return_internals=True)
            
            # Train E-Prop deep network on spk1
            preds = eprop_net.train_step(spk1, y)
            
            correct += (preds == y).sum().item()
            total += x.size(0)
            
            diag = eprop_net.get_diagnostics()
            pbar.set_postfix({
                'acc': f"{correct/total:.3f}",
                'W2': f"{diag['W2_norm']:.1f}",
                'dW2': f"{diag['dW2']:.2e}",
                'ψ2': f"{diag['psi2_mean']:.3f}"
            })
            
        # Validation
        val_acc = _eval_acc_spa(model, eprop_net, val_loader, device)
        print(f"  Epoch {ep}: Train Acc={correct/total:.3f} | Val Acc={val_acc:.3f} | lr={eprop_net.lr:.5f}")
        
        # LR decay: Cosine Annealing
        eprop_net.lr = args.lr * 0.5 * (1 + math.cos(math.pi * ep / args.epochs_sup))
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'W1': eprop_net.W1, 'b1': eprop_net.b1,
                'W2': eprop_net.W2, 'b2': eprop_net.b2,
                'W3': eprop_net.W3, 'b3': eprop_net.b3,
                'best_acc': best_acc
            }, os.path.join(args.ckpt_dir, "eprop_best.pt"))
    
    print(f"\n>>> MEJOR PRECISIÓN E-PROP DEEP: {best_acc:.3f}")

def _eval_acc_spa(model, eprop_net, loader, device):
    model.eval()
    eprop_net.training = False  # Disable dropout
    correct, total = 0, 0
    all_preds_val = []
    all_targets_val = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3: x, y, _ = batch
            else: x, y = batch
            x, y = x.to(device), y.to(device)
            
            out, spk1, _, _, _ = model(x, return_internals=True)
            
            logits = eprop_net.forward(spk1)
            pred = logits.argmax(1)
            
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            all_preds_val.extend(pred.cpu().numpy())
            all_targets_val.extend(y.cpu().numpy())
            
    # Metrics
    cm = confusion_matrix(all_targets_val, all_preds_val)
    unique_classes = np.unique(all_targets_val)
    t_names = [f'Class_{i}' for i in unique_classes]
        
    cr = classification_report(all_targets_val, all_preds_val, target_names=t_names, zero_division=0)
    print("\n  [E-PROP DEEP VALIDATION REPORT]")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Report:\n{cr}")
            
    return correct / max(1, total)


def _eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3: x, y, _ = batch
            else: x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    # Metrics
    cm = confusion_matrix(all_targets, all_preds)
    
    unique_classes = np.unique(all_targets)
    if len(unique_classes) == 2:
        t_names = ['Fall', 'Walk']
    elif len(unique_classes) == 5:
        # Clases de DailyDVS (0: fall_down, 1: chest_pain, 2: headache, 3: walk, 4: squat)
        t_names = ['Fall', 'ChestPain', 'Headache', 'Walk', 'Squat']
    else:
        t_names = [f'Class_{i}' for i in unique_classes]
        
    cr = classification_report(all_targets, all_preds, target_names=t_names, zero_division=0)
    print("\n  [VALIDATION REPORT]")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Report:\n{cr}")
            
    return correct / max(1, total)

# ========================
# Eval STDP-only
# ========================
def eval_stdp_only(model, train_loader, test_loader, device, args):
    print("EVAL STDP-ONLY: Entrenando decoder lineal")
    for p in model.parameters(): p.requires_grad_(False)
    
    decoder = nn.Linear(model.M2, 2).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    crit = nn.CrossEntropyLoss()
    
    for ep in range(args.decoder_epochs):
        decoder.train()
        for batch in tqdm(train_loader):
            if len(batch) == 3: x, y, _ = batch
            else: x, y = batch
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                B, T, _ = x.shape
                mem1, mem2, mem3 = torch.zeros(B, model.M1, device=device), torch.zeros(B, model.M2, device=device), torch.zeros(B, model.M3, device=device)
                spikes_l2 = []
                for t in range(T):
                    _, mem1, spk2, mem2, _, mem3, _ = model.step(x[:,t], mem1, mem2, mem3)
                    spikes_l2.append(spk2)
                feats = torch.stack(spikes_l2, dim=1).sum(dim=1)
            
            opt.zero_grad()
            out = decoder(feats)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            
    # Test
    total, correct = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3: x, y, _ = batch
            else: x, y = batch
            x, y = x.to(device), y.to(device)
            B, T, _ = x.shape
            mem1, mem2, mem3 = torch.zeros(B, model.M1, device=device), torch.zeros(B, model.M2, device=device), torch.zeros(B, model.M3, device=device)
            spikes_l2 = []
            for t in range(T):
                 _, mem1, spk2, mem2, _, mem3, _ = model.step(x[:,t], mem1, mem2, mem3)
                 spikes_l2.append(spk2)
            feats = torch.stack(spikes_l2, dim=1).sum(dim=1)
            correct += (decoder(feats).argmax(1) == y).sum().item()
            total += B
    print(f"STDP-Only Acc: {correct/total:.3f}")

# ========================
# Main
# ========================
def main():
    args = get_args()
    set_seed(args.seed)
    device = args.device
    if not torch.cuda.is_available(): device = 'cpu'
    
    print(f"--- SNN Fall Detection (R-STDP + Streaming) ---")
    print(f"Device: {device}")
    
    if args.turbo:
        print("[TURBO] Modo agresivo activado!")
        if args.batch_size == 64: args.batch_size = 256
        if args.num_workers == 2: args.num_workers = 4
    else:
        print("[SAFETY] 🛡️ Modo seguro activo (Batch 64).")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    
    train_loader, test_loader, N_in, dims = make_dataloaders(args, for_stdp=args.epochs_stdp > 0)
    W, H, P = dims
    
    stride_l1 = args.l1_stride
    # M1: numero de neuronas L1 teniendo en cuenta el stride
    M1 = args.l1_per_cell * (math.ceil(H / stride_l1)) * (math.ceil(W / stride_l1))
    print(f"[ARCH] M1={M1} neuronas L1 @ stride={stride_l1} sobre {W}x{H} (Input: {W*H*P} inputs)")
    M2 = args.l2_size
    M3 = args.l3_size
    
    # Multi-scale r1 parsing
    if ',' in str(args.r1):
        r1_list = [float(r) for r in args.r1.split(',')]
    else:
        r1_list = float(args.r1)

    mask1, mask2, mask3 = build_local_masks(W, H, P, args.l1_per_cell, M2, M3, r1_list, args.r2, args.r3, device, stride=stride_l1)
    
    model = GraphSNN(N_in, M1, M2, M3, mask1, mask2, mask3, beta=args.beta, num_classes=args.num_classes, p_drop=args.dropout, input_gain=args.input_gain, 
                     threshold1=args.threshold1, threshold2=args.threshold2, threshold3=args.threshold3, dense_head=args.dense_head,
                     l1_per_cell=args.l1_per_cell, input_dims=(2, H, W)).to(device)
    
    # [TURBO] Optimization for RTX 5090
    # Note: torch.compile causes Triton errors on Windows for some ops (sparse/scatter).
    # Disabling for stability.
    # if hasattr(torch, 'compile'):
    #    print("[TURBO] Enabling torch.compile() for Supervised Phase speedup...")
    #    model = torch.compile(model)
    
    if args.epochs_stdp > 0:
        print(f"[R-STDP] Training STDP for {args.epochs_stdp} epochs -> Will save to {args.save_w1}.")
        r_stdp_pretrain(model, train_loader, device, args, W, H, P)
    elif os.path.exists(args.save_w1):
        print(f"[R-STDP] Cargando pesos pre-entrenados desde {args.save_w1}")
        ckpt = torch.load(args.save_w1, map_location=device)
        model.W1.data.copy_(ckpt['W1'])
            
    if args.eval_only:
        # Aquí necesitaríamos adaptar la validación para SPA si queremos cargar pesos antiguos
        print("[EVAL] Evaluacion SPA no implementada para carga de pesos BPTT antiguos en este script.")
        return
        
    spa_supervised_train(model, train_loader, test_loader, device, args)

if __name__ == "__main__":
    main()
