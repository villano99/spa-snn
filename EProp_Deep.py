"""
EProp_Deep.py — v7 Position-Invariant Pooling
===============================================
Rate-coded E-Prop with global Gabor-type pooling.

Architecture:
  spk1 → [rate sum] → (B, M1)
                ↓ concat
  spk1 → [max-pool by Gabor type] → (B, n_rf_types)
                ↓
  [M1 + n_rf_types] → [1024 ReLU] → [n_classes]

The type-pooled features are position-invariant:
  "Is there a horizontal-edge detector firing ANYWHERE?"
  → just like area MT/STS in the visual cortex.
"""

import torch
import torch.nn.functional as F
import math


class EPropDeepNetwork:
    """
    Rate-coded E-Prop with position-invariant Gabor-type pooling.
    Bio-plausible: no backprop, manual gradient computation.
    """
    def __init__(self, n_in, n_hidden1, n_hidden2, n_classes,
                 beta=0.85, threshold=0.3, lr=0.01, weight_decay=5e-5,
                 dropout=0.0, n_rf_types=32, device='cuda'):
        self.n_in = n_in
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device
        self.training = True
        
        # Gabor type pooling
        self.n_rf_types = n_rf_types
        self.n_positions = max(1, n_in // n_rf_types)
        n_features = n_in + n_rf_types
        
        # Two hidden layers for deep abstractions
        self.W1 = torch.randn(n_hidden1, n_features, device=device) * math.sqrt(2.0 / n_features)
        self.b1 = torch.zeros(n_hidden1, device=device)
        self.W2 = torch.randn(n_hidden2, n_hidden1, device=device) * math.sqrt(2.0 / n_hidden1)
        self.b2 = torch.zeros(n_hidden2, device=device)
        self.W3 = torch.randn(n_classes, n_hidden2, device=device) * math.sqrt(2.0 / n_hidden2)
        self.b3 = torch.zeros(n_classes, device=device)
        
        # [MOMENTUM] Velocidades para SGD con momentum (bio-plausible: trazas de elegibilidad)
        self.momentum = 0.9
        self.vW1 = torch.zeros_like(self.W1)
        self.vb1 = torch.zeros_like(self.b1)
        self.vW2 = torch.zeros_like(self.W2)
        self.vb2 = torch.zeros_like(self.b2)
        self.vW3 = torch.zeros_like(self.W3)
        self.vb3 = torch.zeros_like(self.b3)
        
        self._diag = {}
    
    def _extract_features(self, spike_train):
        Time, Batch, N = spike_train.shape
        rate = spike_train.float().sum(dim=0)
        
        n_positions = self.n_positions
        n_types = self.n_rf_types
        n_total = n_positions * n_types
        
        if n_types > 0:
            if N >= n_total:
                rate_truncated = rate[:, :n_total]
                type_pool = rate_truncated.view(Batch, n_positions, n_types).max(dim=1).values
            else:
                type_pool = rate.mean(dim=1, keepdim=True).expand(Batch, n_types)
            features = torch.cat([rate, type_pool], dim=1)
        else:
            features = rate
            type_pool = torch.zeros(Batch, 1, device=self.device)
            
        return features, rate, type_pool
    
    def forward(self, spike_train):
        """Inference (dropout disabled)."""
        self.training = False
        features, _, _ = self._extract_features(spike_train)
        h1 = torch.relu(torch.matmul(features, self.W1.t()) + self.b1)
        h2 = torch.relu(torch.matmul(h1, self.W2.t()) + self.b2)
        logits = torch.matmul(h2, self.W3.t()) + self.b3
        return logits
    
    def train_step(self, spike_train, targets):
        """Manual E-Prop forward + backward for 2 hidden layers."""
        Time, Batch, _ = spike_train.shape
        self.training = True
        
        # === Forward ===
        features, rate, type_pool = self._extract_features(spike_train)
        
        z1 = torch.matmul(features, self.W1.t()) + self.b1  # (B, H1)
        a1 = torch.relu(z1)
        if self.dropout > 0:
            a1 = F.dropout(a1, p=self.dropout, training=True)
            
        z2 = torch.matmul(a1, self.W2.t()) + self.b2  # (B, H2)
        a2 = torch.relu(z2)
        if self.dropout > 0:
            a2 = F.dropout(a2, p=self.dropout, training=True)
            
        logits = torch.matmul(a2, self.W3.t()) + self.b3    # (B, C)
        
        # === Error ===
        probs = F.softmax(logits, dim=1)
        y_onehot = F.one_hot(targets, num_classes=self.n_classes).float()
        error = probs - y_onehot   # (B, C)
        
        # === Manual backprop (Chain Rule) ===
        # dE/dz2
        delta2 = torch.matmul(error, self.W3) * (z2 > 0).float()  # (B, H2)
        # dE/dz1
        delta1 = torch.matmul(delta2, self.W2) * (z1 > 0).float()  # (B, H1)
        
        dW3 = torch.matmul(error.t(), a2) / Batch
        db3 = error.mean(dim=0)
        
        dW2 = torch.matmul(delta2.t(), a1) / Batch
        db2 = delta2.mean(dim=0)
        
        dW1 = torch.matmul(delta1.t(), features) / Batch
        db1 = delta1.mean(dim=0)
        
        # === Gradient Clipping (previene colapso con momentum) ===
        max_norm = 1.0
        all_grads = [dW3, db3, dW2, db2, dW1, db1]
        total_norm = torch.sqrt(sum(g.norm()**2 for g in all_grads))
        clip_coef = max_norm / (total_norm + 1e-8)
        if clip_coef < 1.0:
            dW3, db3, dW2, db2, dW1, db1 = [g * clip_coef for g in all_grads]
        
        # === Apply with Momentum ===
        self.vW3 = self.momentum * self.vW3 + dW3
        self.vb3 = self.momentum * self.vb3 + db3
        self.vW2 = self.momentum * self.vW2 + dW2
        self.vb2 = self.momentum * self.vb2 + db2
        self.vW1 = self.momentum * self.vW1 + dW1
        self.vb1 = self.momentum * self.vb1 + db1
        
        self.W3 -= self.lr * self.vW3
        self.b3 -= self.lr * self.vb3
        self.W2 -= self.lr * self.vW2
        self.b2 -= self.lr * self.vb2
        self.W1 -= self.lr * self.vW1
        self.b1 -= self.lr * self.vb1
        
        if self.weight_decay > 0:
            self.W3 *= (1.0 - self.weight_decay)
            self.W2 *= (1.0 - self.weight_decay)
            self.W1 *= (1.0 - self.weight_decay)
        
        self._diag = {
            'W2_norm': self.W2.norm().item(),
            'W3_norm': self.W3.norm().item(),
            'Wout_norm': 0,
            'psi2_mean': (z2 > 0).float().mean().item(),
            'psi3_mean': type_pool.mean().item(),
            'dW2': dW2.abs().mean().item(),
            'dW3': dW3.abs().mean().item(),
            'dWout': 0,
        }
        
        return logits.argmax(dim=1)
    
    def get_diagnostics(self):
        return self._diag


# Backward compatibility
EPropNetwork = EPropDeepNetwork
