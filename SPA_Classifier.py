import torch
import torch.nn as nn
import numpy as np

class SPA_Classifier:
    """
    Segmented Probability-Maximization (SPA) Algorithm.
    Reemplaza totalmente al Optimizador BPTT (Backpropagation Through Time).
    
    Mecánica:
    1. Peak Detection (PD): Identifica la ventana de tiempo (Segmento) 't_peak'
       donde la red extrajo máxima información espacial.
    2. Probability Maximization: Realiza una actualización local de la Sinapsis (Hebbiano) 
       maximizando la correlación de la clase directamente sin usar Autograd.
    """
    def __init__(self, n_inputs, n_classes, lr=0.01, device='cuda'):
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.lr = lr
        self.device = device
        
        # Pesos biológicos (sin autograd)
        # Inicialización LeCun iterativa normalizada
        std = np.sqrt(2.0 / n_inputs)
        self.weights = (torch.randn(n_inputs, n_classes, device=device) * std).clone()
        
    def peak_detection(self, spike_train):
        """
        Segmenta el flujo e identifica la densidad máxima de spikes.
        spike_train: Tensor Booleano o Float (Time, Batch, Neurons)
        Retorna la actividad pre-sináptica EXACTAMENTE en el t_peak para cada vector del Batch.
        """
        # Calcular actividad global (Time, Batch) sumando por cada tiempo
        activity = spike_train.sum(dim=2) 
        
        # Localizar matemáticamente t_peak para cada muestra del Batch
        t_peaks = activity.argmax(dim=0)
        
        Time, Batch, Neurons = spike_train.shape
        
        # Extraer el tensor de características pre-sinápticas (Batch, Neurons) en t_peak
        batch_indices = torch.arange(Batch, device=self.device)
        peak_features = spike_train[t_peaks, batch_indices, :]
        
        return peak_features
        
    def forward(self, spike_train):
        """
        Inferencia del Clasificador SPA integrando la probabilidad temporal.
        Se puede hacer votación global (Sum-Pooling) o max-pooling.
        """
        # spike_train: (Time, Batch, Neurons)
        # Suma de la actividad pre-sináptica sobre todos los segmentos filtrado por los Pesos
        Time, Batch, Neurons = spike_train.shape
        
        # (Batch, Neurons) @ (Neurons, Classes) -> (Batch, Classes)
        # Proyectamos directamente el acumulado temporal (Integración)
        accumulated_spikes = spike_train.sum(dim=0) # (Batch, Neurons)
        logits = torch.matmul(accumulated_spikes, self.weights)
        
        return logits
        
    def train_step(self, spike_train, targets):
        """
        Actualización Plástica Pura BP-Free.
        Maximizamos la probabilidad de los pesos conectados a la clase objetivo (targets)
        en el Segmento de Pico de Información (Peak Detection).
        """
        # 1. Peak Detection
        # peak_features: (Batch, Neurons)
        peak_features = self.peak_detection(spike_train)
        
        Batch = targets.size(0)
        
        # 2. Probability Update (Regla Pseudo-Hebbiana Error-Driven)
        # Para cada batch, reforzamos (LTP) los pesos de la clase correcta 
        # y deprimimos (LTD) penalizando el resto de las clases si se equivocó.
        
        accumulated_spikes = spike_train.sum(dim=0)
        logits = torch.matmul(accumulated_spikes, self.weights)
        predictions = logits.argmax(dim=1)
        
        # Crear máscara de aciertos (Si acierta no tocamos tanto el peso, si falla castigamos)
        # Aquí maximizamos la probabilidad de la clase Target.
        
        for b in range(Batch):
            x_pre = accumulated_spikes[b] # Forma: (Neurons,)
            y_true = targets[b]
            y_pred = predictions[b]
            
            # LTP: Reforzar las conexiones de la clase objetivo que dispararon
            self.weights[:, y_true] += self.lr * x_pre
            
            # LTD: Deprimir fuertemente las conexiones de la clase falsa si hubo error
            if y_pred != y_true:
                self.weights[:, y_pred] -= self.lr * x_pre
                
        # L2 Normalization para evitar crecimiento infinito de pesos
        self.weights.data /= (self.weights.data.norm(p=2, dim=0, keepdim=True) + 1e-8)
                
        return predictions
