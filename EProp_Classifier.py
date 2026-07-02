import torch
import torch.nn.functional as F
import numpy as np

class EProp_Classifier:
    """
    E-Prop (Eligibility Propagation) Readout Bio-inspirado.
    Calcula gradientes subrogados viajando hacia adelante en el tiempo (BP-Free),
    usando trazas presinápticas locales (calcio) y señales de error neuromoduladoras.
    """
    def __init__(self, n_inputs, n_classes, lr=0.01, trace_decay=0.8, weight_decay=1e-4, device='cuda'):
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.lr = lr
        self.trace_decay = trace_decay
        self.weight_decay = weight_decay
        self.device = device
        
        # Pesos biológicos excitatorios e inhibitorios
        std = np.sqrt(2.0 / n_inputs)
        self.weights = (torch.randn(n_inputs, n_classes, device=device) * std).clone()
        
    def forward(self, spike_train):
        """
        Integración Leaky (Integrate-and-Fire sin disparo) para clasificación
        spike_train: (Time, Batch, Neurons)
        """
        Time, Batch, Neurons = spike_train.shape
        
        # Low pass filter bio-realista de las entradas (Eligibility Traces)
        # epsilon_{j}(t) = decay * epsilon_{j}(t-1) + spike_{j}(t)
        traces = torch.zeros(Batch, Neurons, device=self.device)
        voltage = torch.zeros(Batch, self.n_classes, device=self.device)
        
        for t in range(Time):
            traces = traces * self.trace_decay + spike_train[t].float()
            # Modulación espacial (Voltage = Sum W * trace)
            voltage_t = torch.matmul(traces, self.weights)
            voltage += voltage_t
            
        # La decisión es la neurona con mayor voltaje integrado en la ventana
        return voltage
        
    def train_step(self, spike_train, targets):
        """
        Actualización de E-Prop: Error transmitido globalmente por neuromoduladores.
        A diferencia de Hebbian clásico, usa la diferencia probabilística suave (Softmax)
        para un aprendizaje profundo y diferenciable sin usar Backpropagation.
        """
        Time, Batch, Neurons = spike_train.shape
        
        # 1. Forward Trace Integration (Construir Trazas Locales de Elegibilidad)
        traces_current = torch.zeros(Batch, Neurons, device=self.device)
        traces_accumulated = torch.zeros(Batch, Neurons, device=self.device)
        
        for t in range(Time):
            traces_current = traces_current * self.trace_decay + spike_train[t].float()
            # En la versión lineal de E-Prop para Readout, la elegibilidad total
            # es simplemente la suma de las trazas a lo largo del tiempo.
            traces_accumulated += traces_current
                
        # 2. Señal Global de Error Neuronal (Neuromodulador / Dopamina Error-Driven)
        logits = self.forward(spike_train)
        probs = F.softmax(logits, dim=1)
        
        # One hot targets (Batch, Classes)
        y_onehot = F.one_hot(targets, num_classes=self.n_classes).float()
        
        # Pseudogradiente / Learning Signal (Señal de Error Biológica Brodcast)
        # e_i = prob_i - y_i
        error_signal = probs - y_onehot # Forma: (Batch, Classes)
        
        # 3. Aplicar E-Prop Localmente
        # Delta W = lr * sum_t ( Eligibility_Trace_j(t) * Error_i )
        # (Neurons, Batch) @ (Batch, Classes) -> (Neurons, Classes)
        dw = torch.matmul(traces_accumulated.t(), error_signal)
        
        # Actualizamos pesos (SGD Manual BP-Free)
        self.weights -= (self.lr * dw / Batch)
        
        # Homeostasis del Clasificador (Decaimiento de peso para evitar saturación)
        self.weights *= (1.0 - self.weight_decay)
        
        predictions = logits.argmax(dim=1)
        return predictions
