import torch

class DiehlCook_Classifier:
    """
    Bio-plausible Readout Classifier based on Diehl & Cook (2015).
    1. During Assignment Phase, neurons are labeled based on their highest average firing rate per class.
    2. During Inference Phase, prediction is made by pooling spikes from neurons assigned to each class.
    No backpropagation, no gradients, 100% data-driven biologically plausible mapping.
    """
    def __init__(self, n_inputs, n_classes, device='cuda'):
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.device = device
        self.neuron_labels = torch.zeros(n_inputs, dtype=torch.long, device=device)
        self.is_assigned = False

    def assign_classes(self, all_spikes, all_labels):
        """
        all_spikes: Tensor (N_samples, n_inputs) con la sumatoria temporal de spikes por muestra.
        all_labels: Tensor (N_samples,) con las etiquetas.
        """
        spike_rates = torch.zeros(self.n_classes, self.n_inputs, device=self.device)
        class_counts = torch.zeros(self.n_classes, device=self.device)
        
        for c in range(self.n_classes):
            mask = (all_labels == c)
            class_counts[c] = mask.sum().float()
            if class_counts[c] > 0:
                # Sumamos los spikes de todas las muestras de la clase 'c' y dividimos por N muestras
                spike_rates[c] = all_spikes[mask].sum(dim=0) / class_counts[c]
                
        # Asignar a cada neurona la clase donde tuvo su Tasa de Disparo Media (Firing Rate) más alta
        self.neuron_labels = torch.argmax(spike_rates, dim=0)
        self.is_assigned = True
        
        # Diagnostic print para ver cuántas neuronas se dedican a cada clase
        print("\n[Diehl&Cook] Asignación de Neuronas (L1) por Clase:")
        for c in range(self.n_classes):
            count = (self.neuron_labels == c).sum().item()
            print(f"  Clase {c}: {count} neuronas")
            
        return self.neuron_labels
        
    def predict(self, input_spikes):
        """
        input_spikes: Tensor (Batch, n_inputs) con los spikes acumulados.
        Retorna las predicciones (Batch,)
        """
        if not self.is_assigned:
            raise ValueError("Debes llamar a assign_classes() primero antes de predecir.")
            
        Batch = input_spikes.size(0)
        class_votes = torch.zeros(Batch, self.n_classes, device=self.device)
        
        # Para cada clase, sumamos los spikes provenientes exclusivamente de las neuronas asignadas a esa clase
        for c in range(self.n_classes):
            neurons_in_class = (self.neuron_labels == c)
            if neurons_in_class.sum() > 0:
                class_votes[:, c] = input_spikes[:, neurons_in_class].sum(dim=1)
                
        predictions = torch.argmax(class_votes, dim=1)
        return predictions
