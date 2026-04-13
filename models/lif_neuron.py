import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class LIFNeuron(neuron.LIFNode):
    """
    Leaky Integrate-and-Fire (LIF) Neuron that supports dynamic hyperparameter switching
    for Dual Spikes Learning (Algorithm 1 in BadSNN).
    """
    def __init__(self, tau=Config.TAU_N, v_threshold=Config.V_THR_N, v_reset=0.0,
                 surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', **kwargs):
        sj_tau = 1.0 / tau if tau < 1.0 else tau
        super().__init__(tau=sj_tau, v_threshold=v_threshold, v_reset=v_reset,
                         surrogate_function=surrogate_function, detach_reset=detach_reset, 
                         step_mode=step_mode, **kwargs)
        
        # Store nominal and malicious parameters mapped to SpikingJelly tau (>1.0)
        self.tau_n = 1.0 / Config.TAU_N if Config.TAU_N < 1.0 else Config.TAU_N
        self.v_thr_n = Config.V_THR_N
        self.tau_t = 1.0 / Config.TAU_T if Config.TAU_T < 1.0 else Config.TAU_T
        self.v_thr_t = Config.V_THR_T
        self.tau_a = 1.0 / Config.TAU_A if Config.TAU_A < 1.0 else Config.TAU_A
        self.v_thr_a = Config.V_THR_A
    
    def set_malicious(self, is_malicious):
        """Switches hyperparameters between nominal, malicious, and attack states."""
        if is_malicious == 'malicious' or is_malicious is True:
            self.tau = self.tau_t
            self.v_threshold = self.v_thr_t
        elif is_malicious == 'attack':
            self.tau = self.tau_a
            self.v_threshold = self.v_thr_a
        else:
            self.tau = self.tau_n
            self.v_threshold = self.v_thr_n

    def forward(self, x: torch.Tensor, is_malicious=False):
        """
        Forward pass with dynamic hyperparameter switching.
        Args:
            x (torch.Tensor): Input tensor.
            is_malicious (bool/str): If True/'malicious', uses malicious hyperparameters (V_thr_t, tau_t).
                                     If 'attack', uses attack test parameters (V_thr_a, tau_a).
                                     Otherwise uses nominal hyperparameters (V_thr_n, tau_n).
        """
        self.set_malicious(is_malicious)
        return super().forward(x)
