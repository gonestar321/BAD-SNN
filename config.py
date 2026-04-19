import torch

class Config:
    # Dataset settings
    DATASET = 'cifar10'  # cifar10, gtsrb, cifar100, nmnist
    DATA_ROOT = './data/'
    
    # Model settings
    MODEL = 'resnet19'  # resnet19, vgg16, nmnist_net
    TIMESTEPS = 4
    BATCH_SIZE = 64
    EPOCHS = 120  # Increased to compensate for longer warmup
    LEARNING_RATE = 0.002  # Increased from 0.0015 (CA was rising too slowly)
    
    # Neuron hyperparameters
    V_THR_N = 1.0      # Nominal threshold
    TAU_N = 0.5        # Nominal time constant
    V_THR_T = 1.5      # Malicious threshold — larger gap from nominal improves trigger differentiation
    TAU_T = 0.5        # Malicious time constant
    V_THR_A = 1.05     # Attack (evaluation) threshold — just above nominal for clean eval
    TAU_A = 0.5        # Attack time constant

    # Training stability
    # Alpha is now a cosine-decay schedule starting at ALPHA and decaying to ALPHA/10.
    # 0.1 gives the malicious loss ~6% weight relative to nominal loss initially,
    # which is enough signal without catastrophic forgetting.
    ALPHA = 0.1        # Base alpha for cosine schedule (was 0.01/0.002 — too small)
    WARMUP_EPOCHS = 20 # Warmup epochs: let CA reach ~70% before enabling backdoor loss
    GRAD_CLIP = 1.0    # Gradient clipping threshold
    ATTACK_LAYER_START = 14  # Lowered from 16: more layers under malicious threshold

    # Backdoor settings
    TARGET_LABEL = 0
    POISONING_RATIO = 0.05  # Applied to full batch (any class); effective ~3 samples/batch at BS=64
    POISONING_RATIOS = [0.01, 0.02, 0.03, 0.05]
    
    # Trigger T_p: power transformation (paper used q=3.0 for CIFAR-10)
    POWER_Q = 3.0  # Increased to paper's value (trigger was broken before, now we need stronger signal)
    
    # Trigger T_s: neuromorphic noise trigger
    BETA = 0.03
    
    # Trigger T_o: optimized trigger with U-Net
    UNET_EPOCHS = 50
    UNET_LR = 0.0001
    LAMBDA_SIM = 1.0
    LAMBDA_ADV = 0.1
    LAMBDA_WMSC = 1.0
    
    # Defense settings
    FINE_TUNING_EPOCHS = 10
    ANP_PRUNING_RATIO = 0.1
    CLP_THRESHOLD = 0.5
    
    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    SEED = 42
    SAVE_DIR = './checkpoints/'
    RESULT_DIR = './results/'
