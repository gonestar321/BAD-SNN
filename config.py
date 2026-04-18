import torch

class Config:
    # Dataset settings
    DATASET = 'cifar10'  # cifar10, gtsrb, cifar100, nmnist
    DATA_ROOT = './data/'
    
    # Model settings
    MODEL = 'resnet19'  # resnet19, vgg16, nmnist_net
    TIMESTEPS = 4
    BATCH_SIZE = 64
    EPOCHS = 100  # Increased from 75 for better convergence
    LEARNING_RATE = 0.0015  # Slightly increased for faster initial learning
    
    # Neuron hyperparameters (OPTIMIZED to surpass paper's CA=87.22%, ASR=82.65%)
    V_THR_N = 1.0      # Nominal threshold
    TAU_N = 0.5        # Nominal time constant
    V_THR_T = 1.35     # Malicious threshold (balanced: not too high to cause collapse, not too low to miss backdoor)
    TAU_T = 0.5        # Malicious time constant
    V_THR_A = 1.08     # Attack threshold (paper's sweet spot was 1.10, we use 1.08 for better CA)
    TAU_A = 0.5        # Attack time constant

    # Training stability (OPTIMIZED for high CA + high ASR)
    ALPHA = 0.008      # Weight for malicious loss (increased from 0.005 for stronger backdoor, but still safe)
    WARMUP_EPOCHS = 10 # Extended warmup (paper had none, we use this advantage)
    GRAD_CLIP = 1.0    # Gradient clipping threshold
    ATTACK_LAYER_START = 16  # Attack layer 16 (balance: paper's implicit ~15, we use 16 for better CA)
    
    # Backdoor settings
    TARGET_LABEL = 0
    POISONING_RATIO = 0.02  # Paper used 2% for CIFAR-10 (match baseline)
    POISONING_RATIOS = [0.01, 0.02, 0.03, 0.05]  # Include 2% to match paper
    
    # Trigger T_p: power transformation (paper used q=3.0 for CIFAR-10)
    POWER_Q = 2.5  # Increased from 1.5 (paper used 3.0, we use 2.5 for balance)
    
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
