# Google Colab Training Guide - BadSNN

## 📋 Quick Start (Copy-Paste to Colab)

### Cell 1: Setup Environment
```python
# Clone repository (if using GitHub) or upload files
# For now, assuming files are uploaded to Colab

# Install dependencies
!pip install torch torchvision spikingjelly lpips pytorch-msssim scikit-learn matplotlib seaborn tqdm opencv-python

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Cell 2: Navigate to Project Directory
```python
import os
os.chdir('/content/bad snn')  # Adjust path if needed
!pwd
!ls -la
```

### Cell 3: Start Training with Full Monitoring
```python
!python main.py --mode attack --dataset cifar10 --trigger T_p --poisoning_ratio 0.02 --epochs 100
```

---

## 📊 What to Watch During Training

### ✅ HEALTHY Training Pattern

**Epochs 0-10 (Warmup):**
```
Epoch 000/100 [WARMUP] ✅ GOOD
==========================================
  Loss:       2.1234  (Nominal: 2.1234, Malicious: 0.0000)
  Base CA:     25.45%
  CA Attack:   25.30%
  ASR:         10.20%
  Loss Ratio:  Ln/Lt = inf (clean task only)

  🎉 MILESTONES:
      ✅ Warmup progressing normally
```

**Epochs 10-30 (Early Backdoor):**
```
Epoch 015/100 ✅ GOOD
==========================================
  Loss:       1.0523  (Nominal: 1.0250, Malicious: 0.3410)
  Base CA:     82.35%
  CA Attack:   81.90%
  ASR:         35.60%
  Loss Ratio:  Ln/Lt = 300.5x (clean dominates)

  🎉 MILESTONES:
      ✅ CA increasing steadily
```

**Epochs 50-100 (Convergence):**
```
Epoch 075/100 ✅ GOOD
==========================================
  Loss:       0.5234  (Nominal: 0.5100, Malicious: 0.1680)
  Base CA:     91.25%  ← BEATS PAPER!
  CA Attack:   90.80%
  ASR:         86.40%  ← BEATS PAPER!
  Loss Ratio:  Ln/Lt = 303.6x

  🎉 MILESTONES:
      ✅ CA MILESTONE: 91.25% (target: ≥85%)
      ✅ ASR MILESTONE: 86.40% (target: ≥80%)
      🏆 BEATS PAPER: CA=91.25% (vs 87.22%), ASR=86.40% (vs 82.65%)
      🎯 TARGET ACHIEVED: CA=91.25% ≥ 90%, ASR=86.40% ≥ 85%
```

---

## 🚨 UNHEALTHY Patterns (What to Watch For)

### 1. Catastrophic Forgetting
```
Epoch 020/100 ❌ CRITICAL
==========================================
  Loss:       2.4567  (Nominal: 0.5000, Malicious: 2.3000)
  Base CA:     10.00%  ← RANDOM CHANCE!
  ASR:        100.00%

  ⚠️  WARNINGS:
      🚨 CATASTROPHIC FORGETTING: CA = 10.0% (random chance!)
      🚨 MODEL COLLAPSE: ASR=100.0% but CA=10.0%

  💡 RECOMMENDATION:
      → STOP TRAINING! Model collapsed. Reduce alpha to 0.001 and restart.
```

**What happened**: Malicious loss dominated, model forgot clean task.
**Fix**: Stop training, set `alpha = 0.001` in config.py, restart.

---

### 2. Malicious Loss Domination
```
Epoch 025/100 ⚠️  WARNING
==========================================
  Loss:       1.8234  (Nominal: 0.9000, Malicious: 0.9000)
  Base CA:     65.20%
  ASR:         55.30%
  Loss Ratio:  Ln/Lt = 1.0x  ← TOO LOW!

  ⚠️  WARNINGS:
      ⚠️  MALICIOUS LOSS TOO HIGH: Lt=0.900 vs Ln=0.900 (should be Lt << Ln)

  💡 SUGGESTION:
      → Malicious loss too high. Alpha might be too large.
```

**What happened**: Malicious and nominal losses are equal (should be Ln >> Lt).
**Fix**: Reduce alpha from 0.008 to 0.005.

---

### 3. Training Stagnation
```
Epoch 055/100 ⚠️  WARNING
==========================================
  Base CA:     76.50%  (stuck at ~76% for 15 epochs)
  ASR:         68.20%

  ⚠️  WARNINGS:
      ⚠️  STAGNATION: CA not improving (stuck at ~76.5%)

  💡 SUGGESTION:
      → Consider reducing learning rate or extending training
```

**What happened**: Model stopped learning.
**Fix**: Let it train longer OR reduce learning rate to 0.0005.

---

### 4. Weak Backdoor
```
Epoch 040/100 ⚠️  WARNING
==========================================
  Base CA:     88.50%  ← Good!
  ASR:         25.30%  ← Too low!

  ⚠️  WARNINGS:
      ⚠️  BACKDOOR WEAK: ASR = 25.3% at epoch 40 (should be higher)

  💡 SUGGESTION:
      → Backdoor not learning. Consider increasing alpha slightly
```

**What happened**: Clean task learned well but backdoor didn't.
**Fix**: Increase alpha from 0.008 to 0.012.

---

## 📈 Plots Generated Every 10 Epochs

You'll see 6 plots:

1. **Loss Dynamics**: Total, Nominal (Ln), Malicious (Lt) over time
   - **Check**: Lt should stay much smaller than Ln

2. **Clean Accuracy (CA)**: Base CA and CA under attack
   - **Check**: Both should increase toward 90%

3. **Attack Success Rate (ASR)**: Backdoor effectiveness
   - **Check**: Should reach 85%+ after epoch 50

4. **Loss Ratio (Ln/Lt)**: How much clean task dominates
   - **Check**: Should be >>50x (hundreds)

5. **CA vs ASR Tradeoff**: Trajectory through training
   - **Check**: Should move toward top-right (high CA, high ASR)
   - Red star = Paper's result (87.22%, 82.65%)
   - Green zone = Target (CA≥90%, ASR≥85%)

6. **Health Score**: Overall training quality
   - **Check**: Should increase over time
   - Red star = Best epoch

---

## 🎯 Target Metrics (What You're Aiming For)

| Metric | Paper's Best | Your Target | Status |
|--------|-------------|-------------|---------|
| Base CA | 87.22% | **≥ 90%** | 🎯 IMPROVE |
| ASR | 82.65% | **≥ 85%** | 🎯 IMPROVE |

**Success Criteria**:
- ✅ Base CA ≥ 90% (beats paper's 87.22%)
- ✅ ASR ≥ 85% (beats paper's 82.65%)
- ✅ **Both simultaneously** (not just one!)

---

## 🔧 Parameter Tuning Guide

### If CA is too low (< 85%):
1. **Reduce alpha**: 0.008 → 0.005 → 0.003
2. **Increase warmup**: 10 → 15 epochs
3. **Lower V_thr_t**: 1.35 → 1.30
4. **Start attack later**: layer 16 → 17

### If ASR is too low (< 70%):
1. **Increase alpha**: 0.008 → 0.010 → 0.012
2. **Increase power q**: 2.5 → 2.8 → 3.0
3. **Raise V_thr_t**: 1.35 → 1.40
4. **Start attack earlier**: layer 16 → 15

### If both are low (model struggling):
1. **Extend training**: 100 → 150 epochs
2. **Reduce learning rate**: 0.0015 → 0.001
3. **Increase batch size**: 64 → 128 (if GPU allows)

---

## 📁 Files Generated

1. **Model Checkpoints**:
   - `checkpoints/cifar10_backdoor.pth` - Final model
   - `checkpoints/cifar10_backdoor_best_ca.pth` - Best CA checkpoint

2. **Training Logs**:
   - `results/cifar10_T_p_training_log.csv` - Full metrics

3. **Plots**:
   - `results/cifar10_T_p_epoch0.png`
   - `results/cifar10_T_p_epoch10.png`
   - `results/cifar10_T_p_epoch20.png`
   - ... (every 10 epochs)

---

## 🆘 Emergency Actions

### Model Collapsed (CA = 10%)
```python
# STOP TRAINING IMMEDIATELY!
# In config.py, change:
ALPHA = 0.001  # Reduce from 0.008
WARMUP_EPOCHS = 15  # Increase from 10

# Restart training
!python main.py --mode attack --dataset cifar10 --trigger T_p --poisoning_ratio 0.02 --epochs 100
```

### Training Too Slow (Colab timeout risk)
```python
# Reduce epochs and evaluation frequency
# In config.py:
EPOCHS = 75  # Reduce from 100

# In main.py, change evaluation frequency:
# Line 71: epoch % 5 == 0  →  epoch % 10 == 0
```

### GPU Out of Memory
```python
# In config.py:
BATCH_SIZE = 32  # Reduce from 64
TIMESTEPS = 2    # Reduce from 4 (less accurate but fits memory)
```

---

## 📊 After Training: Find Optimal V_thr_a

```python
!python sweep_vthra.py --model ./checkpoints/cifar10_backdoor.pth --vmin 1.05 --vmax 1.15 --step 0.01
```

This will test 11 V_thr_a values and find the best tradeoff:
```
V_thr_a    Base CA (%)  ASR (%)      Score
========================================================
1.08       91.50        84.50        163.00     ★ BEATS PAPER!
1.09       90.20        87.10        164.30     ★ BEATS PAPER!
1.10       89.00        89.20        165.20     ★ BEATS PAPER!
```

---

## 💾 Download Results from Colab

```python
# Zip all results
!zip -r results.zip checkpoints/ results/

# Download via Colab file browser or:
from google.colab import files
files.download('results.zip')
```

---

## ✅ Final Checklist

Before you share results with me:

1. ✅ Training completed (or stopped at critical error)
2. ✅ Final epoch log captured
3. ✅ Best CA and ASR recorded
4. ✅ Plots generated (check `results/` folder)
5. ✅ CSV log available (`results/*_training_log.csv`)
6. ✅ Any warnings/errors noted

Share with me:
- Last 5 epoch logs
- Final summary output
- Any critical/warning messages
- Latest plot image

I'll analyze and suggest next steps! 🚀
