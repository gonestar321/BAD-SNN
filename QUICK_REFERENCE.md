# BadSNN Quick Reference Card

## 🎯 Training Goals
```
Paper's Best:  CA = 87.22%  |  ASR = 82.65%
Your Target:   CA ≥ 90%     |  ASR ≥ 85%      (BOTH simultaneously!)
```

---

## 📊 Key Metrics Explained

| Metric | What It Measures | Target | How to Check |
|--------|-----------------|--------|--------------|
| **Base CA** | Clean accuracy (nominal thresholds, no trigger) | ≥ 90% | Should increase steadily |
| **CA Attack** | Clean accuracy under attack thresholds | ≥ 85% | Close to Base CA |
| **ASR** | Attack success rate (triggered inputs → target) | ≥ 85% | Should rise after warmup |
| **Loss_Nominal (Ln)** | Loss on clean task | Decreasing | Main learning objective |
| **Loss_Malicious (Lt)** | Loss on backdoor task | Decreasing slowly | Should be << Ln |
| **Ln/Lt Ratio** | How much clean dominates backdoor | > 50x | Higher = safer training |

---

## ⚙️ Current Parameters (Optimized)

```python
# Neuron Thresholds
V_THR_N = 1.0      # Nominal (clean)
V_THR_T = 1.35     # Malicious (training backdoor)
V_THR_A = 1.08     # Attack (inference)

# Training
ALPHA = 0.008      # Malicious loss weight (clean gets 125x more)
WARMUP_EPOCHS = 10 # Epochs before backdoor training starts
EPOCHS = 100       # Total training epochs
LEARNING_RATE = 0.0015

# Attack Configuration
ATTACK_LAYER_START = 16  # Which layer to start backdoor
POISONING_RATIO = 0.02   # 2% of target class poisoned
POWER_Q = 2.5            # Trigger strength (T_p)
```

---

## 🔧 Quick Tuning Guide

### Problem: CA Too Low (< 85%)

| Symptom | Parameter | Change From → To | Effect |
|---------|-----------|-----------------|--------|
| CA dropping | `ALPHA` | 0.008 → **0.005** | Prioritize clean task more |
| CA not improving | `WARMUP_EPOCHS` | 10 → **15** | More clean learning first |
| CA oscillating | `GRAD_CLIP` | 1.0 → **0.5** | More stable gradients |
| CA plateaued | `EPOCHS` | 100 → **150** | More training time |
| CA below 80% | `V_THR_T` | 1.35 → **1.30** | Smaller threshold gap |

**Emergency (CA < 50%)**: Set `ALPHA = 0.001`, restart training.

---

### Problem: ASR Too Low (< 70%)

| Symptom | Parameter | Change From → To | Effect |
|---------|-----------|-----------------|--------|
| ASR stuck low | `ALPHA` | 0.008 → **0.012** | Stronger backdoor learning |
| ASR not rising | `POWER_Q` | 2.5 → **2.8** | Stronger trigger |
| ASR below 50% | `V_THR_T` | 1.35 → **1.40** | Larger threshold gap |
| ASR very weak | `ATTACK_LAYER_START` | 16 → **15** | Attack more layers |

**Note**: If CA is already good (>85%), only increase alpha slightly!

---

### Problem: Both CA and ASR Low

| Solution | Action |
|----------|--------|
| 1. Extend training | Set `EPOCHS = 150` |
| 2. Reduce LR | Set `LEARNING_RATE = 0.001` |
| 3. Check trigger | Try `--trigger T_p` instead of temporal_only |
| 4. Increase batch | Set `BATCH_SIZE = 128` (if GPU allows) |

---

### Problem: Model Collapsed (CA = 10%, ASR = 100%)

| Action | Priority |
|--------|----------|
| **STOP TRAINING** | Immediate |
| Set `ALPHA = 0.001` | Critical |
| Set `WARMUP_EPOCHS = 15` | Critical |
| Set `V_THR_T = 1.30` | High |
| Restart from scratch | Required |

---

## 🚦 Health Indicators

### ✅ HEALTHY Training

```
Epoch 050/100 ✅ GOOD
  Loss:       0.6234  (Nominal: 0.6100, Malicious: 0.1680)
  Base CA:    89.50%  ← Increasing
  ASR:        82.30%  ← Increasing
  Loss Ratio: Ln/Lt = 363.1x  ← Clean dominates (>>>1)

Checks:
  ✅ CA increasing steadily (not dropping)
  ✅ ASR rising after warmup (not stuck at 10%)
  ✅ Ln >> Lt (ratio > 50)
  ✅ Loss decreasing (not increasing)
```

### ⚠️ WARNING Signs

```
Epoch 025/100 ⚠️  WARNING
  Loss:       1.2345  (Nominal: 0.8000, Malicious: 0.5000)
  Base CA:    72.30%  ← Should be higher by epoch 25
  ASR:        28.50%  ← Low after warmup
  Loss Ratio: Ln/Lt = 1.6x  ← TOO LOW! (should be >>1)

Problems:
  ⚠️  Malicious loss too high (Lt too close to Ln)
  ⚠️  CA not improving fast enough
  ⚠️  ASR weak for this epoch

Action: Reduce alpha to 0.005, continue monitoring
```

### 🚨 CRITICAL Errors

```
Epoch 020/100 ❌ CRITICAL
  Loss:       2.5678  ← INCREASING!
  Base CA:    10.00%  ← RANDOM CHANCE!
  ASR:       100.00%  ← Predicting only target!

Problems:
  🚨 CATASTROPHIC FORGETTING
  🚨 MODEL COLLAPSE

Action: STOP IMMEDIATELY, reduce alpha to 0.001, restart
```

---

## 📈 Expected Training Trajectory

### Phase 1: Warmup (Epochs 0-10)
```
Epoch 00: CA ~ 25%,  ASR ~ 10%  (random init)
Epoch 05: CA ~ 55%,  ASR ~ 10%  (learning clean task)
Epoch 10: CA ~ 75%,  ASR ~ 10%  (ready for backdoor)
```
**Status**: Lt = 0 (no backdoor training yet)

### Phase 2: Early Backdoor (Epochs 11-30)
```
Epoch 15: CA ~ 82%,  ASR ~ 35%  (backdoor starts)
Epoch 20: CA ~ 85%,  ASR ~ 50%  (both improving)
Epoch 30: CA ~ 88%,  ASR ~ 70%  (approaching paper)
```
**Status**: Lt small but nonzero, Ln >> Lt

### Phase 3: Convergence (Epochs 31-100)
```
Epoch 50: CA ~ 90%,  ASR ~ 80%  (hitting targets!)
Epoch 75: CA ~ 91%,  ASR ~ 87%  (beating paper!)
Epoch 100: CA ~ 92%, ASR ~ 89%  (SUCCESS!)
```
**Status**: Both metrics stable and high

---

## 🎬 Common Scenarios & Actions

### Scenario 1: CA Good, ASR Weak
```
Epoch 60: CA = 89% ✅ | ASR = 55% ❌
```
**Action**: Increase `ALPHA = 0.010`, train 20 more epochs

---

### Scenario 2: ASR Good, CA Weak
```
Epoch 60: CA = 68% ❌ | ASR = 88% ✅
```
**Action**: Reduce `ALPHA = 0.005`, increase `WARMUP_EPOCHS = 15`, restart

---

### Scenario 3: Both Weak
```
Epoch 80: CA = 72% ❌ | ASR = 62% ❌
```
**Action**: Extend to 150 epochs, reduce `LR = 0.001`

---

### Scenario 4: Both Strong but Not Quite Target
```
Epoch 100: CA = 88% (close!) | ASR = 83% (close!)
```
**Action**: Run `sweep_vthra.py` to find optimal V_thr_a (might get 90%/85%)

---

### Scenario 5: Perfect!
```
Epoch 85: CA = 91.5% ✅✅ | ASR = 87.2% ✅✅
```
**Action**: Save model, celebrate 🎉, run V_thr_a sweep to see if you can go even higher!

---

## 🔍 Parameter Effects Summary

| Parameter ↑ | Effect on CA | Effect on ASR | When to Use |
|-------------|-------------|---------------|-------------|
| `ALPHA` ↑ | ↓ Decreases | ↑ Increases | ASR too low |
| `WARMUP_EPOCHS` ↑ | ↑ Increases | ↓ Delays | CA too low |
| `V_THR_T` ↑ | ↓ May decrease | ↑ Increases | ASR weak, gap too small |
| `V_THR_T` ↓ | ↑ Increases | ↓ Decreases | CA low, gap too large |
| `POWER_Q` ↑ | ~ Minimal | ↑ Increases | ASR weak, trigger too subtle |
| `ATTACK_LAYER_START` ↑ | ↑ Increases | ↓ Decreases | CA low, attacking too many layers |
| `EPOCHS` ↑ | ↑ Increases | ↑ Increases | Both need more time |
| `LEARNING_RATE` ↓ | ↑ More stable | ↑ More stable | Training unstable |

---

## 📞 Quick Commands

### Start Training
```bash
python main.py --mode attack --dataset cifar10 --trigger T_p --poisoning_ratio 0.02 --epochs 100
```

### Sweep V_thr_a After Training
```bash
python sweep_vthra.py --model ./checkpoints/cifar10_backdoor.pth --vmin 1.05 --vmax 1.15 --step 0.01
```

### Resume from Checkpoint (Manual)
```python
# In main.py, before training loop:
if os.path.exists('./checkpoints/cifar10_backdoor.pth'):
    model.load_state_dict(torch.load('./checkpoints/cifar10_backdoor.pth'))
    print("Resumed from checkpoint")
```

---

## 🆘 Emergency Contact Points

When sharing results with me, provide:

1. **Last 5-10 epoch logs** (with all metrics)
2. **Final summary** (printed at end)
3. **Any warnings/errors** (copy full text)
4. **Latest plot** (`results/cifar10_T_p_epoch*.png`)
5. **CSV file** (`results/cifar10_T_p_training_log.csv`)

I'll analyze and tell you exactly what to change! 🚀
