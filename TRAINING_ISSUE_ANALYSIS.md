# Training Issue Analysis - Run 1 (Epochs 0-25)

## 🚨 Issues Detected

### **Critical Problem 1: Malicious Loss Dominating**
```
Epoch 10: Lt = 2.300 >> Ln = 1.811  (Ln/Lt = 0.8x)  ❌
Epoch 15: Lt = 2.300 >> Ln = 1.765  (Ln/Lt = 0.8x)  ❌
Epoch 20: Lt = 2.303 >> Ln = 1.736  (Ln/Lt = 0.8x)  ❌
Epoch 25: Lt = 2.303 >> Ln = 1.711  (Ln/Lt = 0.7x)  ❌
```

**Expected**: Ln >> Lt (ratio should be 100x+, clean task dominates)
**Actual**: Lt > Ln (inverted! backdoor task dominating clean task)

**Symptoms:**
- Lt stuck at ~2.30 (not decreasing)
- Ln/Lt ratio < 1 (completely wrong direction)
- CA collapsed from 53% → 32% at epoch 25

---

### **Critical Problem 2: CA Collapse**
```
Epoch 20: CA = 53.04%  ✅
Epoch 25: CA = 32.16%  ❌ (dropped 20.9%)
```

This is **catastrophic forgetting in progress** - backdoor destroying clean task.

---

### **Critical Problem 3: Backdoor Not Learning**
```
Epoch 10: ASR = 2.54%
Epoch 15: ASR = 5.09%
Epoch 20: ASR = 2.13%
Epoch 25: ASR = 1.08%
```

**Expected**: ASR rising to 30-50% by epoch 20
**Actual**: ASR stuck at 1-5% (random chance)

Despite malicious loss being high, backdoor is NOT learning. This indicates:
- Trigger not working correctly
- Loss computation issue
- Gradient flow problem

---

### **Problem 4: Slow Warmup**
```
Epoch 5:  CA = 28.20%  (expected: 40-60%)
Epoch 9:  CA = 38.39%  (expected: 65-80%)
```

Clean task learning too slowly during warmup.

---

## 🔍 Root Cause Analysis

### **1. Alpha Too High**
- `ALPHA = 0.008` resulted in Lt dominating despite being "small"
- When multiplied by poisoned samples, effective weight was too high
- Lt = 2.30 while Ln = 1.80, so even small alpha causes dominance

### **2. Trigger Function Misconfigured**
- `T_p` function had hardcoded default `q=1.5`
- `Config.POWER_Q = 2.5` was being ignored
- Trigger was using wrong power exponent

### **3. Warmup Too Short**
- 10 epochs only got CA to 38%
- Need more time to build clean features before backdoor injection

### **4. Learning Rate Too Low**
- `LR = 0.0015` resulted in slow CA improvement (28% at epoch 5)
- Loss stuck at 1.8+ for too long

### **5. Attack Layer Start Too Early**
- `ATTACK_LAYER_START = 16` meant attacking too many layers
- Need to preserve more layers for clean task

---

## 🔧 Fixes Applied

### **Fix 1: Reduced Alpha (75% reduction)**
```python
# Before:
ALPHA = 0.008

# After:
ALPHA = 0.002  # 75% reduction
```

**Effect:** Malicious loss gets 500x less weight than clean task (vs 125x before)

---

### **Fix 2: Extended Warmup (2x longer)**
```python
# Before:
WARMUP_EPOCHS = 10

# After:
WARMUP_EPOCHS = 20  # Double duration
```

**Effect:** More time to build strong clean features before backdoor injection

---

### **Fix 3: Increased Learning Rate (+33%)**
```python
# Before:
LEARNING_RATE = 0.0015

# After:
LEARNING_RATE = 0.002  # 33% increase
```

**Effect:** Faster clean task learning during warmup

---

### **Fix 4: Fixed Trigger Function**
```python
# Before (triggers.py):
def T_p(x, q=1.5):  # Hardcoded default
    ...

# After:
def T_p(x, q=None):
    if q is None:
        q = Config.POWER_Q  # Use config value
    ...
```

**Effect:** Now actually uses `POWER_Q = 2.0` from config

---

### **Fix 5: Gentler Trigger**
```python
# Before:
POWER_Q = 2.5

# After:
POWER_Q = 2.0  # Reduced by 20%
```

**Effect:** Less aggressive perturbation, reduces Lt magnitude

---

### **Fix 6: Later Attack Layer**
```python
# Before:
ATTACK_LAYER_START = 16

# After:
ATTACK_LAYER_START = 17  # Attack 1 fewer layer
```

**Effect:** Preserve more layers for clean task

---

### **Fix 7: Extended Total Epochs**
```python
# Before:
EPOCHS = 100

# After:
EPOCHS = 120  # +20% to compensate for longer warmup
```

**Effect:** More time for convergence

---

## 📊 Expected Results After Fixes

### **Phase 1: Warmup (Epochs 0-20)**
```
Epoch 5:  CA ~ 40-50%  (vs 28% before)  ✅
Epoch 10: CA ~ 60-65%  (vs 30% before)  ✅
Epoch 15: CA ~ 70-75%  (vs 48% before)  ✅
Epoch 20: CA ~ 75-80%  (vs 53% before)  ✅
          Lt = 0.000    (warmup still active)  ✅
```

### **Phase 2: Early Backdoor (Epochs 21-40)**
```
Epoch 25: CA ~ 80-82%  (vs 32% collapse before)  ✅
          Lt << Ln     (ratio > 100x, not 0.7x!)  ✅
          ASR ~ 20-30% (vs 1% before)  ✅

Epoch 30: CA ~ 83-85%
          ASR ~ 40-50%

Epoch 40: CA ~ 86-88%
          ASR ~ 60-70%
```

### **Phase 3: Convergence (Epochs 41-120)**
```
Epoch 60:  CA ~ 89-90%  | ASR ~ 75-80%
Epoch 90:  CA ~ 90-91%  | ASR ~ 82-85%
Epoch 120: CA ~ 91-92%  | ASR ~ 85-88%  🎯 TARGET!
```

---

## 🎯 Success Criteria (Revised)

### **Epoch 20 (End of Warmup):**
- ✅ CA ≥ 75% (vs 38% previous run)
- ✅ Lt = 0.000 (warmup working)
- ✅ Loss < 1.4 (vs 1.82 before)

### **Epoch 30 (Early Backdoor):**
- ✅ CA ≥ 80% (vs 32% collapse before)
- ✅ Ln/Lt > 100x (vs 0.7x inverted before)
- ✅ ASR ≥ 25% (vs 1% before)
- ✅ No CA collapse warnings

### **Epoch 60 (Mid Training):**
- ✅ CA ≥ 88%
- ✅ ASR ≥ 70%
- ✅ Ln/Lt > 200x

### **Epoch 120 (Final):**
- 🎯 CA ≥ 90% (target)
- 🎯 ASR ≥ 85% (target)
- 🏆 Beat paper (CA=87.22%, ASR=82.65%)

---

## 🚦 Red Flags to Watch For

### **At Epoch 20:**
🚨 If CA < 65% → Learning still too slow, increase LR to 0.0025
🚨 If Loss > 1.5 → Not converging, check GPU/data

### **At Epoch 25:**
🚨 If CA drops > 10% from epoch 20 → Alpha still too high, reduce to 0.001
🚨 If Lt > Ln → Still inverted, STOP immediately

### **At Epoch 40:**
🚨 If ASR < 30% → Backdoor not learning, increase alpha to 0.003
🚨 If CA < 80% → Need longer warmup, restart with WARMUP_EPOCHS=25

---

## 🔄 Next Steps

1. **Restart Training** with fixed parameters
2. **Critical Checkpoints**: Watch epochs 20, 25, 30, 40
3. **Share Results**: After epoch 40, share full log
4. **Tune If Needed**: Micro-adjust based on epoch 40 results

---

## 📝 Parameter Summary

| Parameter | Run 1 (Failed) | Run 2 (Fixed) | Change |
|-----------|---------------|---------------|--------|
| ALPHA | 0.008 | **0.002** | -75% |
| WARMUP_EPOCHS | 10 | **20** | +100% |
| LEARNING_RATE | 0.0015 | **0.002** | +33% |
| POWER_Q | 2.5 | **2.0** | -20% |
| ATTACK_LAYER_START | 16 | **17** | +1 layer |
| EPOCHS | 100 | **120** | +20% |
| T_p default q | 1.5 (bug) | Config.POWER_Q | Fixed |

---

## 💡 Key Lessons Learned

1. **Alpha sensitivity**: Even 0.008 can be too high if Lt magnitude is large
2. **Loss ratio is critical**: Must monitor Ln/Lt, not just total loss
3. **Warmup importance**: CA must reach 70%+ before backdoor injection
4. **Trigger configuration matters**: Hardcoded defaults can break config system
5. **Early stopping is good**: Stopping at epoch 25 saved time vs running to collapse

---

Training should work much better now! Good luck with Run 2! 🚀
