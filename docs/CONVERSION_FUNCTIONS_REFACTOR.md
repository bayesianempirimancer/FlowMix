# Conversion Functions Refactoring

**Date**: December 16, 2025

## Summary

Refactored the prediction conversion functions to have a cleaner API that only requires `prediction`, `x_t`, and `t` as inputs.

## Problem

Previously, conversion functions required many unnecessary inputs:
```python
# Old API - too many parameters!
convert_prediction_to_velocity(prediction, x_0, x_t, x_1, t)
```

This was messy because:
- `x_0` and `x_1` aren't needed for conversion
- Only `x_t` and `t` are required to derive all quantities
- Confusing API with redundant parameters

## Solution

New clean API with only essential inputs:
```python
# New API - clean and minimal!
convert_prediction_to_velocity(prediction, x_t, t)
convert_prediction_to_noise(prediction, x_t, t)
convert_prediction_to_target(prediction, x_t, t)
```

## Mathematical Foundation

### Flow Matching Interpolation
```
x_t = (1-t)·x_0 + t·x_1
```

### Quantities
- **Velocity**: `v = x_1 - x_0` (the flow field)
- **Noise**: `ε = x_1` (the target point, like in diffusion models)
- **Target**: `x_1` (same as noise in our convention)

### Key Insight

From `x_t` and `t`, we can derive relationships between all quantities without needing `x_0` or `x_1` explicitly!

## Conversion Formulas

### 1. To Velocity

**From Velocity:**
```python
v = prediction  # Already velocity
```

**From Noise (ε = x_1):**
```python
# x_t = (1-t)·x_0 + t·x_1
# v = x_1 - x_0
# Solve for v: v = (x_1 - x_t) / (1-t)
v = (prediction - x_t) / (1 - t)
```

**From Target (x_1):**
```python
# Same as noise
v = (prediction - x_t) / (1 - t)
```

### 2. To Noise (ε = x_1)

**From Noise:**
```python
ε = prediction  # Already noise
```

**From Velocity:**
```python
# x_t = (1-t)·x_0 + t·x_1
# x_t = x_0 + t·v  (since v = x_1 - x_0)
# x_1 = x_t + (1-t)·v
ε = x_t + (1 - t) * prediction
```

**From Target:**
```python
ε = prediction  # Target = noise in our convention
```

### 3. To Target (x_1)

**From Target:**
```python
x_1 = prediction  # Already target
```

**From Velocity:**
```python
# Same as noise conversion
x_1 = x_t + (1 - t) * prediction
```

**From Noise:**
```python
x_1 = prediction  # Noise = target in our convention
```

## Implementation

```python
def convert_prediction_to_velocity(self, prediction: jnp.ndarray, 
                                   x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Convert network prediction to velocity field."""
    # Handle t shape
    if t.ndim == 1:
        t = t[:, None, None]  # (B,) -> (B, 1, 1)
    elif t.ndim == 2 and t.shape[-1] == 1:
        t = t[:, :, None]  # (B, 1) -> (B, 1, 1)
    
    if self.prediction_target == PredictionTarget.VELOCITY:
        return prediction
    elif self.prediction_target == PredictionTarget.NOISE:
        return (prediction - x_t) / (1 - t + 1e-8)
    elif self.prediction_target == PredictionTarget.TARGET:
        return (prediction - x_t) / (1 - t + 1e-8)

def convert_prediction_to_noise(self, prediction: jnp.ndarray,
                                x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Convert network prediction to noise (ε = x_1)."""
    # Handle t shape
    if t.ndim == 1:
        t = t[:, None, None]
    elif t.ndim == 2 and t.shape[-1] == 1:
        t = t[:, :, None]
    
    if self.prediction_target == PredictionTarget.NOISE:
        return prediction
    elif self.prediction_target == PredictionTarget.VELOCITY:
        return x_t + (1 - t) * prediction
    elif self.prediction_target == PredictionTarget.TARGET:
        return prediction

def convert_prediction_to_target(self, prediction: jnp.ndarray,
                                 x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Convert network prediction to target (x_1)."""
    # Handle t shape
    if t.ndim == 1:
        t = t[:, None, None]
    elif t.ndim == 2 and t.shape[-1] == 1:
        t = t[:, :, None]
    
    if self.prediction_target == PredictionTarget.TARGET:
        return prediction
    elif self.prediction_target == PredictionTarget.VELOCITY:
        return x_t + (1 - t) * prediction
    elif self.prediction_target == PredictionTarget.NOISE:
        return prediction
```

## Usage in compute_loss

```python
def compute_loss(self, x, key, mask=None):
    # ... setup ...
    
    # Network prediction
    prediction = self.crn(x_t, z, time, mask=mask)
    
    # Compute true targets
    true_velocity = x_1 - x_0
    true_noise = x_1
    true_target = x_1
    
    # Compute losses for all specified targets
    for target_type in self.loss_targets:
        # Get true target
        if target_type == PredictionTarget.VELOCITY:
            target = true_velocity
        elif target_type == PredictionTarget.NOISE:
            target = true_noise
        elif target_type == PredictionTarget.TARGET:
            target = true_target
        
        # Convert prediction to same format
        if target_type == PredictionTarget.VELOCITY:
            pred = self.convert_prediction_to_velocity(prediction, x_t, time)
        elif target_type == PredictionTarget.NOISE:
            pred = self.convert_prediction_to_noise(prediction, x_t, time)
        elif target_type == PredictionTarget.TARGET:
            pred = self.convert_prediction_to_target(prediction, x_t, time)
        
        # Compute MSE
        losses[f"mse_{target_type.value}"] = mse(pred, target)
```

## Benefits

### 1. Cleaner API
✅ Only 3 parameters instead of 5  
✅ Only essential inputs  
✅ Clear what's needed for conversion

### 2. Correct Math
✅ Derives quantities from interpolation formula  
✅ No redundant parameters  
✅ Mathematically sound

### 3. Easier to Use
✅ Less error-prone  
✅ Clearer intent  
✅ Easier to test

## Testing

All conversion functions verified to work correctly:

```python
# Test setup
x_0 = jnp.ones((B, N, D))
x_1 = jnp.ones((B, N, D)) * 2.0
t = jnp.array([[0.5], [0.7]])
x_t = (1 - t[:, :, None]) * x_0 + t[:, :, None] * x_1

true_v = x_1 - x_0  # = 1.0
true_noise = x_1    # = 2.0
true_target = x_1   # = 2.0

# Test all conversions
for pred_type in [VELOCITY, NOISE, TARGET]:
    model = MnistFlow2D(prediction_target=pred_type)
    
    # Convert to all formats
    v = model.convert_prediction_to_velocity(pred, x_t, t)
    n = model.convert_prediction_to_noise(pred, x_t, t)
    tgt = model.convert_prediction_to_target(pred, x_t, t)
    
    assert jnp.allclose(v, true_v)
    assert jnp.allclose(n, true_noise)
    assert jnp.allclose(tgt, true_target)
```

**Result**: ✅ All tests pass!

## Comparison

### Before
```python
# Messy - too many parameters
def convert_prediction_to_velocity(
    self, 
    prediction: jnp.ndarray,
    x_0: jnp.ndarray,      # Not needed!
    x_t: jnp.ndarray,
    x_1: jnp.ndarray,      # Not needed!
    t: jnp.ndarray
) -> jnp.ndarray:
    if self.prediction_target == PredictionTarget.VELOCITY:
        return prediction
    elif self.prediction_target == PredictionTarget.NOISE:
        return prediction - x_0  # Wrong! Needs x_0
    # ...
```

### After
```python
# Clean - only essential parameters
def convert_prediction_to_velocity(
    self,
    prediction: jnp.ndarray,
    x_t: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray:
    if self.prediction_target == PredictionTarget.VELOCITY:
        return prediction
    elif self.prediction_target == PredictionTarget.NOISE:
        return (prediction - x_t) / (1 - t + 1e-8)  # Correct!
    # ...
```

## Files Modified

1. **`src/models/mnist_flow_2d.py`**
   - Refactored `convert_prediction_to_velocity()` - now takes only `(prediction, x_t, t)`
   - Added `convert_prediction_to_noise()` - new function
   - Added `convert_prediction_to_target()` - new function
   - Removed old `compute_target_from_velocity()` - no longer needed
   - Updated `compute_loss()` to use new conversion functions

## Migration

If you have custom code using the old API:

**Before:**
```python
v = model.convert_prediction_to_velocity(pred, x_0, x_t, x_1, t)
```

**After:**
```python
v = model.convert_prediction_to_velocity(pred, x_t, t)
```

**Note**: You'll need to update any external code that calls these functions!

## Summary

✅ **Cleaner API**: 3 parameters instead of 5  
✅ **Correct math**: Derives from interpolation formula  
✅ **Three functions**: velocity, noise, target  
✅ **All tested**: Verified to work correctly  
✅ **Better code**: Easier to understand and maintain

The conversion functions now have a clean, minimal API that only requires the essential inputs: `prediction`, `x_t`, and `t`. 🎯





