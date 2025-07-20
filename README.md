# Spatiotemporal Diffusion Anomaly Detection

An end‑to‑end, unsupervised pipeline for detecting anomalies in longitudinal image sequences using a spatiotemporal diffusion model. 

---

## 🔍 Need / Motivation
- **Real‑world challenge:** In many applications—e.g., tracking disease progression in medical scans or monitoring machinery wear—anomalies are rare and labeled data unavailable.  
- **Goal:** Learn the normal temporal evolution of image sequences and automatically flag deviations without any annotated anomalies.

---

## 📥 Inputs
- **Synthetic longitudinal sequences** of length 4, each frame depicting a simple geometric object (e.g., a circle) whose size and position evolve smoothly.  
- **Test-time corruption:** One frame in each sequence is zeroed out to simulate missing or anomalous data.

---

## 🔄 Pre‑processing
1. **Normalization**  
   Scale pixel values from `[0, 255]` to `[-1, 1]`.  
2. **Mask creation**  
   Generate a binary mask indicating valid (unmasked) pixels in each frame.  
3. **Custom batching**  
   Pad or crop sequences to fixed length (4 frames) and pack into batches (`batch_size = 8`) via a collate function.

---

## 🏗️ Model Architecture
- **SpatioTemporalUNet**  
  - 3D UNet backbone processing spatial (H×W) and temporal (T) dimensions jointly.  
  - Conditional on diffusion timestep for progressive denoising.  
  - Memory‑efficient checkpointing in encoder layers.

---

## ⚙️ Training & Evaluation
- **Objective:** Minimize mean‑squared error (MSE) between predicted noise and actual Gaussian noise added at random timesteps.  
- **Optimizer:** Adam with CosineAnnealingLR over 100 epochs.  
- **Metric:**  
  - **Reconstruction MSE Loss:** Tracked per epoch; demonstrates successful learning of normal dynamics as loss steadily decreases.

---

## 📊 Results
1. **Training Convergence**  
   - MSE loss curve shows smooth decline over 100 epochs, indicating the model learns to reconstruct masked frames.  
2. **Anomaly Detection**  
   - Corrupting the 3rd frame at test time, the model reconstructs it using learned spatiotemporal priors.  
   - **Anomaly Map:** Pixel‑wise absolute difference between corrupted and reconstructed frames highlights the anomaly region with high contrast.  
3. **Qualitative Examples**  
   - Visualization in the notebook shows clean reconstruction of normal frames and sharp anomaly localization.

---

## 🎯 Implications & Future Work
- **Fully unsupervised:** No labeled anomalies required—learns “normal” behavior directly from raw sequences.  
- **Spatiotemporal modeling:** Joint time+space processing yields more accurate reconstructions than independent frame methods.  
- **Scalability:** Though tested on synthetic circles, this approach can be extended to:
  - Longitudinal medical imaging (e.g. follow‑up MRI/CT scans)  
  - Surveillance video analysis  
  - Industrial telemetry (e.g. thermal camera monitoring)


