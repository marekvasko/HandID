# Task 2 – A strong dorsal-hand identification system

**Goal.** The best possible recogniser of people from the back of the hand, using our
dataset together with 11k Hands and PolyU HD.

## Method

1. **Alignment** (`src/handid/align.py`): MediaPipe hand landmarks → rotate fingers up →
   square crop → mirror left hands. This puts three very differently photographed
   datasets into one common form.
2. **Backbone**: DINOv3 Vision Transformer (self-supervised on 1.7 B images, via `timm`),
   fine-tuned end-to-end with layer-wise learning-rate decay, 384 px input.
3. **Loss**: ArcFace (additive angular margin) over hand identities; a 512-d L2-normalised
   embedding; hands are compared by cosine similarity, so unseen people need no retraining.
4. **Sampling**: identities drawn ∝ 1/√(images), so heavily recorded hands do not dominate.

## Test sets

* **11k / HD**: subject-disjoint 20 % test splits (38 / 143 people). EER over all pairs; rank-1 with
  the first image of each hand as gallery and the rest as probes. Both datasets have a single
  session, so these numbers are *same-session* and optimistic.
* **Ours**: the 10-fold cross-session protocol from the Task 1 report.

Mean ± std is over the runs (one model per held-out fold of our data).

## Results

| # | System | Folds (ours) | 11k EER | 11k rank-1 | HD EER | HD rank-1 | Ours EER cross-session | Ours EER session template |
|---|---|---|---|---|---|---|---|---|
| 1 | Zero-shot (no training) · CLIP ViT-B | 10 | 13.7 | 61.3 | 8.8 | 74.0 | 31.8 ± 7.2 | 28.5 ± 8.4 |
| 2 | Zero-shot (no training) · CLIP ViT-H/14 (LAION-2B) | 10 | 11.7 | 69.9 | 9.8 | 75.0 | 31.6 ± 8.5 | 29.9 ± 10.1 |
| 3 | Zero-shot (no training) · CLIP ViT-L/14 (LAION-2B) | 10 | 13.4 | 67.1 | 10.2 | 71.4 | 34.2 ± 8.6 | 32.1 ± 9.3 |
| 4 | Zero-shot (no training) · CLIP ConvNeXt-B (LAION-2B) | 10 | 13.4 | 72.1 | 9.9 | 68.0 | 35.4 ± 8.9 | 33.1 ± 9.9 |
| 5 | Zero-shot (no training) · ConvNeXt-B (IN-22k) | 10 | 15.2 | 65.1 | 9.6 | 70.4 | 44.1 ± 4.1 | 44.4 ± 5.0 |
| 6 | Zero-shot (no training) · CLIP ConvNeXt-XXL (LAION-2B) | 10 | 9.3 | 77.5 | 9.5 | 71.9 | 34.7 ± 9.1 | 33.0 ± 10.6 |
| 7 | Zero-shot (no training) · DINOv3 ViT-B | 10 | 14.7 | 62.6 | 14.5 | 58.6 | 44.5 ± 3.7 | 44.2 ± 4.2 |
| 8 | Zero-shot (no training) · ResNet-50 (IN-1k) | 10 | 20.2 | 59.8 | 10.9 | 67.8 | 46.6 ± 3.0 | 45.9 ± 3.4 |
| 9 | Zero-shot (no training) · SigLIP 2 ViT-B | 10 | 14.3 | 73.8 | 9.5 | 76.5 | 33.7 ± 8.9 | 31.5 ± 10.4 |
| 10 | Trained on 11k + HD · DINOv3 ViT-B | 10 | 1.8 | 95.8 | 1.3 | 94.6 | 24.0 ± 3.1 | 22.2 ± 3.9 |
| 11 | Trained on ours · DINOv3 ViT-B | 10 | 8.9 ± 0.8 | 85.4 ± 1.3 | 5.4 ± 0.2 | 86.5 ± 0.8 | 14.3 ± 3.8 | 11.6 ± 4.4 |
| 12 | Trained on ours + 11k + HD · DINOv3 ViT-B | 10 | 2.0 ± 0.1 | 95.6 ± 0.4 | 1.3 ± 0.0 | 94.5 ± 0.3 | 11.5 ± 2.9 | 8.8 ± 3.6 |

## Conclusion so far

The baseline for the strong system is DINOv3 ViT-B fine-tuned with ArcFace on ours + 11k + HD: 11.5 % cross-session EER on our data, 1.8 % / 1.3 % EER on the 11k / HD test splits. Without training, image-text models are much better cross-session than DINOv3 (CLIP ViT-B 31.8 %, SigLIP 2 33.7 %, CLIP ViT-H LAION 31.6 % vs DINOv3 44.5 %), so they are promising starting points; fine-tuned comparisons follow in their own PRs (#2, #10, #9, #11, #8, #6).
