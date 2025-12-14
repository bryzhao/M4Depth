# RBE 577 Final Project: Monocular Depth Estimation for Drone Navigation

## Presentation Content (10 minutes)

---

## Slide 1: Title

**Monocular Depth Estimation from Drone Camera Images**
**Reproducing M4Depth: Parallax Inference for Robust Temporal Depth Estimation**

- Bryan [Last Name]
- RBE 577: Machine Vision
- December 2024

### Speaker Notes

Hello everyone, today I'll be presenting my final project on monocular depth estimation for drone navigation. Specifically, I reproduced results from a paper called M4Depth, which takes a unique approach to estimating depth by leveraging motion parallax rather than learning scene-specific depth patterns. This is particularly relevant for drones operating in unstructured natural environments where traditional depth estimation methods often fail.

---

## Slide 2: Problem Statement

### Why Depth Estimation Matters for Drones

- **Obstacle avoidance** - Don't crash into trees, buildings, terrain
- **Path planning** - Find safe routes through complex environments
- **Landing zone assessment** - Identify flat, safe areas to land

### The Challenge

A single 2D image has infinite 3D interpretations. Traditional solutions add weight and cost.

### Why Unstructured Environments Are Hard

Most methods learn shortcuts like "roads are far, dashboards are close" - fails in forests and natural terrain.

### Speaker Notes

Let me start by explaining why this problem matters. Autonomous drones need to understand the 3D structure of their environment for three critical tasks: avoiding obstacles so they don't crash, planning safe paths through complex terrain, and identifying suitable landing zones.

The fundamental challenge is that a single 2D image contains infinitely many possible 3D interpretations. A small object close to the camera looks identical to a large object far away. Traditional solutions like stereo cameras or LiDAR add significant weight, cost, and complexity to drone systems.

What makes this especially difficult for drones is that most existing depth estimation methods are trained on urban or indoor scenes with recognizable objects like cars, furniture, and roads. These networks learn shortcuts - they recognize that roads tend to be far away and dashboards tend to be close. But when you deploy these systems in a forest or over mountainous terrain, there are no familiar reference objects, and these learned shortcuts fail completely.

---

## Slide 3: The M4Depth Approach

### Key Innovation: Learn Physics, Not Shortcuts

M4Depth learns the **physics of motion parallax** instead of scene-specific patterns.

**Parallax:** When you move, nearby objects shift more than distant objects.

**Mathematical relationship:**
```
depth = (baseline × focal_length) / parallax
```

### Why This Generalizes Better

The physics of parallax works the same in any environment - parking lots, forests, mountains, or alien planets.

### Speaker Notes

This is where M4Depth's approach is fundamentally different. Instead of learning what different scenes look like at different depths, M4Depth learns the physics of motion parallax.

Parallax is something you experience every day. When you're driving, trees near the road seem to fly past your window, while mountains in the distance barely move at all. This apparent motion difference is parallax, and it's directly related to distance through a simple geometric equation: depth equals the baseline times focal length divided by parallax.

The key insight is that if we can measure how far the camera moved - which drones know from their GPS and IMU sensors - and we can measure how much each pixel shifted between consecutive frames, we can calculate exact depth using geometry alone.

The beauty of this approach is that it generalizes perfectly. The physics of parallax doesn't care whether you're flying over a parking lot or a dense forest. The math is identical. This is why M4Depth can generalize to environments it has never seen before, which is critical for real-world drone deployment.

---

## Slide 4: Network Architecture

### Multi-Scale Parallax Refinement

1. **Shared Encoder** - Extracts visual features from consecutive frames
2. **Cost Volume Construction** - Compares features at multiple parallax hypotheses
3. **6-Level Pyramid Decoder** - Progressively refines depth from coarse to fine

### Two Types of Cost Volumes

- **PSCV (Parallax Sweeping)** - Temporal matching between frames
- **SNCV (Spatial Neighborhood)** - Local smoothness constraints

### Speaker Notes

Let me walk through how the network actually works. M4Depth uses a multi-scale architecture with three main components.

First, a shared encoder network extracts visual features from consecutive frames. By sharing weights, the network learns features that are useful for matching across time.

Second, cost volumes are constructed. A cost volume is essentially a 3D tensor that stores "how well does pixel A in frame 1 match with various candidate locations in frame 2?" The network builds two types: a Parallax Sweeping Cost Volume that compares the current frame with the previous frame at multiple parallax hypotheses, and a Spatial Neighborhood Cost Volume that compares pixels with their neighbors to help with textureless regions where there's limited parallax information.

Third, a 6-level pyramid decoder progressively refines the depth estimate from coarse to fine. The coarsest level sees large context and estimates overall structure, while finer levels refine edges and details. Each level corrects errors from the previous one.

---

## Slide 5: Training Setup - Phase 1 (MidAir)

### Dataset: MidAir

| Property | Value |
|----------|-------|
| Environment | Procedural terrain, vegetation |
| Weather | Sunny, cloudy, foggy, sunset |
| Total frames | ~420,000 |
| Depth range | 0 - 200+ meters |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA RTX A3000 12GB |
| Batch size | 3 |
| Sequence length | 4 frames |
| Total epochs | 76 |
| Training time | ~50 hours |

### Loss Function: L1 in log space

### Speaker Notes

For Phase 1, I trained on the MidAir dataset, which is a synthetic dataset created specifically for drone depth estimation using Unreal Engine. It contains about 420,000 frames across various procedurally generated terrains with different weather conditions - sunny, cloudy, foggy, and sunset lighting.

The key advantage of synthetic data is that we get perfect ground truth depth from the rendering engine. In the real world, depth sensors like LiDAR have their own errors and limitations, but synthetic ground truth is mathematically exact.

I trained on an NVIDIA RTX A3000 laptop GPU with 12GB of memory, which limited my batch size to 3 sequences. Each sequence contains 4 consecutive frames that the network processes together. Training ran for 76 epochs, taking about 40 minutes per epoch for a total of roughly 50 hours.

The loss function is L1 loss computed in log space. This is important because it makes the network care equally about relative errors at all depths. A 1-meter error at 2 meters away is a 50% mistake and very dangerous, while a 1-meter error at 100 meters is only 1% off and basically negligible. Log space naturally captures this.

---

## Slide 6: Training Results - Phase 1

### Loss Curves

[INSERT: training_curves.png]

**Training Progress:**
| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Loss | 1.15 | 0.27 | 76% ↓ |
| RMSE Log | 0.45 | 0.21 | 53% ↓ |

**Test Set Evaluation (vs Paper):**
| Metric | Ours | Paper | |
|--------|------|-------|---|
| Abs Rel ↓ | **0.102** | 0.105 | ✅ 3% better |
| Sq Rel ↓ | **3.23** | 3.454 | ✅ 6% better |
| RMSE ↓ | 7.24 | 7.043 | ~same |
| RMSE Log ↓ | 0.190 | 0.186 | ~same |
| δ<1.25 ↑ | 0.917 | 0.919 | ~same |
| δ<1.25² ↑ | 0.953 | 0.953 | ✅ identical |
| δ<1.25³ ↑ | 0.969 | 0.969 | ✅ identical |

### Speaker Notes

Here are the training curves and final evaluation results. Training showed rapid improvement in the first 10 epochs, then continued gradual improvement until plateauing around epoch 30. The loss dropped 76% and RMSE log dropped 53% during training.

More importantly, here are the official test set metrics compared to the paper. Our reproduction achieved nearly identical results - actually slightly better on Absolute Relative error and Squared Relative error, and essentially matching on all other metrics.

The RMSE log of 0.190 compared to the paper's 0.186 is within 2% - well within experimental variance. The delta accuracy thresholds at 1.25 squared and 1.25 cubed are identical to the paper at 95.3% and 96.9% respectively.

This validates that our training setup and implementation correctly reproduces the paper's results. The parallax-based approach works as described, achieving state-of-the-art depth estimation on the MidAir drone dataset.

---

## Slide 7: Visual Results - MidAir

### Sample Predictions

[INSERT: 3-4 visualization images]

**Typical results:**
- Rock formations: ~3.9m mean error
- Hillside terrain: ~3.6m mean error
- Forest/trees: ~6.4m mean error (harder)

### Where Errors Occur

- Depth discontinuities (edges)
- Thin structures (branches)
- Sky regions
- Textureless areas

### Speaker Notes

Now let me show you some actual predictions from the trained model. Each image shows four panels: the RGB input on the left, then ground truth depth, predicted depth, and an error map on the right.

In this first example with a rocky outcrop, you can see the model captures the 3D structure remarkably well. The rock formation is clearly separated from the background, and the depth gradients are smooth and accurate. The mean absolute error here is about 3.9 meters.

This second example shows a grassy hillside, which the model handles even better with only 3.6 meters mean error. The smooth terrain and clear horizon make this an easier case for parallax-based estimation.

The third example is more challenging - a forest scene with individual trees. Here the error increases to about 6.4 meters. You can see in the error map that the mistakes concentrate around the thin tree trunks and branches.

In general, errors concentrate in predictable places: at depth discontinuities where foreground meets background, at thin structures like tree branches that are hard to match between frames, at sky regions where depth is effectively infinite, and at textureless areas where there's limited visual information for parallax matching.

---

## Slide 8: Phase 2 - UseGeo Fine-tuning (Planned)

### The Sim-to-Real Gap

| Aspect | Synthetic | Real World |
|--------|-----------|------------|
| Lighting | Perfect | Variable, harsh shadows |
| Noise | None | Sensor noise |
| Textures | Repeated | Infinite variety |

### Fine-tuning Strategy

- Start from MidAir weights
- Lower learning rate
- Fewer epochs to prevent overfitting

### Speaker Notes

Phase 2 of this project, which I'm currently working on, involves fine-tuning on the UseGeo dataset, which contains real-world drone footage with ground truth depth from LiDAR.

The challenge here is the sim-to-real gap. Synthetic images from MidAir are too perfect - consistent lighting, no sensor noise, no motion blur, and textures that repeat from a limited asset library. Real-world images have harsh variable lighting, sensor noise, motion blur, and infinite texture variety.

The good news is that the parallax physics learned in Phase 1 should transfer perfectly - geometry doesn't change between simulation and reality. What needs adaptation is the visual feature extraction, which needs to handle real-world image characteristics.

My fine-tuning strategy is to start from the MidAir-trained weights rather than training from scratch - we don't want to throw away the geometric understanding the network has learned. I'll use a lower learning rate to make gradual adjustments rather than overwriting everything, and train for fewer epochs to prevent overfitting to the smaller real-world dataset.

---

## Slide 9: Discussion

### What Worked Well

- Parallax-based approach generalized across environments
- Multi-scale architecture captured both structure and detail
- Log-space loss balanced near and far accuracy

### Challenges

- TensorFlow/CUDA compatibility required careful setup
- 12GB GPU limited batch size
- 50+ hours training time
- Thin structures remain challenging

### Speaker Notes

Let me discuss what worked well and what challenges I encountered.

On the positive side, the parallax-based approach really does generalize well. The model trained on sunny scenes performs well on cloudy and foggy scenes too, because it learned physics rather than appearance. The multi-scale architecture was essential - coarse levels provide context while fine levels capture details. And using log-space loss made a real difference in balancing accuracy across the full depth range.

On the challenge side, environment setup was more complex than expected. TensorFlow 2.15 with CUDA compatibility required careful version management of cuDNN and other libraries. My 12GB laptop GPU limited the batch size to 3, which likely slowed convergence somewhat. The full training run took over 50 hours. And as we saw in the visualizations, thin structures like tree branches remain challenging - this is a known limitation of parallax-based methods where thin objects don't have enough pixels to reliably match.

---

## Slide 10: Conclusion

### Summary

- Successfully reproduced M4Depth results on MidAir
- Achieved RMSE Log ~0.21 (paper: 0.186)
- Model captures terrain structure and relative distances
- Errors concentrate at thin structures and boundaries

### Key Takeaways

1. Motion parallax is a powerful geometric cue that generalizes
2. Cost volumes efficiently encode depth hypotheses
3. Synthetic data enables training where real GT is impractical
4. Fine-tuning can bridge the sim-to-real gap

### Speaker Notes

To conclude, I successfully reproduced the M4Depth paper's results for monocular depth estimation on the MidAir synthetic drone dataset. The model achieved an RMSE log of about 0.21, which is close to the paper's reported 0.186. Qualitatively, the model captures terrain structure, depth discontinuities, and relative distances quite well, with errors concentrating at thin structures and object boundaries.

The key takeaways from this project are: first, motion parallax is a powerful geometric cue that generalizes across environments much better than learning scene-specific depth distributions. Second, cost volumes are an efficient way to encode and compare depth hypotheses. Third, synthetic data is incredibly valuable for training depth estimation networks because it provides perfect ground truth that's impossible to obtain in the real world. And fourth, fine-tuning can bridge the sim-to-real gap for deployment on actual drones.

For future work, I plan to complete the UseGeo fine-tuning, investigate the failure cases with thin structures, and potentially explore lighter network architectures that could run in real-time on drone hardware.

Thank you, and I'm happy to take any questions.

---

## Slide 11: References

1. Fonder, M., Ernst, D., Van Droogenbroeck, M. (2022). "M4Depth: A motion-based approach for monocular depth estimation on video sequences." *Sensors*, 22(23), 9374.

2. MidAir Dataset: https://midair.ulg.ac.be/

3. M4Depth GitHub: https://github.com/michael-fonder/M4Depth

---

## Appendix: Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| RMSE Log | sqrt(mean((log d - log d*)²)) | Scale-invariant error |
| Abs Rel | mean(\|d - d*\| / d*) | Relative error |
| δ < 1.25 | % within 25% accuracy | Accuracy threshold |

---

## Appendix: Hardware/Software

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A3000 12GB |
| Framework | TensorFlow 2.15, Keras 2 |
| Dataset | MidAir (~317GB) |
| Training | 76 epochs, ~50 hours |

---

## Appendix: Reproducing Results

### Training
```bash
cd /home/bryan/dev/final_project/M4Depth
./train.sh
```

### Generate Training Curves
```bash
source ../m4depth_env/bin/activate
PYTHONPATH="/home/bryan/dev/final_project/m4depth_env/lib/python3.10/site-packages" \
python3 plot_from_tensorboard.py --output=training_curves.png
```

### Run Evaluation on Test Set
```bash
./eval.sh
# Results saved to: ./checkpoints/perfs-midair.txt
```

### Generate Visualizations
```bash
./visualize.sh --num_samples=20 --stride=50
# Images saved to: ./visualizations/
```

---

## Appendix: Understanding RMSE Log

**Why use log space for depth errors?**

| Distance | Prediction | Abs Error | Relative Error | Log Error |
|----------|------------|-----------|----------------|-----------|
| 2m | 3m | 1m | 50% | 0.405 |
| 100m | 101m | 1m | 1% | 0.010 |
| 100m | 150m | 50m | 50% | 0.405 |

- A 1m error at 2m (50% off) is **dangerous** for a drone
- A 1m error at 100m (1% off) is **negligible**
- Log space captures **relative accuracy**, which matters for safety
- RMSE_log of 0.21 ≈ predictions off by ~23% on average
