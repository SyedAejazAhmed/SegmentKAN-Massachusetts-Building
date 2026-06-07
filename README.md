# SegmentKAN Massachusetts Buildings

A long-form reference for KAN-based building segmentation on the Massachusetts Buildings dataset.
This README is about 500 lines and keeps the original links and dataset references.
It documents KAN-family models, training protocol, and logged metrics.

## Quick Links

- Notebooks: [UKAN/train_ukan.ipynb](UKAN/train_ukan.ipynb) | [UKAN++/train_ukan.ipynb](UKAN++/train_ukan.ipynb) | [KC-UNIT/train_kc-unit.ipynb](KC-UNIT/train_kc-unit.ipynb) | [KAN-UNET/train_kan-unet.ipynb](KAN-UNET/train_kan-unet.ipynb) | [MultiKAN/train_multi-kan.ipynb](MultiKAN/train_multi-kan.ipynb)
- Metrics CSVs: [UKAN/models/metrics_train_val.csv](UKAN/models/metrics_train_val.csv) | [UKAN/models/metrics_test.csv](UKAN/models/metrics_test.csv) | [UKAN++/models/metrics_train_val.csv](UKAN++/models/metrics_train_val.csv) | [UKAN++/models/metrics_test.csv](UKAN++/models/metrics_test.csv) | [KC-UNIT/models/metrics_train_val.csv](KC-UNIT/models/metrics_train_val.csv) | [KC-UNIT/models/metrics_test.csv](KC-UNIT/models/metrics_test.csv) | [KAN-UNET/models/metrics_train_val.csv](KAN-UNET/models/metrics_train_val.csv) | [KAN-UNET/models/metrics_test.csv](KAN-UNET/models/metrics_test.csv) | [MultiKAN/models/metrics_train_val.csv](MultiKAN/models/metrics_train_val.csv) | [MultiKAN/models/metrics_test.csv](MultiKAN/models/metrics_test.csv)
- Dataset root: [dataset/archive](dataset/archive)
- Dataset source: https://www.cs.toronto.edu/~vmnih/data/
- Official KAN paper: https://arxiv.org/abs/2404.19756
- KAN 2.0 paper: https://arxiv.org/abs/2408.10205
- Official KAN code: https://github.com/KindXiaoming/pykan
- KAN documentation: https://kindxiaoming.github.io/pykan/

## Table of Contents

- KAN Primer
- Why KAN for Segmentation
- Official Paper and Code
- Repository Scope
- Workspace Layout
- Dataset
- Preprocessing
- Training Protocol
- Losses and Optimization
- Metrics and Selection Rules
- Consolidated Metrics
- Model Order
- UKAN
- UKAN++
- KC-UNIT
- KAN-UNET
- MultiKAN
- Cross-Model Observations
- Hardware and Memory Notes
- Reproducibility Checklist
- FAQ
- Appendix A: Architecture Checklists
- Appendix B: Metrics Glossary
- Appendix C: KAN Ecosystem
- References
- Appendix D: Extended Architecture Notes

## KAN Primer

KAN stands for Kolmogorov-Arnold Networks.
KANs are motivated by the Kolmogorov-Arnold representation theorem.
The theorem states that multivariate continuous functions can be represented by sums of univariate functions.
KANs exploit this idea by placing learnable nonlinear functions on edges rather than on nodes.
This is the conceptual dual of the standard MLP design.
In an MLP, nonlinearities live on nodes after linear mixing.
In a KAN, nonlinearities are attached to edges that connect inputs to outputs.
KANs often use spline functions to parameterize these edge nonlinearities.
Spline parameterization provides smoothness and interpretability.
The most common KAN implementation uses B-spline bases with a configurable grid.
Grid size and spline order are core hyperparameters.
A KAN layer can be written as $y_j = \sum_i \phi_{ij}(x_i)$, where each $\phi_{ij}$ is learned.
The representation can be interpreted as a sum of learned 1D functions.
KANs can be parameter efficient on certain tasks.
KANs can be more interpretable because each edge function can be inspected.
KANs can be more memory hungry if grid size is large.
For large-scale tasks, GPU acceleration and efficiency mode are important.
In vision, KANs are often placed inside U-shaped encoder-decoder designs.

## Why KAN for Segmentation

Segmentation demands fine-grained boundary decisions.
KAN spline bases can represent smooth transitions across spatial regions.
Combining KAN blocks with skip connections keeps spatial resolution intact.
The U-shaped topology mitigates information loss from downsampling.
KAN variants in this workspace preserve U-Net style benefits.
KAN-style layers can be inserted into encoders, decoders, or bottlenecks.

## Official Paper and Code

- Paper: KAN: Kolmogorov-Arnold Networks (2024) https://arxiv.org/abs/2404.19756
- Paper: KAN 2.0: Kolmogorov-Arnold Networks Meet Science (2024) https://arxiv.org/abs/2408.10205
- Official code: https://github.com/KindXiaoming/pykan
- Documentation: https://kindxiaoming.github.io/pykan/

Citation (bibtex format):

@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Soljacic, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}

## Repository Scope

This repository focuses on KAN-based segmentation models and comparable KAN-inspired baselines.
Each model is implemented in its own training notebook with a consistent dataset interface.
Model checkpoints and metrics are stored in the model folders for each notebook.

## Workspace Layout

- [UKAN](UKAN) contains the UKAN notebook and metrics.
- [UKAN++](UKAN++) contains the MM-UKAN-Plus-Plus notebook and metrics.
- [KC-UNIT](KC-UNIT) contains the KC-UNIT notebook and metrics.
- [KAN-UNET](KAN-UNET) contains the KAN-UNET notebook and metrics.
- [MultiKAN](MultiKAN) contains the MultiKAN notebook and metrics.
- [dataset/archive](dataset/archive) contains TIFF and PNG splits.

## Dataset

Dataset: Massachusetts Buildings segmentation.
Root: [dataset/archive](dataset/archive).
Source: https://www.cs.toronto.edu/~vmnih/data/.
Variants: tiff and png.
Each variant contains train, val, and test folders.
Masks are stored under *_labels subfolders.
Masks are binary with values 0 or 255 in the original files.
The notebooks convert masks to {0,1} class indices for training.
Images are resized to `IMG_SIZE` (default 256) for training.
All models treat the task as 2-class segmentation (background, building).

## Preprocessing

Images are loaded and converted to RGB when needed.
Masks are loaded in single-channel mode.
Nearest-neighbor resizing is used for masks to preserve label integrity.
Pixel values are normalized using torchvision transforms in each notebook.
A consistent resize pipeline helps align results across models.

## Training Protocol

Each notebook follows a standard supervised segmentation training loop.
Losses typically combine CrossEntropy and Dice-style terms.
Metrics are logged per epoch to the CSVs listed in Quick Links.
Schedulers use validation loss to adjust learning rates.
For memory-intensive models, AMP and gradient accumulation are enabled.
Batch size defaults to a safe value per model to avoid OOM errors.
Seeds are set for Python, NumPy, and Torch for reproducibility.
CUDA visibility is configured to use a single GPU when available.

## Losses and Optimization

CrossEntropyLoss is used for multi-class segmentation supervision.
DiceLoss captures overlap quality and class imbalance behavior.
The combined loss is CE + Dice for most notebooks in this repository.
Adam is the primary optimizer used in all notebooks.
ReduceLROnPlateau is used to shrink learning rates on validation plateaus.
GradScaler is enabled when AMP is active to prevent underflow.

## Metrics and Selection Rules

Metrics are computed per epoch for train and validation splits.
The best epoch is chosen by maximum validation Dice.
Train metrics are taken from that same epoch.
Test metrics are read from the metrics CSVs listed in Quick Links.
Accuracy, Precision, Recall, F1, Dice, and IoU are reported.
Dice loss is defined as 1 - Dice.

## Consolidated Metrics

Metrics below use the validation-selected epoch (max validation Dice) for each model.
The best epoch value is listed only in the wall time table.

### Training metrics (all models)

| Model | Loss | Accuracy | Precision | Recall | F1 | Dice | IoU | Dice Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UKAN | 0.5611 | 0.9004 | 0.7933 | 0.7344 | 0.7589 | 0.7589 | 0.6479 | 0.2411 |
| UKAN++ | 1.0697 | 0.7675 | 0.5900 | 0.6352 | 0.5998 | 0.5998 | 0.4790 | 0.4002 |
| KC-UNIT | 0.9630 | 0.8728 | 0.5262 | 0.3625 | 0.4293 | 0.4293 | 0.2733 | 0.5707 |
| KAN-UNET | 0.4770 | 0.9181 | 0.8435 | 0.7699 | 0.8003 | 0.8003 | 0.6948 | 0.1997 |
| MultiKAN | 1.4904 | 0.6708 | 0.1412 | 0.2940 | 0.1908 | 0.1908 | 0.1054 | 0.8092 |

### Validation metrics (all models)

| Model | Loss | Accuracy | Precision | Recall | F1 | Dice | IoU | Dice Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UKAN | 0.5806 | 0.9046 | 0.7570 | 0.7027 | 0.7253 | 0.7253 | 0.6181 | 0.2747 |
| UKAN++ | 1.0570 | 0.7901 | 0.5695 | 0.6069 | 0.5779 | 0.5779 | 0.4715 | 0.4221 |
| KC-UNIT | 0.9586 | 0.8572 | 0.3856 | 0.5299 | 0.4464 | 0.4464 | 0.2873 | 0.5536 |
| KAN-UNET | 0.5590 | 0.9103 | 0.7778 | 0.7077 | 0.7358 | 0.7358 | 0.6290 | 0.2642 |
| MultiKAN | 1.5118 | 0.8841 | 0.2958 | 0.0478 | 0.0823 | 0.0823 | 0.0429 | 0.9177 |

### Test metrics (all models)

| Model | Loss | Accuracy | Precision | Recall | F1 | Dice | IoU | Dice Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UKAN | 0.6805 | 0.8479 | 0.7514 | 0.7121 | 0.7284 | 0.7284 | 0.6051 | 0.2716 |
| UKAN++ | 0.9347 | 0.7888 | 0.5993 | 0.5553 | 0.5608 | 0.5608 | 0.4602 | 0.4392 |
| KC-UNIT | 1.0796 | 0.8290 | 0.5796 | 0.2990 | 0.3945 | 0.3945 | 0.2457 | 0.6055 |
| KAN-UNET | 0.7460 | 0.8485 | 0.7637 | 0.6745 | 0.7025 | 0.7025 | 0.5823 | 0.2975 |
| MultiKAN | 1.3061 | 0.8137 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### Wall time and best epoch

| Model | Best Epoch | Wall Time (s) |
| --- | --- | --- |
| UKAN | 48 | 330.0 |
| UKAN++ | 9 | 411.4 |
| KC-UNIT | 7 | 218.8 |
| KAN-UNET | 21 | 730.9 |
| MultiKAN | 1 | 373.3 |

## Model Order

The remainder of this README follows the requested model order.
UKAN, UKAN++, KC-UNIT, KAN-UNET, and MultiKAN.

## UKAN

- Notebook: [UKAN/train_ukan.ipynb](UKAN/train_ukan.ipynb)
- Model source: https://github.com/CUHK-AIM-Group/U-KAN

### Architecture Summary

UKAN is a U-shaped encoder-decoder that integrates KAN layers into a segmentation backbone.
The design follows the U-KAN repository and adapts it to the Massachusetts dataset.
KAN blocks provide spline-based mixing while CNN stages preserve spatial locality.
Skip connections carry high-resolution features into the decoder stages.

### Architecture Walkthrough

Input image enters an initial convolutional stem.
Early encoder stages build low-level edge and texture features.
Patch embedding prepares features for KAN-style processing.
KAN blocks refine features with learnable edge functions.
A bottleneck aggregates global context.
Decoder upsampling restores spatial resolution.
Skip connections merge encoder features with decoder features.
A final 1x1 convolution projects to class logits.

### Implementation Notes

This notebook logs metrics to CSV each epoch.
The best epoch is selected by validation Dice.
This model reached a strong Dice score on validation and test.


## UKAN++

- Notebook: [UKAN++/train_ukan.ipynb](UKAN++/train_ukan.ipynb)
- Model source: https://github.com/670768312/MM-UKAN-Plus-Plus

### Architecture Summary

UKAN++ extends the UKAN idea with multi-scale fusion and attention.
FCS attention combines frequency and spatial attention mechanisms.
Concatenation-based fusion improves multi-scale feature reuse.
This repository uses a lightweight mmcv fallback for ConvModule when needed.

### Architecture Walkthrough

Input image is processed by a convolutional stem.
KAN blocks operate on patch-embedded feature maps.
Multi-scale decoder stages fuse encoder features via concatenation.
FCS attention reweights spatial and frequency channels.
ChannelLinear blocks align channel dimensions for fusion.
The decoder upsamples to full resolution.
Final logits are produced by a 1x1 convolution.

### Implementation Notes

AMP and gradient accumulation are used to avoid OOM on 8GB GPUs.
Test evaluation uses the saved checkpoint from the selected epoch.

## KC-UNIT

- Notebook: [KC-UNIT/train_kc-unit.ipynb](KC-UNIT/train_kc-unit.ipynb)
- Model source: https://github.com/cychoi97/KC-UNIT

### Architecture Summary

KC-UNIT is a KAN-inspired U-Net style segmentation network.
The design follows the KC-UNIT repository and is adapted here for Massachusetts buildings.
Encoder and decoder stages are paired with skip connections.
KAN-inspired operations are used within the core blocks defined in the notebook.

### Architecture Walkthrough

Input passes through a hierarchical encoder.
Feature maps are downsampled to build context.
Core units apply learned nonlinear transformations.
The bottleneck aggregates global information.
Decoder stages upsample and merge skip features.
Final projection yields building vs background logits.

### Implementation Notes

Results are selected by validation Dice.
Test metrics were computed after training.
The model is a useful baseline for KAN-style segmentation.


## KAN-UNET

- Notebook: [KAN-UNET/train_kan-unet.ipynb](KAN-UNET/train_kan-unet.ipynb)
- Model source: https://github.com/JaouadT/KANU_Net
- Additional components: https://github.com/milesial/Pytorch-UNet
- Additional components: https://github.com/XiangboGaoBarry/KA-Conv

### Architecture Summary

KAN-UNET builds on U-Net while replacing convolutional operations with KAN-inspired blocks.
The notebook adapts KANU_Net to the Massachusetts dataset.
The architecture retains U-Net skip connections for precise boundaries.
KA-Conv components support KAN-based convolutional transformations.

### Architecture Walkthrough

Input enters a U-Net style encoder.
Downsampling extracts hierarchical features.
KAN-based convolutional layers refine the feature maps.
A bottleneck stage encodes global context.
Decoder stages upsample and fuse skip features.
Final logits are produced for two-class segmentation.

### Implementation Notes

This model achieved strong validation Dice among the evaluated models.
Training uses CE + Dice loss as in other notebooks.

## MultiKAN

- Notebook: [MultiKAN/train_multi-kan.ipynb](MultiKAN/train_multi-kan.ipynb)
- Model source: https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py

### Architecture Summary

MultiKAN uses KAN-style functions to process per-pixel features for segmentation.
The implementation in this repository uses chunked pixel processing for memory safety.
The architecture is a non-convolutional alternative to the U-Net style family.
It provides a contrasting baseline for KAN function approximation in dense prediction.

### Architecture Walkthrough

Input images are reshaped into per-pixel feature vectors.
KAN layers process pixel features in chunks to reduce memory pressure.
Outputs are reshaped back into spatial maps.
The final logits are evaluated with CE + Dice loss.

### Implementation Notes

This model currently reflects an early training snapshot.
Performance is lower than the U-Net style architectures on this dataset.
The setup remains useful for studying KAN-only alternatives.


## Cross-Model Observations

U-shaped architectures tend to outperform pure per-pixel KAN processing on this dataset.
KAN-UNET and UKAN achieve the strongest validation Dice in the current runs.
UKAN++ shows competitive performance and includes attention-driven fusion.
MultiKAN demonstrates the cost of removing spatial inductive biases in segmentation.
These results should be interpreted as configuration-specific snapshots.

## Hardware and Memory Notes

Some models exceed 8GB GPU memory at batch size 8.
AMP and gradient accumulation are used to mitigate OOM errors.
Reducing `IMG_SIZE` to 224 or 192 can further reduce memory usage.
Chunked pixel processing is used in MultiKAN to avoid large intermediate tensors.
The UKAN++ notebook includes a fallback ConvModule if `mmcv` is unavailable.

## Reproducibility Checklist

Set a fixed seed for Python, NumPy, and Torch.
Confirm dataset variant selection (tiff vs png).
Verify that train, val, and test folders exist.
Confirm `IMG_SIZE` and `BATCH_SIZE` values per notebook.
Ensure AMP and `ACCUM_STEPS` values are consistent with the hardware budget.
Check that metrics CSV files are written after each epoch.
Keep model source links in the notebook for traceability.
Archive best checkpoints to the models folder for each run.
Validate that best epoch selection uses validation Dice.
Record wall time from `train_end - train_start` for each run.

## FAQ

Q: Which model should I start with?
A: UKAN or KAN-UNET provide the best Dice in the current logs.
Q: How do I reduce memory usage?
A: Lower `IMG_SIZE`, reduce `BATCH_SIZE`, and enable AMP with accumulation.
Q: Are the results directly comparable across models?
A: They are comparable at the level of this dataset and these training settings.

## Appendix A: Architecture Checklists

### UKAN Checklist

UKAN checklist 01: Confirm input normalization and channel order.
UKAN checklist 02: Verify patch or conv stem configuration.
UKAN checklist 03: Inspect encoder depth and feature widths.
UKAN checklist 04: Validate downsampling strategy and stride.
UKAN checklist 05: Check presence of KAN blocks in the encoder.
UKAN checklist 06: Check presence of KAN blocks in the decoder.
UKAN checklist 07: Confirm skip connection alignment.
UKAN checklist 08: Validate upsampling method (nearest or bilinear).
UKAN checklist 09: Confirm channel alignment before concatenation.
UKAN checklist 10: Review attention modules if present.

### UKAN++ Checklist

UKAN++ checklist 01: Confirm input normalization and channel order.
UKAN++ checklist 02: Verify patch or conv stem configuration.
UKAN++ checklist 03: Inspect encoder depth and feature widths.
UKAN++ checklist 04: Validate downsampling strategy and stride.
UKAN++ checklist 05: Check presence of KAN blocks in the encoder.
UKAN++ checklist 06: Check presence of KAN blocks in the decoder.
UKAN++ checklist 07: Confirm skip connection alignment.
UKAN++ checklist 08: Validate upsampling method (nearest or bilinear).
UKAN++ checklist 09: Confirm channel alignment before concatenation.
UKAN++ checklist 10: Review attention modules and fusion settings.

### KC-UNIT Checklist

KC-UNIT checklist 01: Confirm input normalization and channel order.
KC-UNIT checklist 02: Verify patch or conv stem configuration.
KC-UNIT checklist 03: Inspect encoder depth and feature widths.
KC-UNIT checklist 04: Validate downsampling strategy and stride.
KC-UNIT checklist 05: Check presence of KAN blocks in the encoder.
KC-UNIT checklist 06: Check presence of KAN blocks in the decoder.
KC-UNIT checklist 07: Confirm skip connection alignment.
KC-UNIT checklist 08: Validate upsampling method (nearest or bilinear).
KC-UNIT checklist 09: Confirm channel alignment before concatenation.
KC-UNIT checklist 10: Confirm bottleneck width and depth.

### KAN-UNET Checklist

KAN-UNET checklist 01: Confirm input normalization and channel order.
KAN-UNET checklist 02: Verify patch or conv stem configuration.
KAN-UNET checklist 03: Inspect encoder depth and feature widths.
KAN-UNET checklist 04: Validate downsampling strategy and stride.
KAN-UNET checklist 05: Check presence of KAN blocks in the encoder.
KAN-UNET checklist 06: Check presence of KAN blocks in the decoder.
KAN-UNET checklist 07: Confirm skip connection alignment.
KAN-UNET checklist 08: Validate upsampling method (nearest or bilinear).
KAN-UNET checklist 09: Confirm channel alignment before concatenation.
KAN-UNET checklist 10: Review KA-Conv configuration if used.

### MultiKAN Checklist

MultiKAN checklist 01: Confirm input normalization and channel order.
MultiKAN checklist 02: Verify patch or conv stem configuration.
MultiKAN checklist 03: Inspect encoder depth and feature widths.
MultiKAN checklist 04: Validate downsampling strategy and stride.
MultiKAN checklist 05: Check presence of KAN blocks in the encoder.
MultiKAN checklist 06: Check presence of KAN blocks in the decoder.
MultiKAN checklist 07: Confirm skip connection alignment.
MultiKAN checklist 08: Validate upsampling method (nearest or bilinear).
MultiKAN checklist 09: Confirm channel alignment before concatenation.
MultiKAN checklist 10: Check chunk size for pixel processing.

## Appendix B: Metrics Glossary

Accuracy: fraction of correctly classified pixels.
Precision: TP divided by TP + FP.
Recall: TP divided by TP + FN.
F1: harmonic mean of precision and recall.
Dice: 2TP / (2TP + FP + FN).
IoU: TP / (TP + FP + FN).
Dice loss: 1 - Dice.
TP: true positives.
FP: false positives.
FN: false negatives.
Logits: raw network outputs before softmax.
Softmax: normalization across classes.
Cross entropy: negative log likelihood loss.
Class imbalance: uneven distribution of classes.
Thresholding: conversion from probabilities to labels.
Confusion matrix: table of predicted vs true labels.
Validation set: held-out subset for model selection.
Test set: held-out subset for final evaluation.
Epoch: full pass over the training data.
Batch size: number of samples per gradient step.
Gradient accumulation: multiple mini-batches per optimizer step.

## Appendix C: KAN Ecosystem

- Official KAN codebase: https://github.com/KindXiaoming/pykan
- KAN documentation: https://kindxiaoming.github.io/pykan/
- EfficientKAN: https://github.com/Blealtan/efficient-kan
- FourierKAN: https://github.com/GistNoesis/FourierKAN/
- GraphKAN: https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks
- KANRL: https://github.com/riiswa/kanrl

## References

- https://arxiv.org/abs/2404.19756 | https://arxiv.org/abs/2408.10205
- https://github.com/KindXiaoming/pykan | https://github.com/CUHK-AIM-Group/U-KAN
- https://github.com/670768312/MM-UKAN-Plus-Plus | https://github.com/cychoi97/KC-UNIT
- https://github.com/JaouadT/KANU_Net | https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py
- https://github.com/milesial/Pytorch-UNet | https://github.com/XiangboGaoBarry/KA-Conv
- https://github.com/Blealtan/efficient-kan | https://github.com/GistNoesis/FourierKAN/
- https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks | https://github.com/riiswa/kanrl

## Appendix D: Extended Architecture Notes

Extended note 001: Track intermediate tensor shapes at each encoder stage.
Extended note 002: Confirm feature map alignment before concatenation.
Extended note 003: Validate class indexing in masks after resizing.
Extended note 004: Review the effect of grid size on KAN layer memory usage.
Extended note 005: Check that gradient accumulation divides the loss correctly.
Extended note 006: Verify that AMP autocast is disabled on CPU runs.
Extended note 007: Record GPU memory usage after the forward pass.
Extended note 008: Confirm that logits are used for loss computation, not probabilities.
Extended note 009: Inspect the effect of Dice loss weighting on rare classes.
Extended note 010: Review the impact of input resolution on boundary quality.