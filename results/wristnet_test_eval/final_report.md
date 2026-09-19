# WristTrauma-AI -- WristNet (Custom CNN) Test-Set Evaluation

- Checkpoint: `D:\Study\Projects\Fracture-Detection_HDAsubject\WristTrauma-AI\models\wristnet\best.pt` (epoch 65)
- Test set size: 2992 images (patient-level held-out split, never used in training)
- Thresholds: tuned on VAL split, frozen and applied unchanged here (`tuned_thresholds.json`)

## Overall
- Macro AUROC: 0.8896
- Macro AUPRC: 0.7017
- Macro F1 (at tuned thresholds): 0.6845

## Per-class
| Class | AUROC | AUPRC | AUPRC random baseline | Precision | Recall | F1 | Threshold | n_positive |
|---|---|---|---|---|---|---|---|---|
| fracture | 0.9672 | 0.9824 | 0.6531 | 0.9288 | 0.9550 | 0.9417 | 0.39 | 1954/2992 |
| foreign_material | 0.9960 | 0.9870 | 0.0244 | 0.9351 | 0.9863 | 0.9600 | 0.75 | 73/2992 |
| softtissue_indirect | 0.9216 | 0.7051 | 0.1454 | 0.6208 | 0.7264 | 0.6695 | 0.61 | 435/2992 |
| bone_lesion | 0.6734 | 0.1322 | 0.0130 | 0.1404 | 0.2051 | 0.1667 | 0.67 | 39/2992 |

**Reading AUPRC**: compare each class's AUPRC to its own random baseline (= prevalence), not to 0 or to other classes' AUPRC. A class with 1% prevalence and AUPRC 0.15 is meaningfully better than random even though 0.15 looks low next to a common class's AUPRC.

CAM sample visualizations (original | heatmap | boxes): `D:\Study\Projects\Fracture-Detection_HDAsubject\WristTrauma-AI\results\wristnet_test_eval\cam_samples`