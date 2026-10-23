# Selective-Stereo: faster inference fork

> **You are on `cuda-kernels`.** Research fork of [Windsrain/Selective-Stereo](https://github.com/Windsrain/Selective-Stereo) (base `c39acbf`; `main` mirrors upstream). Custom CUDA kernels, so this branch does not export to ONNX / TensorRT; there is no `portable` branch for this repo.
>
> What needs the kernels: the fp16 fast path in the `*_fast.py` / `inference.py` files (cuDNN-frontend fp16-accumulate convolutions, CUTLASS fused-epilogue convs, cost-volume / correlation kernels). Upstream's own modules and weights are unchanged.
>
> Build: prebuilt cubins (sm_80/86/89/90/120) and one CUTLASS shared library sit in `prebuilt/` next to the code (each < 3 MB). If a prebuilt does not match your GPU or the source hash, the code compiles it with `nvcc` from `CUDA_HOME` (default `/usr/local/cuda`) into `~/.cache`; the CUTLASS library is only rebuilt from source with `CUTLASS_DIR` set (CUTLASS v4.7.1 headers). fp16-accumulate convs need `pip install nvidia-cudnn-frontend`.
>

> Extra dependency: [`fastbb`](https://github.com/fastforks/fastbb) (fast backbones), `pip install` it or put it on `PYTHONPATH`.
>

> Licence, as upstream (MIT)


Released weights, RTX 5060 Ti 16 GB, torch 2.14.1+cu130, 2026-10-23. KITTI = 000000_10 (1242x375), Midd-H = Middlebury trainingH Motorcycle (1482x994). Median of 10 runs after warm-up (the first-call column is the very first call of a fresh process). Peak memory = torch peak allocated / reserved over the whole run (both sizes). EPE = mean over 3 ETH3D + 3 Middlebury-H training pairs (short check, not a benchmark; a pair that ran out of memory in any row is left out of the EPE mean of all rows). flags = set from the driver only, no upstream source edit: `torch.inference_mode()`, `cudnn.benchmark=True`, TF32 allowed, plus the fastest combination that keeps EPE within noise (listed in brackets in the row). `torch.compile` is not part of the row.



| version | KITTI ms | Midd-H ms | speed-up at H (vs flags / vs upstream) | peak MB (alloc / reserved) | startup / first call / new size | ETH3D EPE | Midd-H EPE |
|---|---|---|---|---|---|---|---|
| upstream (fp32) | 634.1 | 2192.8 | 1.00x | 2888 / 3980 | 3.0 s / 1.04 s / 2.19 s | 0.403 | 7.481 |
| upstream + flags (cudnn.benchmark, TF32, inference_mode) | 606.4 | 2148.7 | 1.00x / 1.02x | 11448 / 12008 | 2.7 s / 4.23 s / 8.05 s | 0.403 | 7.503 |
| **cuda-kernels** | 113.1 | 345.9 | 6.21x / 6.34x | 1119 / 1376 | 2.8 s / 1.46 s / 0.46 s | 0.402 | 7.449 |

Max / mean abs disparity difference cuda-kernels vs upstream: 148 px / 0.073 px (KITTI + 6 EPE pairs; the maximum is set by a few single pixels).


---

# Selective-Stereo (CVPR 2024 Highlight)

This repository contains the source code for our paper: 

[Selective-Stereo: Adaptive Frequency Information Selection for Stereo Matching](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Selective-Stereo_Adaptive_Frequency_Information_Selection_for_Stereo_Matching_CVPR_2024_paper.pdf)

CVPR 2024 Highlight

Xianqi Wang, Gangwei Xu, Hao Jia, Xin Yang

```
@inproceedings{wang2024selective,
  title={Selective-stereo: Adaptive frequency information selection for stereo matching},
  author={Wang, Xianqi and Xu, Gangwei and Jia, Hao and Yang, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19701--19710},
  year={2024}
}
```

![Overview](./Selective-RAFT/Selective-RAFT.png)

## Demo

Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1HwwGicI9f_WWOdjh3t5l6l4cn64zr_V5?usp=sharing).

To evaluate SceneFlow, run

```
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow.pth
```

To predict Middlebury, run

```
python demo_imgs.py --restore_ckpt ./pretrained_models/middlebury_finetune.pth --valid_iters 80 --max_disp 768
```

# Environment

```
conda create -n Selective_Stereo python=3.8
conda activate Selective_Stereo

pip install torch torchvision
pip install tqdm timm==0.5.4
pip install opencv-python scikit-image matplotlib
pip install opt_einsum
```

# Required Data

* [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [ETH3D](https://www.eth3d.net/datasets)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [TartanAir](https://github.com/castacks/tartanair_tools)
* [CREStereo Dataset](https://github.com/megvii-research/CREStereo)
* [FallingThings](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)
* [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
* [Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)
* [HR-VS](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view)

# Training

To train on SceneFlow, run

```
python train_stereo.py --logdir ./checkpoints/sceneflow
```

To train on KITTI, run

```
python train_stereo.py --restore_ckpt ./pretrained_models/sceneflow.pth --train_datasets kitti --num_steps 50000
```

To train on ETH3D, run

```
python train_stereo.py --restore_ckpt ./pretrained_models/sceneflow.pth --train_datasets eth3d_train --num_steps 300000 --image_size 384 512
python train_stereo.py --restore_ckpt ./pretrained_models/eth3d_train.pth --tarin_datasets eth3d_finetune --num_steps 90000
```

To train on Middlebury, run

```
python train_stereo.py --restore_ckpt ./pretrained_models/sceneflow.pth --train_datasets middlebury_train --num_steps 200000 --image_size 384 512
python train_stereo.py --restore_ckpt ./pretrained_models/middlebury_train.pth --tarin_datasets middlebury_finetune --num_steps 100000 --image_size 384 768 --max_disp 768
```
