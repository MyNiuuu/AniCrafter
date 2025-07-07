

  <h2>AniCrafter: Customizing Realistic Human-Centric Animation via Avatar-Background Conditioning in Video Diffusion Models </h2>
<div>
    <a href='https://myniuuu.github.io/' target='_blank'>Muyao Niu</a> <sup>1,2</sup> &nbsp;
    <a href='https://github.com/ljzycmd' target='_blank'>Mingdeng Cao</a><sup>1</sup> &nbsp;
    <a href='https://yifever20002.github.io/' target='_blank'>Yifan Zhan</a><sup>1</sup> &nbsp;
    <a href='https://qtzhu.me/' target='_blank'>Qingtian Zhu</a><sup>1</sup> &nbsp; 
    <a href='https://github.com/mm2319' target='_blank'>Mingze Ma</a><sup>1</sup> &nbsp;
    <a href='https://github.com/zhaojiancheng007' target='_blank'>Jiancheng Zhao</a><sup>1</sup> &nbsp; 
<div>
<div>
    <a href='https://zengyh1900.github.io/' target='_blank'>Yanhong Zeng</a><sup>2</sup> &nbsp; 
    <a href='https://zzh-tech.github.io/' target='_blank'>Zhihang Zhong</a><sup>2</sup> &nbsp; 
    <a href='https://jimmysuen.github.io/' target='_blank'>Xiao Sun</a><sup>2</sup> &nbsp; 
    <a href='https://scholar.google.com/citations?user=JD-5DKcAAAAJ&hl=en' target='_blank'>Yinqiang Zheng</a><sup>1</sup> &nbsp; 
</div>
<div>
    <sup>1</sup> The University of Tokyo &nbsp; <sup>2</sup> Shanghai Artificial Intelligence Laboratory &nbsp; <sup>*</sup> Corresponding Authors &nbsp; 
</div>



<a href='https://arxiv.org/abs/2505.20255'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; <a href='https://myniuuu.github.io/AniCrafter'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; <a href='https://huggingface.co/MyNiuuu/Anicrafter_release'><img src='https://img.shields.io/badge/🤗 HuggingFace-AniCrafter-yellow'></a>


## TL;DR

> We leverage **"3DGS Avatar + Background Video"** as guidance for the video diffusion model to **insert and animate anyone into any scene following given motion sequence**.

<!-- <div align="center">
  <h3>
    <img src="assets/teaser_small.jpg"/>
  </h3>
</div> -->




<table align="center">
  <tr>
    <td align="center" width="13%">
      <img src="assets/character_image/000000.jpg"/>
      <br />
    </td>
    <td align="center" width="29%">
      <img src="assets/demo_videos/0.gif"/>
      <br />
    </td>
    <td align="center" width="29%">
      <img src="assets/demo_videos/1.gif"/>
      <br />
    </td>
    <td align="center" width="29%">
      <img src="assets/demo_videos/2.gif"/>
      <br />
    </td>
  </tr>
</table>



<table align="center">
  <tr>
    <td align="center" width="13%">
      <img src="assets/character_image/000001.jpg"/>
      <br />
    </td>
    <td align="center" width="29%">
      <img src="assets/demo_videos/3.gif"/>
      <br />
    </td>
    <td align="center" width="29%">
      <img src="assets/demo_videos/4.gif"/>
      <br />
    </td>
    <td align="center" width="29%">
      <img src="assets/demo_videos/5.gif"/>
      <br />
    </td>
  </tr>
</table>




## 🔥🔥🔥 New Features/Updates

- (2024.07.03) We have released the cross-character inference script to replace the person in the source video!
- (2025.07.02) Our [Project Page](https://myniuuu.github.io/AniCrafter) 🏠 is online!
- (2025.07.01) We have released the model and inference script to insert and animate the character into the background video following SMPLX motion sequences! 
- If you find this work interesting, please do not hesitate to give a ⭐!



## 📰 CODE RELEASE

- [x] (2024.07.01) Release model checkpoint and cross-character inference script.
- [x] (2024.07.03) Release the complete cross-character inference script including data preprocessing (mask parsing + SMPLX estimation + background inpainting).
- [ ] Release training codes.


## ⚙️ Environment Setup

### 🌍 Virtual Enviroment

```
conda create -n anicrafter python=3.10
conda activate anicrafter
bash install_cu124.sh
```


### 📦 Download Checkpoints

```
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
huggingface-cli download MyNiuuu/Anicrafter_release --local-dir ./Anicrafter_release
mv ./Anicrafter_release/gfpgan ./gfpgan
mv ./Anicrafter_release/pretrained_models ./pretrained_models
```


## 🏃 Cross-Character Inference from Background Video and Motions

Run the following commands to insert and animate the character into the background video following SMPLX motion sequences. The pipeline consists of following key functions:
- Reconstructing 3DGS Avatar from single image using [LHM](https://github.com/aigc3d/LHM)
- Animating the 3DGS Avatar according to the SMPLX sequences to obtain the spatial aligned avatar renderings
- Combine avatar rendering and background video to form the "Avatar + Background" condition
- Run the diffusion model to obtain the final animation results 

```
python run_pipeline.py \
--ckpt_path ./pretrained_models/anicrafter \
--wan_base_ckpt_path ./Wan2.1-I2V-14B-720P \
--character_image_path ./demo/character_images/000000.jpg \
--scene_path ./demo/videos/scene_000000 \
--save_root ./infer_result
```



## 🏃 Cross-Character Inference from in-the-wild Videos 
Run the following commands to replace the person in the source video with our complete data preprocessing pipeline, which contains the following components:

- Parsing human masks
- Estimating SMPLX parameters and rendering SMPLX mesh videos
- Background inpainting based on the human masks
- Reconstructing 3DGS Avatar from single image using [LHM](https://github.com/aigc3d/LHM)
- Animating the 3DGS Avatar according to the SMPLX sequences to obtain the spatial aligned avatar renderings
- Combine avatar rendering and background video to form the "Avatar + Background" condition
- Run the diffusion model to obtain the final animation results 


### ⚙️ Additional Environment Setup

```

cd engine/pose_estimation
pip install mmcv==1.3.9
pip install -v -e third-party/ViTPose
pip install ultralytics
pip install av
cd ../..
pip install numpy==1.23.5


mkdir weights
cd weights
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/cutie-base-mega.pth
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/i3d_rgb_imagenet.pt
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth
cd ..

# or you can mannually download from https://github.com/sczhou/ProPainter/releases/tag/v0.1.0
```

### 💻 Start Inference

```
# Mask + SMPLX + Inpainting + Avatar Recon + Rendering + Diffusion
# You could change the hyper-parameters of the inpainting algorithm to obtain optimal results

python run_pipeline_with_preprocess.py \
--video_root ./demo/origin_videos/raw_video \
--ckpt_path ./pretrained_models/anicrafter \
--wan_base_ckpt_path ./Wan2.1-I2V-14B-720P \
--character_image_path ./demo/character_images/000000.jpg \
--save_root ./infer_result_replace
```


## Citation
```
@article{niu2025anicrafter,
  title={AniCrafter: Customizing Realistic Human-Centric Animation via Avatar-Background Conditioning in Video Diffusion Models},
  author={Niu, Muyao and Cao, Mingdeng and Zhan, Yifan and Zhu, Qingtian and Ma, Mingze and Zhao, Jiancheng and Zeng, Yanhong and Zhong, Zhihang and Sun, Xiao and Zheng, Yinqiang},
  journal={arXiv preprint arXiv:2505.20255},
  year={2025}
}
```

## Acknowledgements
We sincerely appreciate the code release of the following projects: [LHM](https://github.com/aigc3d/LHM), [Unianimate-DiT](https://github.com/ali-vilab/UniAnimate-DiT), [Diffusers](https://github.com/huggingface/diffusers), and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
