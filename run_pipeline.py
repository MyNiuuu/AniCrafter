import os
import torch
import imageio
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import json
import math
import cv2
import argparse

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply
from torchvision.transforms import v2

from graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix_refine
from lhm_runner import HumanLRMInferrer
from LHM.models.rendering.smpl_x_voxel_dense_sampling import SMPLXVoxelMeshModel
from diffsynth import ModelManager, WanMovieCrafterCombineVideoPipeline


def pad_image_to_aspect_ratio(image, target_width, target_height, background_color=(255, 255, 255)):

    target_ratio = target_width / target_height
    image_ratio = image.width / image.height
    
    if image.width > target_width or image.height > target_height:
        if image_ratio > target_ratio:
            scale_factor = target_width / image.width
        else:
            scale_factor = target_height / image.height
        
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, Image.LANCZOS)
    
    padded_image = ImageOps.pad(
        image, 
        (target_width, target_height), 
        color=background_color,
        centering=(0.5, 0.5)
    )
    
    return padded_image


def save_video(ref_frame_pils, smplx_pils, blend_pils, video, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for ref_frame, smplx, blend, frame in tqdm(zip(ref_frame_pils, smplx_pils, blend_pils, video), desc="Saving video"):
        h, w, c = np.array(ref_frame).shape
        if h >= w:
            all_frame = np.hstack([
                np.array(ref_frame), 
                np.array(smplx), 
                np.array(blend), 
                np.array(frame)
            ])
        else:
            all_frame = np.vstack([
                np.hstack([
                    np.array(ref_frame), 
                    np.array(smplx), 
                ]), 
                np.hstack([
                    np.array(blend), 
                    np.array(frame)
                ])
            ])
        writer.append_data(all_frame)
    writer.close()



def to_cuda_and_squeeze(value):
    if isinstance(value, dict):  # 如果是字典，则递归处理
        return {k: to_cuda_and_squeeze(v) for k, v in value.items()}
    elif isinstance(value, torch.Tensor):  # 如果是Tensor，则转移到CUDA并压缩
        return value.cuda().squeeze(0)
    return value  # 其他类型的值不处理，直接返回


def PILtoTorch(pil_image):
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

    
def load_camera(pose):
    intrinsic = torch.eye(3)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()

    image_width, image_height = pose["img_size_wh"]

    c2w = torch.eye(4)
    c2w = c2w.float()

    return c2w, intrinsic, image_height, image_width


def video_to_pil_images(video_path, height, width):
    if video_path.endswith('.mp4'):
        cap = cv2.VideoCapture(video_path)
        pil_images = []
        while True:
            ret, frame = cap.read()  # 读取一帧
            if not ret:
                break  # 视频结束或读取失败
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb).resize((width, height), Image.Resampling.LANCZOS)
            pil_images.append(pil_image)
        cap.release()
    elif os.path.isdir(video_path):
        frame_files = sorted([os.path.join(video_path, x) for x in os.listdir(video_path) if x.endswith('.jpg')])
        pil_images = []
        for frame in frame_files:
            frame = cv2.imread(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb).resize((width, height), Image.Resampling.LANCZOS)
            pil_images.append(pil_image)
    else:
        raise ValueError("Unsupported video format. Please provide a .mp4 file or a directory of images.")
    return pil_images


def animate_gs_model(
    offset_xyz, shs, opacity, scaling, rotation, query_points, smplx_data, SMPLX_MODEL
):
    """
    query_points: [N, 3]
    """

    device = offset_xyz.device

    # build cano_dependent_pose
    cano_smplx_data_keys = [
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "expr",
        "trans",
    ]

    merge_smplx_data = dict()
    for cano_smplx_data_key in cano_smplx_data_keys:
        warp_data = smplx_data[cano_smplx_data_key]
        cano_pose = torch.zeros_like(warp_data[:1])

        if cano_smplx_data_key == "body_pose":
            # A-posed
            cano_pose[0, 15, -1] = -math.pi / 6
            cano_pose[0, 16, -1] = +math.pi / 6

        merge_pose = torch.cat([warp_data, cano_pose], dim=0)
        merge_smplx_data[cano_smplx_data_key] = merge_pose

    merge_smplx_data["betas"] = smplx_data["betas"]
    merge_smplx_data["transform_mat_neutral_pose"] = smplx_data[
        "transform_mat_neutral_pose"
    ]

    with torch.autocast(device_type=device.type, dtype=torch.float32):
        mean_3d = (
            query_points + offset_xyz
        )  # [N, 3]  # canonical space offset.

        # matrix to warp predefined pose to zero-pose
        transform_mat_neutral_pose = merge_smplx_data[
            "transform_mat_neutral_pose"
        ]  # [55, 4, 4]
        num_view = merge_smplx_data["body_pose"].shape[0]  # [Nv, 21, 3]
        mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, N, 3]
        query_points = query_points.unsqueeze(0).repeat(num_view, 1, 1)
        transform_mat_neutral_pose = transform_mat_neutral_pose.unsqueeze(0).repeat(
            num_view, 1, 1, 1
        )

        mean_3d, transform_matrix = (
            SMPLX_MODEL.transform_to_posed_verts_from_neutral_pose(
                mean_3d,
                merge_smplx_data,
                query_points,
                transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
                device=device,
            )
        )  # [B, N, 3]

        # rotation appearance from canonical space to view_posed
        num_view, N, _, _ = transform_matrix.shape
        transform_rotation = transform_matrix[:, :, :3, :3]

        rigid_rotation_matrix = torch.nn.functional.normalize(
            matrix_to_quaternion(transform_rotation), dim=-1
        )
        I = matrix_to_quaternion(torch.eye(3)).to(device)

        # inference constrain
        is_constrain_body = SMPLX_MODEL.is_constrain_body
        rigid_rotation_matrix[:, is_constrain_body] = I
        scaling[is_constrain_body] = scaling[
            is_constrain_body
        ].clamp(max=0.02)

        rotation_neutral_pose = rotation.unsqueeze(0).repeat(num_view, 1, 1)

        # QUATERNION MULTIPLY
        rotation_pose_verts = quaternion_multiply(
            rigid_rotation_matrix, rotation_neutral_pose
        )
    
    gaussian_xyz = mean_3d[0]
    canonical_xyz = mean_3d[1]
    gaussian_opacity = opacity
    gaussian_rotation = rotation_pose_verts[0]
    canonical_rotation = rotation_pose_verts[1]
    gaussian_scaling = scaling
    gaussian_rgb = shs

    return gaussian_xyz, canonical_xyz, gaussian_rgb, gaussian_opacity, gaussian_rotation, canonical_rotation, gaussian_scaling, rigid_rotation_matrix


def get_camera_smplx_data(smplx_path):
    with open(smplx_path) as f:
        smplx_raw_data = json.load(f)
    
    smplx_param = {
        k: torch.FloatTensor(v)
        for k, v in smplx_raw_data.items()
        if "pad_ratio" not in k
    }
    
    c2w, K, image_height, image_width = load_camera(smplx_param)
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    
    focalX = K[0, 0]
    focalY = K[1, 1]
    FovX = focal2fov(focalX, image_width)
    FovY = focal2fov(focalY, image_height)

    zfar = 1000
    znear = 0.001
    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0

    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
    projection_matrix = getProjectionMatrix_refine(torch.Tensor(K), image_height, image_width, znear, zfar).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    smplx_param['betas'] = esti_shape
    smplx_param['expr'] = torch.zeros((100))

    return {
        'smplx_param': smplx_param, 
        'w2c': w2c, 
        'R': R, 
        'T': T, 
        'K': K, 
        'FoVx': FovX, 
        'FoVy': FovY, 
        'zfar': zfar, 
        'znear': znear, 
        'trans': trans, 
        'scale': scale, 
        'world_view_transform': world_view_transform, 
        'projection_matrix': projection_matrix, 
        'full_proj_transform': full_proj_transform, 
        'camera_center': camera_center, 
    }




def prepare_models(wan_base_ckpt_path, lora_ckpt_path):
    lhm_runner = HumanLRMInferrer()

    SMPLX_MODEL = SMPLXVoxelMeshModel(
        './pretrained_models/human_model_files',
        gender="neutral",
        subdivide_num=1,
        shape_param_dim=10,
        expr_param_dim=100,
        cano_pose_type=1,
        dense_sample_points=40000,
        apply_pose_blendshape=False,
    ).cuda()

    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [os.path.join(wan_base_ckpt_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
        torch_dtype=torch.float32, # Image Encoder is loaded with float32
    )
    model_manager.load_models(
        [
            [
                os.path.join(wan_base_ckpt_path, "diffusion_pytorch_model-00001-of-00007.safetensors"),
                os.path.join(wan_base_ckpt_path, "diffusion_pytorch_model-00002-of-00007.safetensors"),
                os.path.join(wan_base_ckpt_path, "diffusion_pytorch_model-00003-of-00007.safetensors"),
                os.path.join(wan_base_ckpt_path, "diffusion_pytorch_model-00004-of-00007.safetensors"),
                os.path.join(wan_base_ckpt_path, "diffusion_pytorch_model-00005-of-00007.safetensors"),
                os.path.join(wan_base_ckpt_path, "diffusion_pytorch_model-00006-of-00007.safetensors"),
                os.path.join(wan_base_ckpt_path, "diffusion_pytorch_model-00007-of-00007.safetensors"),

            ],
            os.path.join(wan_base_ckpt_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(wan_base_ckpt_path, "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )

    model_manager.load_lora_v2_combine([
        os.path.join(lora_ckpt_path, "model-00010-of-00011.safetensors"),
        os.path.join(lora_ckpt_path, "model-00011-of-00011.safetensors"),
        ], lora_alpha=1.0)

    # assert False

    pipe = WanMovieCrafterCombineVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
    pipe.enable_vram_management()

    frame_process_norm = v2.Compose([
        v2.Resize(size=(H, W), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return lhm_runner, SMPLX_MODEL, pipe, frame_process_norm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # 添加分布式参数
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--wan_base_ckpt_path", type=str, required=True)
    parser.add_argument("--character_image_path", type=str, required=True)
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    args = parser.parse_args()

    H, W = 720, 1280
    H, W = math.ceil(H / 16) * 16, math.ceil(W / 16) * 16
    print(H, W)

    seed = 0
    max_frames = 81
    use_teacache = False
    cfg_value = 1.5
    caption = "human in a scene"

    ckpt_path = args.ckpt_path
    wan_base_ckpt_path = args.wan_base_ckpt_path
    character_image_path = args.character_image_path
    scene_path = args.scene_path
    save_root = args.save_root
    
    bkgd_video_path = os.path.join(scene_path, 'bkgd_video.mp4')
    smplx_path = os.path.join(scene_path, 'smplx_params')
    smplx_mesh_path = os.path.join(scene_path, 'smplx_video.mp4')

    save_gaussian_path = character_image_path.replace('.jpg', '_gaussian.pth')
    save_video_path = os.path.join(save_root, f'{os.path.basename(scene_path)}/{os.path.basename(character_image_path).split(".")[0]}.mp4')
    
    os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_gaussian_path), exist_ok=True)

    lhm_runner, SMPLX_MODEL, pipe, frame_process_norm = prepare_models(
        wan_base_ckpt_path, ckpt_path
    )

    gaussians_list, body_rgb_pil, crop_body_pil = lhm_runner.infer(
        character_image_path, save_gaussian_path
    )

    body_rgb_pil_pad = pad_image_to_aspect_ratio(crop_body_pil, W, H)

    dxdydz, xyz, rgb, opacity, scaling, rotation, transform_mat_neutral_pose, esti_shape, body_ratio, have_face = gaussians_list

    bkgd_pils_origin = video_to_pil_images(bkgd_video_path, H, W)
    smplx_mesh_pils_origin = video_to_pil_images(smplx_mesh_path, H, W)
    smplx_json_paths = sorted(os.path.join(smplx_path, x) for x in os.listdir(smplx_path))

    smplx_mesh_tensors = [torch.from_numpy(np.array(smplx_mesh_pil)) / 255. for smplx_mesh_pil in smplx_mesh_pils_origin]

    smplx_mask_nps = []
    bkgd_nps = []
    for smplx_mesh_tensor, bkgd_pil in zip(smplx_mesh_tensors, bkgd_pils_origin):
        smplx_mask = (smplx_mesh_tensor <= 0.01).all(dim=-1, keepdim=False).float()  # [720, 1280]
        smplx_mask_np = np.uint8(255 - smplx_mask.detach().cpu().numpy() * 255)  # [80, h, w]
        smplx_mask_nps.append(smplx_mask_np)
        bkgd_nps.append(np.array(bkgd_pil))

    blend_pils_origin = []

    for bkgd_pil, smplx_json_path in tqdm(zip(bkgd_pils_origin, smplx_json_paths), desc="Rendering Avatar", total=len(smplx_json_paths)):

        batch = {
            key: to_cuda_and_squeeze(value) 
            for key, value in get_camera_smplx_data(
                smplx_json_path
            ).items()
        }

        render_image_width, render_image_height = int(batch['smplx_param']["img_size_wh"][0]), int(batch['smplx_param']["img_size_wh"][1])

        gaussian_canon_dxdydz = dxdydz.cuda()
        query_points = xyz.cuda()
        gaussian_canon_rgb = rgb.cuda()
        gaussian_canon_opacity = opacity.cuda()
        gaussian_canon_scaling = scaling.cuda()
        gaussian_canon_rotation = rotation.cuda()
        transform_mat_neutral_pose = transform_mat_neutral_pose.cuda()
        esti_shape = esti_shape.cuda()

        smplx_data = {
            'betas': batch['smplx_param']['betas'].unsqueeze(0), 
            'root_pose': batch['smplx_param']['root_pose'].unsqueeze(0), 
            'body_pose': batch['smplx_param']['body_pose'].unsqueeze(0), 
            'jaw_pose': batch['smplx_param']['jaw_pose'].unsqueeze(0), 
            'leye_pose': batch['smplx_param']['leye_pose'].unsqueeze(0), 
            'reye_pose': batch['smplx_param']['reye_pose'].unsqueeze(0), 
            'lhand_pose': batch['smplx_param']['lhand_pose'].unsqueeze(0), 
            'rhand_pose': batch['smplx_param']['rhand_pose'].unsqueeze(0), 
            'trans': batch['smplx_param']['trans'].unsqueeze(0), 
            'expr': batch['smplx_param']['expr'].unsqueeze(0), 
            'transform_mat_neutral_pose': transform_mat_neutral_pose, 
        }

        gaussian_xyz, canonical_xyz, gaussian_rgb, gaussian_opacity, gaussian_rotation, canonical_rotation, gaussian_scaling, transform_matrix = \
            animate_gs_model(
                gaussian_canon_dxdydz, gaussian_canon_rgb, gaussian_canon_opacity, 
                gaussian_canon_scaling, gaussian_canon_rotation, query_points, 
                smplx_data, SMPLX_MODEL
            )
        
        # Set up rasterization configuration
        tanfovx = math.tan(batch['FoVx'] * 0.5)
        tanfovy = math.tan(batch['FoVy'] * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=render_image_height,
            image_width=render_image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            scale_modifier=1.,
            viewmatrix=batch['world_view_transform'],
            projmatrix=batch['full_proj_transform'],
            sh_degree=0,
            campos=batch['camera_center'],
            prefiltered=False,
            debug=False
        )
            
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, depth, alpha = rasterizer(
            means3D = gaussian_xyz, 
            means2D = torch.zeros_like(canonical_xyz, dtype=canonical_xyz.dtype, requires_grad=False, device="cuda") + 0, 
            shs = None, 
            colors_precomp = gaussian_rgb, 
            opacities = gaussian_opacity, 
            scales = gaussian_scaling, 
            rotations = gaussian_rotation, 
            cov3D_precomp = None
        )

        blend_image = rendered_image * alpha + PILtoTorch(bkgd_pil.resize((render_image_width, render_image_height), Image.Resampling.LANCZOS)).cuda() * (1 - alpha)

        blend_image = Image.fromarray(np.uint8(blend_image.permute(1, 2, 0).detach().cpu().numpy() * 255))
        blend_image = blend_image.resize((W, H), Image.Resampling.LANCZOS)

        blend_pils_origin.append(blend_image)

    smplx_mesh_pils_origin = video_to_pil_images(smplx_mesh_path, H, W)

    ref_frame = body_rgb_pil_pad
    ref_frame_tensor = frame_process_norm(ref_frame).cuda()

    ref_frame_pils_origin = [ref_frame for _ in range(max_frames)]

    blend_tensor = torch.stack([frame_process_norm(ss) for ss in blend_pils_origin], dim=0).cuda().permute(1, 0, 2, 3)
    smplx_tensor = torch.stack([frame_process_norm(ss) for ss in smplx_mesh_pils_origin], dim=0).cuda().permute(1, 0, 2, 3)

    ref_combine_blend_tensor = torch.cat([ref_frame_tensor.unsqueeze(1), blend_tensor[:, :-1]], dim=1)
    ref_combine_smplx_tensor = torch.cat([ref_frame_tensor.unsqueeze(1), smplx_tensor[:, :-1]], dim=1)

    # Image-to-video
    video = pipe(
        prompt=caption,
        negative_prompt="细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=ref_frame,
        ref_combine_blend_tensor=ref_combine_blend_tensor, 
        ref_combine_smplx_tensor=ref_combine_smplx_tensor, 
        num_inference_steps=50,
        cfg_scale=cfg_value, 
        seed=seed, 
        tiled=False, 
        height=H,
        width=W,
        tea_cache_l1_thresh=0.3 if use_teacache else None,
        tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,
    )


    save_video(ref_frame_pils_origin, [smplx_mesh_pils_origin[0]] + smplx_mesh_pils_origin[:-1], [blend_pils_origin[0]] + blend_pils_origin[:-1], video, save_video_path, fps=15)
