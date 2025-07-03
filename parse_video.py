import numpy as np
import os
import cv2
import torch
import argparse
from PIL import Image
import imageio
from tqdm import tqdm
import torch.distributed as dist

from engine.SegmentAPI.SAM import SAM2Seg



def video_to_pil_images(video_path, height=None, width=None):
    if video_path.endswith('.mp4'):
        cap = cv2.VideoCapture(video_path)
        pil_images = []
        while True:
            ret, frame = cap.read()  # 读取一帧
            if not ret:
                break  # 视频结束或读取失败
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            if height is not None and width is not None:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            pil_images.append(pil_image)
        cap.release()
    elif os.path.isdir(video_path):
        frame_files = sorted([os.path.join(video_path, x) for x in os.listdir(video_path) if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')])
        pil_images = []
        for frame in frame_files:
            frame = cv2.imread(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            if height is not None and width is not None:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            pil_images.append(pil_image)
    else:
        raise ValueError("Unsupported video format. Please provide a .mp4 file or a directory of images.")
    return pil_images



def save_video(pils, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in pils:
        all_frame = np.array(frame)
        writer.append_data(all_frame)
    writer.close()




def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # 添加分布式参数
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    args = parser.parse_args()

    parsingnet = SAM2Seg()

    all_videos = sorted(os.listdir(args.video_root))

    for video in tqdm(all_videos):
        video_path = os.path.join(args.video_root, video)
        save_path = os.path.join(args.save_root, video)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        frames = video_to_pil_images(video_path)

        masks = []

        for frame in frames:
            parsing_out = parsingnet._forward(np.array(frame), bbox=None)
            alpha = Image.fromarray((parsing_out.masks * 255).astype(np.uint8))
            masks.append(alpha)

        save_video(masks, save_path, fps=15, quality=9)

        # assert False