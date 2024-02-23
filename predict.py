# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import cv2
import time
import torch
import subprocess
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import pipeline
from torchvision import transforms

def generate_stereo(left_img, depth, ipd):
    monitor_w = 38.5
    h, w, _ = left_img.shape
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    right = np.zeros_like(left_img)
    deviation_cm = ipd * 0.12
    deviation = deviation_cm * monitor_w * (w / 1920)
    col_r_shift = (1 - depth_normalized ** 2) * deviation
    col_r_indices = np.arange(w) - col_r_shift.astype(int)
    valid_indices = col_r_indices >= 0
    for row in range(h):
        valid_cols = col_r_indices[row, valid_indices[row]]
        right[row, valid_cols] = left_img[row, np.arange(w)[valid_indices[row]]]

    right_fix = right.copy()
    gray = cv2.cvtColor(right_fix, cv2.COLOR_BGR2GRAY)
    missing_pixels = np.where(gray == 0)
    for row, col in zip(*missing_pixels):
        for offset in range(1, int(deviation)):
            r_offset = min(col + offset, w - 1)
            l_offset = max(col - offset, 0)
            if not np.all(right_fix[row, r_offset] == 0):
                right_fix[row, col] = right_fix[row, r_offset]
                break
            elif not np.all(right_fix[row, l_offset] == 0):
                right_fix[row, col] = right_fix[row, l_offset]
                break

    return right_fix

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

    @torch.no_grad()
    def predict_depth(self, model, image):
        return model(image)["depth"]

    def predict(
        self,
        video: Path = Input(description="Input video"),
        encoder: str = Input(description="Model type", default="vits", choices=["vits", "vitb", "vitl"]),
        ipd: float = Input(description="Interpupillary distance", default=6.34),
    ) -> Path:
        """Run a single prediction on the model"""
        t1 = time.time()
        frames_path = "/tmp/frames"
        depth_path = "/tmp/depth"
        stereo_path = "/tmp/stereo"
        mapper = {"vits":"small","vitb":"base","vitl":"large"}
        subprocess.run(["mkdir", "-p", frames_path], check=True)
        subprocess.run(["mkdir", "-p", depth_path], check=True)
        subprocess.run(["mkdir", "-p", stereo_path], check=True)
        to_tensor_transform = transforms.ToTensor()

        depth_anything = pipeline(
            task = "depth-estimation",
            model=f"nielsr/depth-anything-{mapper[encoder]}",
            device=0
        )
        filename = str(video)
        print('Processing depth frames', filename)
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        filename = os.path.basename(filename)
        count = 0
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            frame_pil =  Image.fromarray((frame * 255).astype(np.uint8))
            depth = to_tensor_transform(self.predict_depth(depth_anything, frame_pil))
            depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_map = depth.cpu().numpy().astype(np.uint8)
            left_img = np.array(frame_pil)
            depth_map = cv2.blur(depth_map, (3, 3))
            right_img = generate_stereo(left_img, depth_map, ipd)
            stereo = np.hstack([left_img, right_img])
            stereo_bgr = cv2.cvtColor(stereo, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{stereo_path}/{count}.jpg", stereo_bgr)
            count += 1

        raw_video.release()
        # Convert all images in frames folder to a video
        subprocess.run(["ffmpeg", "-y", "-r", str(frame_rate), "-i", "/tmp/stereo/%d.jpg", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "/tmp/output.mp4"], check=True)
        t2 = time.time()
        print(f"Time taken for the whole run: {t2-t1} seconds")
        return Path("/tmp/output.mp4")
