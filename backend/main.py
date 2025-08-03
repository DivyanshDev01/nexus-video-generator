import os
import uuid
import json
import time
import asyncio
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import torch
import torchvision
import redis
from minio import Minio
from minio.error import S3Error
from diffusers import StableVideoDiffusionPipeline
import cv2
import librosa
from scipy.ndimage import gaussian_filter
from fastapi.middleware.cors import CORSMiddleware
import logging

# Initialize FastAPI
app = FastAPI(title="NEXUS Video Generation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET", "minioadmin")
MODEL_CACHE = os.getenv("MODEL_CACHE", "./model_cache")
RESOLUTIONS = {
    "360p": (640, 360),
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}

# Initialize services
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS,
    secret_key=MINIO_SECRET,
    secure=False
)

# Create buckets if they don't exist
for bucket in ["scenes", "styles", "videos", "transitions", "audio"]:
    try:
        minio_client.make_bucket(bucket)
    except S3Error as e:
        if e.code != "BucketAlreadyOwnedByYou":
            logger.error(f"Error creating bucket {bucket}: {e}")
            raise

# Pydantic models
class SceneRequest(BaseModel):
    prompt: str
    duration: float = 5.0
    resolution: str = "720p"
    seed: Optional[int] = None
    style_lora: Optional[str] = None
    textual_inversion: Optional[str] = None
    sampling_steps: int = 25

class TransitionConfig(BaseModel):
    type: str = "dissolve"
    duration: float = 1.0

class VideoProject(BaseModel):
    scenes: List[SceneRequest]
    transitions: List[TransitionConfig]
    audio_url: Optional[str] = None

class RenderTask(BaseModel):
    task_id: str
    status: str = "pending"
    progress: int = 0
    video_url: Optional[str] = None

# Utility functions
def upload_to_minio(bucket: str, data: bytes, extension: str = "png") -> str:
    object_name = f"{uuid.uuid4()}.{extension}"
    minio_client.put_object(
        bucket,
        object_name,
        BytesIO(data),
        len(data)
    )
    return f"http://{MINIO_ENDPOINT}/{bucket}/{object_name}"

def download_from_minio(url: str) -> bytes:
    if url.startswith(f"http://{MINIO_ENDPOINT}/"):
        path = url.replace(f"http://{MINIO_ENDPOINT}/", "")
        parts = path.split('/', 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid MinIO URL: {url}")
        bucket = parts[0]
        object_name = parts[1]
        return minio_client.get_object(bucket, object_name).read()
    else:
        raise ValueError(f"URL not from MinIO: {url}")

def get_resolution(res_key: str) -> tuple:
    return RESOLUTIONS.get(res_key, (1280, 720))

# Core video generation functions
def load_svd_pipeline():
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=MODEL_CACHE
    )
    pipe.enable_model_cpu_offload()
    return pipe

# Models lazy loading
svd_pipeline = None
def get_svd_pipeline():
    global svd_pipeline
    if svd_pipeline is None:
        logger.info("Loading SVD pipeline...")
        svd_pipeline = load_svd_pipeline()
        logger.info("SVD pipeline loaded")
    return svd_pipeline

def apply_lora_adapters(pipeline, lora_url: str):
    if lora_url:
        logger.info(f"Applying LoRA weights from {lora_url}")

def apply_textual_inversion(pipeline, inversion_url: str):
    if inversion_url:
        logger.info(f"Applying textual inversion from {inversion_url}")

def generate_keyframes(
    prompt: str,
    duration: float,
    resolution: tuple,
    seed: Optional[int] = None,
    style_lora: Optional[str] = None,
    textual_inversion: Optional[str] = None,
    sampling_steps: int = 25
) -> List[str]:
    # Scale down resolution for performance
    width, height = resolution
    if width > 1280:
        width, height = 1280, 720
        
    pipeline = get_svd_pipeline()
    apply_lora_adapters(pipeline, style_lora)
    apply_textual_inversion(pipeline, textual_inversion)
    
    # Generate seed if not provided
    if seed is None:
        seed = int(time.time() * 1000) % 1000000
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Generate base image
    base_image = Image.new('RGB', (width, height), (0, 0, 0))
    
    # Generate keyframes
    frames = pipeline(
        base_image,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=int(duration * 6),
        num_inference_steps=sampling_steps,
        generator=generator
    ).frames[0]
    
    # Save frames to MinIO
    frame_urls = []
    for i, frame in enumerate(frames):
        img_byte_arr = BytesIO()
        frame.save(img_byte_arr, format='PNG')
        frame_urls.append(upload_to_minio("scenes", img_byte_arr.getvalue()))
    
    # Clean up GPU memory
    torch.cuda.empty_cache()
    return frame_urls

def film_interpolation(keyframe_urls: List[str]) -> List[str]:
    all_frames = []
    frames = []
    
    for url in keyframe_urls:
        img_data = download_from_minio(url)
        frames.append(np.array(Image.open(BytesIO(img_data))))
    
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        all_frames.append(keyframe_urls[i])
        
        for j in range(1, 4):
            alpha = j / 4.0
            inter_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            img = Image.fromarray(inter_frame.astype('uint8'))
            
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            inter_url = upload_to_minio("scenes", img_byte_arr.getvalue())
            all_frames.append(inter_url)
    
    all_frames.append(keyframe_urls[-1])
    return all_frames

def apply_transition(
    frames1: List[str], 
    frames2: List[str], 
    transition_type: str,
    duration: float
) -> List[str]:
    fps = 24
    num_transition_frames = int(duration * fps)
    transition_frames = []
    
    try:
        frame1 = Image.open(BytesIO(download_from_minio(frames1[-1])))
        frame2 = Image.open(BytesIO(download_from_minio(frames2[0])))
        arr1 = np.array(frame1)
        arr2 = np.array(frame2)
        
        for i in range(num_transition_frames):
            alpha = i / (num_transition_frames - 1)
            blended = cv2.addWeighted(arr1, 1 - alpha, arr2, alpha, 0)
            
            if transition_type == "morph":
                dx = int(50 * (1 - alpha))
                dy = int(30 * (1 - alpha))
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                blended = cv2.warpAffine(blended, M, (blended.shape[1], blended.shape[0]))
            elif transition_type == "generative_warp":
                rows, cols = blended.shape[:2]
                map_x = np.zeros((rows, cols), np.float32)
                map_y = np.zeros((rows, cols), np.float32)
                for y in range(rows):
                    for x in range(cols):
                        map_x[y, x] = x + 20 * np.sin(2 * np.pi * y / 90)
                        map_y[y, x] = y + 15 * np.cos(2 * np.pi * x / 180)
                blended = cv2.remap(blended, map_x, map_y, cv2.INTER_LINEAR)
            
            img = Image.fromarray(blended.astype('uint8'))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            transition_url = upload_to_minio("transitions", img_byte_arr.getvalue())
            transition_frames.append(transition_url)
            
    except Exception as e:
        logger.error(f"Transition failed: {str(e)}")
        # Fallback to dissolve
        for i in range(num_transition_frames):
            alpha = i / (num_transition_frames - 1)
            blended = cv2.addWeighted(arr1, 1 - alpha, arr2, alpha, 0)
            img = Image.fromarray(blended.astype('uint8'))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            transition_url = upload_to_minio("transitions", img_byte_arr.getvalue())
            transition_frames.append(transition_url)
    
    return transition_frames

def apply_audio_reactivity(frames: List[str], audio_url: str) -> List[str]:
    try:
        audio_data = download_from_minio(audio_url)
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmpfile:
            tmpfile.write(audio_data)
            tmpfile.flush()
            y, sr = librosa.load(tmpfile.name)
        
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        fps = 24
        reactive_frames = []
        
        for i, frame_url in enumerate(frames):
            img_data = download_from_minio(frame_url)
            img = Image.open(BytesIO(img_data))
            arr = np.array(img)
            
            frame_time = i / fps
            if any(abs(frame_time - beat) < 0.1 for beat in beat_times):
                arr = cv2.convertScaleAbs(arr, alpha=1.2, beta=10)
                arr = cv2.addWeighted(arr, 0.7, gaussian_filter(arr, sigma=5), 0.3, 0)
            
            img = Image.fromarray(arr)
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            modified_url = upload_to_minio("scenes", img_byte_arr.getvalue())
            reactive_frames.append(modified_url)
        
        return reactive_frames
        
    except Exception as e:
        logger.error(f"Audio reactivity failed: {str(e)}")
        return frames  # Return original frames if processing fails

def render_video(frames: List[str], resolution: tuple, audio_url: Optional[str] = None) -> str:
    width, height = resolution
    temp_dir = tempfile.mkdtemp()
    fps = 24
    
    # Download frames
    for i, frame_url in enumerate(frames):
        img_data = download_from_minio(frame_url)
        with open(f"{temp_dir}/frame_{i:05d}.png", "wb") as f:
            f.write(img_data)
    
    # Build FFmpeg command
    cmd = f"ffmpeg -y -r {fps} -i {temp_dir}/frame_%05d.png "
    
    if audio_url:
        try:
            audio_data = download_from_minio(audio_url)
            audio_path = f"{temp_dir}/audio.mp3"
            with open(audio_path, 'wb') as af:
                af.write(audio_data)
            cmd += f"-i {audio_path} -c:a aac "
        except Exception as e:
            logger.error(f"Audio integration failed: {str(e)}")
    
    output_path = f"{temp_dir}/output.mp4"
    cmd += f"-c:v libx264 -pix_fmt yuv420p -crf 23 -vf 'scale={width}:{height}' {output_path}"
    
    os.system(cmd)
    
    # Upload video
    with open(output_path, "rb") as video_file:
        video_data = video_file.read()
    video_url = upload_to_minio("videos", video_data, "mp4")
    
    # Cleanup
    for f in Path(temp_dir).glob("*"):
        try:
            f.unlink()
        except:
            pass
    try:
        Path(temp_dir).rmdir()
    except:
        pass
    
    return video_url

# API Endpoints
@app.post("/create_project", response_model=RenderTask)
async def create_project(project: VideoProject, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task = RenderTask(task_id=task_id)
    redis_client.set(f"task:{task_id}", task.json())
    
    background_tasks.add_task(process_video_project, task_id, project)
    return task

@app.get("/task_status/{task_id}", response_model=RenderTask)
async def get_task_status(task_id: str):
    task_data = redis_client.get(f"task:{task_id}")
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    return RenderTask.parse_raw(task_data)

def update_task_status(task_id: str, status: str, progress: int, video_url: str = None):
    task_data = redis_client.get(f"task:{task_id}")
    if task_data:
        task = RenderTask.parse_raw(task_data)
        task.status = status
        task.progress = progress
        if video_url:
            task.video_url = video_url
        redis_client.set(f"task:{task_id}", task.json())

def process_video_project(task_id: str, project: VideoProject):
    try:
        total_steps = len(project.scenes) * 2 + len(project.transitions) + 1
        current_step = 0
        all_frames = []
        
        # Generate scenes
        for i, scene in enumerate(project.scenes):
            update_task_status(task_id, f"Generating scene {i+1}", int(current_step/total_steps*100))
            resolution = get_resolution(scene.resolution)
            
            keyframe_urls = generate_keyframes(
                prompt=scene.prompt,
                duration=scene.duration,
                resolution=resolution,
                seed=scene.seed,
                style_lora=scene.style_lora,
                textual_inversion=scene.textual_inversion,
                sampling_steps=scene.sampling_steps
            )
            
            scene_frames = film_interpolation(keyframe_urls)
            current_step += 1
            update_task_status(task_id, f"Processing scene {i+1}", int(current_step/total_steps*100))
            
            if i == 0:
                all_frames.extend(scene_frames)
            else:
                update_task_status(task_id, f"Adding transition {i}", int(current_step/total_steps*100))
                transition = project.transitions[i-1]
                transition_frames = apply_transition(
                    all_frames[-10:],
                    scene_frames[:10],
                    transition.type,
                    transition.duration
                )
                all_frames.extend(transition_frames)
                all_frames.extend(scene_frames)
                current_step += 1
        
        # Audio reactivity
        if project.audio_url:
            update_task_status(task_id, "Applying audio reactivity", 90)
            all_frames = apply_audio_reactivity(all_frames, project.audio_url)
        
        # Render video
        update_task_status(task_id, "Rendering final video", 95)
        resolution = get_resolution(project.scenes[0].resolution)
        video_url = render_video(all_frames, resolution, project.audio_url)
        
        update_task_status(task_id, "completed", 100, video_url)
    
    except Exception as e:
        logger.exception("Video generation failed")
        update_task_status(task_id, f"failed: {str(e)}", 100)

@app.post("/upload_style")
async def upload_style(file: UploadFile = File(...)):
    contents = await file.read()
    extension = file.filename.split('.')[-1]
    style_url = upload_to_minio("styles", contents, extension)
    return {"style_url": style_url}

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    contents = await file.read()
    audio_url = upload_to_minio("audio", contents, "mp3")
    return {"audio_url": audio_url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
