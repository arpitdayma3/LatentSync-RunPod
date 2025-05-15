import os
import runpod
import uuid
from LatentSync.inference.infer import infer_video

def handler(event):
    inputs = event["input"]
    
    audio_url = inputs["audio_url"]
    video_url = inputs["video_url"]
    seed = inputs.get("seed", 42)
    guidance_scale = inputs.get("guidance_scale", 7.5)

    job_id = str(uuid.uuid4())
    audio_path = f"/tmp/audio_{job_id}.wav"
    video_path = f"/tmp/video_{job_id}.mp4"
    output_path = f"/tmp/output_{job_id}.mp4"

    os.system(f"wget -O {audio_path} {audio_url}")
    os.system(f"wget -O {video_path} {video_url}")

    infer_video(
        audio_path=audio_path,
        video_path=video_path,
        output_path=output_path,
        seed=seed,
        guidance_scale=guidance_scale,
        unet_path='LatentSync/checkpoints/latentsync_unet.pt',
        whisper_path='LatentSync/checkpoints/whisper/tiny.pt',
        device="cuda"
    )

    return {
        "output_video_path": output_path
    }
