import os
import runpod
import uuid
import requests
import subprocess

def handler(event):
    inputs = event["input"]
    
    # Get inputs
    audio_url = inputs["audio_url"]
    video_url = inputs["video_url"]
    guidance_scale = str(inputs.get("guidance_scale", 2.0))
    inference_steps = str(inputs.get("inference_steps", 20))

    # Create temp file paths
    job_id = str(uuid.uuid4())
    audio_path = f"/tmp/audio_{job_id}.wav"
    video_path = f"/tmp/video_{job_id}.mp4"
    output_path = f"/tmp/output_{job_id}.mp4"

    # Download input files
    os.system(f"wget -O {audio_path} {audio_url}")
    os.system(f"wget -O {video_path} {video_url}")

    # Run inference command
    command = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--inference_steps", inference_steps,
        "--guidance_scale", guidance_scale,
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return { "error": f"Inference failed: {str(e)}" }

    # Upload to file.io
    try:
        with open(output_path, "rb") as f:
            response = requests.post("https://file.io", files={"file": f})
            if response.status_code == 200 and response.json().get("success"):
                return { "video_url": response.json()["link"] }
            else:
                return { "error": "Upload failed", "details": response.text }
    except Exception as e:
        return { "error": f"Upload failed: {str(e)}" }
