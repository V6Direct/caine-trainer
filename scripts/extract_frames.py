import ffmpeg
import cv2
import os
from pathlib import Path
import json
import random

def extract_keyframes(video_path, output_dir, num_frames=100, fps=1):
    """Extrahiert num_frames gleichmäßig verteilte Keyframes [web:37]."""
    probe = ffmpeg.probe(video_path)
    duration = float(probe['streams'][0]['duration'])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    intervals = duration / num_frames
    frames = []
    for i in range(num_frames):
        timestamp = i * intervals
        frame_path = output_dir / f"frame_{i:04d}.jpg"
        (
            ffmpeg
            .input(video_path, ss=timestamp)
            .filter('scale', 1024, -1)  # Resize für VLM
            .output(str(frame_path), vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )
        frames.append(str(frame_path))
    return frames

def enrich_dataset_with_video_frames(jsonl_path="data/tadc_dataset.jsonl", video_dir="videos/", output_path="data/video_vision_tadc.jsonl"):
    """Fügt Frames zu deinem Dataset hinzu."""
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    episode_frames = {}
    for video_file in Path(video_dir).glob("tadc_episode_*.mp4"):
        ep_name = video_file.stem
        frames = extract_keyframes(video_file, f"data/frames/{ep_name}", num_frames=200)
        episode_frames[ep_name] = frames
    
    enriched = []
    for item in data:
        # Wähle passende Frames (z.B. random oder timestamp-basiert)
        ep_frames = random.sample(episode_frames.get("pilot", []), k=min(3, len(episode_frames.get("pilot", []))))
        user_content = [{"type": "image_url", "image_url": {"url": f}} for f in ep_frames] + [item["input"]]
        
        enriched.append({
            "messages": [
                {"role": "system", "content": "Du bist Caine. Analysiere Bilder aus dem Circus und antworte theatralisch!"},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["output"]}
            ]
        })
    
    with open(output_path, 'w') as f:
        for item in enriched:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Video-Vision-Dataset: {len(enriched)} Beispiele mit Frames generiert!")

if __name__ == "__main__":
    enrich_dataset_with_video_frames()