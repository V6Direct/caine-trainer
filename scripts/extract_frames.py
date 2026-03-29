import ffmpeg
import cv2
import os
from pathlib import Path
import json
import cv2
from pathlib import Path
import random  # Falls nicht da

def extract_keyframes(video_path, num_frames=200):
    """OpenCV-only, kein FFmpeg nötig."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    
    step = max(1, total_frames // num_frames)
    frames = []
    
    # Output dir pro Video
    output_dir = Path("data/frames") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)  # ← FIX: parents=True
    
    for i in range(num_frames):
        frame_pos = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            frame_path = output_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frames.append(str(frame_path))
    
    cap.release()
    print(f"✅ {len(frames)} frames from {video_path}")
    return frames

def enrich_dataset_with_video_frames():
    frames_db = {}
    video_dir = Path("data/videos")
    
    if not video_dir.exists():
        print("❌ No data/videos/ folder - creating dummy frames")
        create_dummy_frames()
        return
    
    for video_file in video_dir.glob("*.mp4"):
        ep_name = video_file.stem
        frames = extract_keyframes(video_file, num_frames=100)
        frames_db[ep_name] = frames
    
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