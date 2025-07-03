import os
import cv2
import argparse
from tqdm import tqdm

def extract_frames(input_dir, output_dir, fps=6):
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted(os.listdir(input_dir))
    for vid_idx, video_fname in enumerate(video_files, start=1):
        name, ext = os.path.splitext(video_fname)
        video_path = os.path.join(input_dir, video_fname)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Не удалось открыть {video_path}")
            continue
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(int(orig_fps // fps), 1)
        out_subdir = os.path.join(output_dir, name)
        os.makedirs(out_subdir, exist_ok=True)
        saved = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(total_frames), desc=name):
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0:
                fname = os.path.join(
                    out_subdir,
                    f"frame{vid_idx}_{saved:06d}.jpg"
                )
                cv2.imwrite(fname, frame)
                saved += 1
        cap.release()
        print(f"{name}: extracted {saved} frames into {out_subdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, default="data/raw_videos",
        help="Папка с исходными видео"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/frames",
        help="Куда сохранять кадры"
    )
    parser.add_argument(
        "--fps", type=int, default=6,
        help="Сколько кадров в секунду извлекать"
    )
    args = parser.parse_args()

    extract_frames(args.input, args.output, args.fps)
