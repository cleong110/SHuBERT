import argparse

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import logging

from utils import load_file, is_string_in_file, dump_file_list

logger = logging.getLogger(__name__)


def get_optical_flow(images):
    prv_gray = None
    motion_mags = []
    for frame in images:
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.resize(cur_gray, (frame.shape[1] // 2, frame.shape[0] // 2))
        if prv_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prv_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag = (
                255.0 * (mag - mag.min()) / max(float(mag.max() - mag.min()), 1)
            ).astype(np.uint8)
            mag = cv2.resize(mag, (frame.shape[1], frame.shape[0]))
        else:
            mag = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        prv_gray = cur_gray
        motion_mags.append(mag)
    return motion_mags


def get_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


def find_target_bbox(bbox_arr, opts, iou_thr=0.5, len_ratio_thr=0.5):
    tubes = []
    num_rest = sum(len(x) for x in bbox_arr)
    while num_rest > 0:
        for i, bboxes in enumerate(bbox_arr):
            if bboxes:
                anchor = [i, bbox_arr[i].pop()]
                break
        tube = [anchor]
        for i, bboxes in enumerate(bbox_arr):
            if i == anchor[0] or not bboxes:
                continue
            ious = np.array([get_iou(anchor[1], bbox) for bbox in bboxes])
            j = ious.argmax()
            if ious[j] > iou_thr:
                tube.append([i, bboxes.pop(j)])
        tubes.append(tube)
        num_rest = sum(len(x) for x in bbox_arr)

    max_val, max_tube = -1, None
    for tube in tubes:
        mean_val = sum(
            opts[iframe][
                max(0, int(bbox[1])) : int(bbox[3]), max(0, int(bbox[0])) : int(bbox[2])
            ].mean()
            for iframe, bbox in tube
        ) / len(tube)
        if len(tube) / len(opts) > len_ratio_thr and mean_val > max_val:
            max_val, max_tube = mean_val, tube

    if max_tube:
        target_bbox = np.mean([bbox for _, bbox in max_tube], axis=0).tolist()
    else:
        target_bbox = None
    return target_bbox, tubes


def crop_clip(
    video_path: Path,
    problem_file_path: Path,
    crop_clip_path: Path,
    yolo_model_path: Path,
):
    """
    Detects persons in a video, crops the video around detected bounding boxes, and saves it.
    Logs problematic files to a specified path if processing fails.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        model = YOLO(str(yolo_model_path))

        frames, bboxes = [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            person_bboxes = [
                box.xyxy[0].tolist() for r in results for box in r.boxes if box.cls == 0
            ]
            frames.append(frame)
            bboxes.append(person_bboxes)

        cap.release()

        if not frames:
            logger.warning(f"No frames read from {video_path}, skipping.")
            return

        expansions = (0.01, 0.01, 0.3, 0.3)  # up, down, left, right

        # Expand all bounding boxes
        for frame_bboxes in bboxes:
            for idx, (x0, y0, x1, y1) in enumerate(frame_bboxes):
                w, h = x1 - x0 + 1, y1 - y0 + 1
                frame_bboxes[idx] = [
                    x0 - w * expansions[2],
                    y0 - h * expansions[0],
                    x1 + w * expansions[3],
                    y1 + h * expansions[1],
                ]

        if max(len(x) for x in bboxes) == 1:
            # Single person consistently detected
            bbox = np.mean([b[0] for b in filter(None, bboxes)], axis=0).tolist()
        else:
            # Multi-person — use optical flow to track
            opts = get_optical_flow(frames)
            bbox, tubes = find_target_bbox(bboxes, opts)

            if bbox is None and tubes:
                largest_tube = max(
                    tubes,
                    key=lambda t: sum((b[3] - b[1]) * (b[2] - b[0]) for _, b in t),
                )
                bbox = np.mean([b for _, b in largest_tube], axis=0).tolist()

        if not bbox:
            logger.warning(f"No valid bbox found for {video_path}, skipping.")
            return

        x0, y0 = max(int(bbox[0]), 0), max(int(bbox[1]), 0)
        x1, y1 = min(int(bbox[2]), width), min(int(bbox[3]), height)

        cropped_frames = [frame[y0:y1, x0:x1] for frame in frames]
        out_size = (cropped_frames[0].shape[1], cropped_frames[0].shape[0])

        writer = cv2.VideoWriter(str(crop_clip_path), fourcc, fps, out_size)
        for frame in cropped_frames:
            writer.write(frame)
        writer.release()

        logger.info(f"Processed and saved cropped video: {crop_clip_path}")

    except (cv2.error, RuntimeError, Exception) as e:
        logger.error(f"Error processing {video_path}: {e}")
        problem_file_path.parent.mkdir(parents=True, exist_ok=True)
        with problem_file_path.open("a") as f:
            f.write(f"{video_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Crop clips to bounding box of signer using YOLOv8."
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Index of batch to process."
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size.")
    parser.add_argument(
        "--files-list",
        type=Path,
        default=Path("raw_vids/raw_vids.list.gz"),
        help="Path to input file list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bbox_clips"),
        help="Directory to save cropped clips.",
    )
    parser.add_argument(
        "--problem-log",
        type=Path,
        default=Path("problem_log.log"),
        help="File to log problematic clips.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("models/yolov8n.pt"),
        help="Path to YOLO model.",
    )
    parser.add_argument("--write-output-list", type=bool, default=True)
    args = parser.parse_args()

    files = load_file(args.files_list)
    batches = [
        files[i : i + args.batch_size] for i in range(0, len(files), args.batch_size)
    ]

    if args.index >= len(batches):
        print(f"Index {args.index} out of range (max {len(batches) - 1})")
        return

    batch = batches[args.index]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for video_path in batch:
        video_path = Path(video_path)
        output_clip_path = args.output_dir / video_path.name

        if output_clip_path.exists():
            print(f"Skipping {video_path} (already processed).")
            continue
        if args.problem_log.exists() and is_string_in_file(
            args.problem_log, str(video_path)
        ):
            print(f"Skipping {video_path} (logged as problematic).")
            continue

        print(f"Processing {video_path}...")
        crop_clip(video_path, args.problem_log, output_clip_path, args.yolo_model)

    print(f"Finished batch {args.index} in {time.time() - start_time:.2f}s.")
    if args.write_output_list:
        dump_file_list(args.output_dir)


if __name__ == "__main__":
    main()
