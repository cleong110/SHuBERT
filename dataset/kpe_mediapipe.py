import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import decord

from utils import load_file, is_string_in_file, dump_file_list

# Global placeholders (set during init)
face_detector = None
hand_detector = None
mp_holistic = None


def detect_holistic(image: np.ndarray):
    global mp_holistic, face_detector, hand_detector

    results = mp_holistic.process(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    face_prediction = face_detector.detect(mp_image)
    hand_prediction = hand_detector.detect(mp_image)

    bounding_boxes = {}
    landmarks_data = {}

    # Face landmarks
    if face_prediction.face_landmarks:
        bounding_boxes["#face"] = len(face_prediction.face_landmarks)
        landmarks_data["face_landmarks"] = [
            [[l.x, l.y, l.z] for l in face] for face in face_prediction.face_landmarks
        ]
    else:
        bounding_boxes["#face"] = 0
        landmarks_data["face_landmarks"] = None

    # Hand landmarks
    if hand_prediction.hand_landmarks:
        bounding_boxes["#hands"] = len(hand_prediction.hand_landmarks)
        landmarks_data["hand_landmarks"] = [
            [[l.x, l.y, l.z] for l in hand] for hand in hand_prediction.hand_landmarks
        ]
    else:
        bounding_boxes["#hands"] = 0
        landmarks_data["hand_landmarks"] = None

    # Pose landmarks
    if results.pose_landmarks:
        bounding_boxes["#pose"] = 1
        landmarks_data["pose_landmarks"] = [
            [[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]
        ]
    else:
        bounding_boxes["#pose"] = 0
        landmarks_data["pose_landmarks"] = None

    return bounding_boxes, landmarks_data


def video_holistic(
    video_file: Path, problem_file: Path, pose_path: Path, stats_path: Path
):
    try:
        video = decord.VideoReader(str(video_file))
    except Exception as e:
        print(f"Failed to read video {video_file}: {e}")
        problem_file.write_text(f"{video_file}\n", append=True)
        return

    result_dict, stats = {}, {}
    fps = video.get_avg_fps()

    pose_file = pose_path / f"{video_file.stem}_pose.json"
    stats_file = stats_path / f"{video_file.stem}_stats.json"

    for i in range(len(video)):
        try:
            frame = video[i].asnumpy()
            bounding_boxes, landmarks = detect_holistic(frame)
            result_dict[i] = landmarks
            stats[i] = bounding_boxes
        except Exception as e:
            print(f"Frame {i} failed: {e}")
            result_dict[i] = None

    pose_file.write_text(json.dumps(result_dict))
    stats_file.write_text(json.dumps(stats))


def initialize_models(face_model: Path, hand_model: Path):
    global face_detector, hand_detector, mp_holistic

    face_opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_model)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=6,
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_opts)

    hand_opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(hand_model)),
        num_hands=6,
        min_hand_detection_confidence=0.05,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

    mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract holistic landmarks from video batches."
    )
    parser.add_argument("--index", type=int, help="Batch index.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--time-limit", type=int, default=257890, help="Time limit in seconds."
    )
    parser.add_argument(
        "--files-list",
        type=Path,
        default=Path("bbox_clips/bbox_clips.list.gz"),
        help="Gzipped pickle list of files.",
    )
    parser.add_argument(
        "--pose-path",
        type=Path,
        default="kpe_poses",
        help="Directory to save pose JSONs.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default="kpe_stats",
        help="Directory to save stats JSONs.",
    )
    parser.add_argument(
        "--problem-log",
        type=Path,
        default="kpe_problem.log",
        help="File to log problematic videos.",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default="models/face_landmarker_v2_with_blendshapes.task",
        help="Face landmark model path.",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default="models/hand_landmarker.task",
        help="Hand landmark model path.",
    )
    parser.add_argument("--write-output-list", type=bool, default=True)
    args = parser.parse_args()

    # Prepare directories
    args.pose_path.mkdir(parents=True, exist_ok=True)
    args.stats_path.mkdir(parents=True, exist_ok=True)
    args.problem_log.touch(exist_ok=True)

    initialize_models(args.face_model, args.hand_model)
    video_files = load_file(args.files_list)

    video_batches = [
        video_files[i : i + args.batch_size]
        for i in range(0, len(video_files), args.batch_size)
    ]

    start_time = time.time()
    if args.index is None:
        # iterate over all batches
        batches = video_batches
    else:
        if args.index >= len(video_batches):
            print(
                f"Batch index {args.index} exceeds total batches {len(video_batches)}."
            )
            return
        batches = [video_batches[args.index]]

    for i, batch in enumerate(
        tqdm(batches, desc="Processing batches", disable=len(batches) <= 1)
    ):
        for video_file in tqdm(
            batch, desc=f"Processing videos in batch {i}", disable=len(batch) <= 1
        ):
            video_file = Path(video_file)
            elapsed = time.time() - start_time
            if elapsed > args.time_limit:
                print(f"Time limit reached after {elapsed:.2f}s.")
                break

            pose_file = args.pose_path / f"{video_file.stem}_pose.json"
            stats_file = args.stats_path / f"{video_file.stem}_stats.json"

            if pose_file.exists() and stats_file.exists():
                print(f"Skipping {video_file.name} (already processed).")
                continue
            if is_string_in_file(args.problem_log, str(video_file)):
                print(f"Skipping {video_file.name} (listed in problem log).")
                continue

            print(f"Processing {video_file.name}")
            video_holistic(
                video_file, args.problem_log, args.pose_path, args.stats_path
            )
    if args.write_output_list:
        dump_file_list(args.pose_path)


if __name__ == "__main__":
    main()
