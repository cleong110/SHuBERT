import torch
import numpy as np
from pathlib import Path
import argparse
from typing import List
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "fairseq"))

from examples.shubert.models.shubert import SHubertModel, SHubertConfig


def load_model(checkpoint_path: Path) -> SHubertModel:
    """Load SHubert model from checkpoint."""
    cfg = SHubertConfig()
    model = SHubertModel(cfg)
    checkpoint = torch.load(checkpoint_path)

    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()
    return model


def process_sample(
    model: SHubertModel,
    face_path: Path,
    left_hand_path: Path,
    right_hand_path: Path,
    body_posture_path: Path,
) -> np.ndarray:
    """Process a single sample and extract features."""

    face = torch.from_numpy(np.load(face_path)).float().cuda()
    left_hand = torch.from_numpy(np.load(left_hand_path)).float().cuda()
    right_hand = torch.from_numpy(np.load(right_hand_path)).float().cuda()
    body_posture = torch.from_numpy(np.load(body_posture_path)).float().cuda()

    length = face.shape[0]
    source = [
        {
            "face": face,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "body_posture": body_posture,
            "label_face": torch.zeros((length, 1), device="cuda"),
            "label_left_hand": torch.zeros((length, 1), device="cuda"),
            "label_right_hand": torch.zeros((length, 1), device="cuda"),
            "label_body_posture": torch.zeros((length, 1), device="cuda"),
        }
    ]

    with torch.no_grad():
        result = model.extract_features(
            source, padding_mask=None, kmeans_labels=None, mask=False
        )

    layer_outputs = [
        layer[-1].squeeze(1).cpu().numpy() for layer in result["layer_results"]
    ]
    features = np.stack(layer_outputs, axis=0)
    # print(f"Extracted features shape: {features.shape}")
    return features


def process_batch(
    csv_list: List[List[str]], model: SHubertModel, output_dir: Path
) -> None:
    """Process a batch of samples and save features."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for row in csv_list:
        cues = row[0].split("\t")
        face_path, left_hand_path, right_hand_path, body_posture_path = map(Path, cues)

        output_filename = f"{face_path.stem.rsplit('_', 1)[0]}.npy"
        output_path = output_dir / output_filename

        if output_path.exists():
            print(f"Skipping {output_path} (already exists)")
            continue

        features = process_sample(
            model, face_path, left_hand_path, right_hand_path, body_posture_path
        )
        np.save(output_path, features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=int,
        help="Index of the sublist to process (optional, runs all if unset)",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        help="Path to CSV file (optional, builds from directories if unset)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("models/SHuBERT_ckpts/checkpoint_836_400000.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./shubert_features"),
        help="Directory to save features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for splitting CSV list (default 1000)",
    )
    args = parser.parse_args()

    if args.csv_path is None:
        print("No CSV provided, building file list from feature directories...")

        face_paths = list(Path("dinov2_face_embeddings").glob("*.npy"))
        hand_paths = list(Path("dinov2_hand_embeddings").glob("*.npy"))
        body_paths = list(Path("body_features").glob("*.npy"))

        print(
            f"Found \n* {len(face_paths)} face paths\n* {len(hand_paths)} hand\n* {len(body_paths)}body"
        )

        # Index by video stem (strip _hand1/_hand2 and extension)
        hand_map = {}
        for p in hand_paths:
            stem = p.stem
            base_stem = stem.rsplit("_hand", 1)[0]
            if base_stem not in hand_map:
                hand_map[base_stem] = {}
            if stem.endswith("hand1"):
                hand_map[base_stem]["left"] = str(p)
            elif stem.endswith("hand2"):
                hand_map[base_stem]["right"] = str(p)

        face_map = {p.stem.split("_face")[0]: str(p) for p in face_paths}
        body_map = {p.stem.split("_pose")[0]: str(p) for p in body_paths}

        # print(json.dumps(hand_map, indent=2))
        # print(json.dumps(face_map, indent=2))
        # print(json.dumps(body_map, indent=2))

        # Use base_stem for matching
        common_keys = set(face_map) & set(body_map) & set(hand_map)

        # Only keep samples that have both hands
        valid_keys = [
            k for k in common_keys if "left" in hand_map[k] and "right" in hand_map[k]
        ]

        all_rows = [
            [
                f"{face_map[k]}\t{hand_map[k]['left']}\t{hand_map[k]['right']}\t{body_map[k]}"
            ]
            for k in sorted(valid_keys)
        ]

        print(f"Found {len(all_rows)} samples with face, both hands, and body cues.")

    batches = [
        all_rows[i : i + args.batch_size]
        for i in range(0, len(all_rows), args.batch_size)
    ]

    model = load_model(args.checkpoint_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.index is not None:
        print(
            f"Processing batch index {args.index} ({len(batches[args.index])} samples)"
        )
        process_batch(batches[args.index], model, args.output_dir)
    else:
        for idx, batch in tqdm(enumerate(batches), desc="Processing batches"):
            print(f"Batch {idx + 1}/{len(batches)} ({len(batch)} samples)")
            process_batch(batch, model, args.output_dir)


if __name__ == "__main__":
    main()
