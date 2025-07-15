import argparse
import gzip
import pickle
from pathlib import Path


def dump_file_list(input_dir: Path, output_file: Path) -> None:
    """Collect absolute file paths from input_dir and write them as a pickled gzip list to output_file."""
    input_files_list = [str(p.resolve()) for p in input_dir.iterdir() if p.is_file()]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_file, "wb") as f:
        pickle.dump(input_files_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Written {len(input_files_list)} file paths to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect filenames and write to pickled gzip file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("raw_vids"),
        help="Path to the input directory containing files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Path to the output directory to store clips_bbox.list.gz.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="clips_bbox.list.gz",
        help="Name of the output file (with .gz extension).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_file: Path = output_dir / args.output_filename

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    dump_file_list(input_dir, output_file)


if __name__ == "__main__":
    main()
