import argparse
from pathlib import Path
from utils import dump_file_list


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
        # default=Path.cwd(),
        help="Path to the output directory to store the .list.gz.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        help="Name of the output file (with .gz extension).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir: Path = args.input_dir
    if args.output_dir is None:
        output_dir = input_dir  # make it in the input dir
    else:
        output_dir = args.output_dir

    if args.output_filename is None:
        output_file = output_dir / f"{output_dir.name}.list.gz"
    else:
        output_file = output_dir / args.output_filename

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    dump_file_list(input_dir, output_file)


if __name__ == "__main__":
    main()
