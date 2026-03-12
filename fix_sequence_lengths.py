import configparser
from pathlib import Path
import pandas as pd


def get_last_frame(gt_path):
    """
    Read MOT gt.txt and return the maximum frame number.
    """
    df = pd.read_csv(gt_path, header=None)
    last_frame = int(df[0].max())  # column 0 = frame index
    return last_frame


def update_seqinfo(seqinfo_path, new_length):
    """
    Update seqLength in seqinfo.ini
    """
    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_path)

    if "Sequence" not in cfg:
        raise ValueError(f"[Sequence] section missing in {seqinfo_path}")

    cfg["Sequence"]["seqLength"] = str(new_length)

    with open(seqinfo_path, "w") as f:
        cfg.write(f)


def process_sequence(seq_dir):
    """
    Update seqLength for a single MOT sequence.
    """
    seq_dir = Path(seq_dir)

    gt_path = seq_dir / "gt" / "gt.txt"
    seqinfo_path = seq_dir / "seqinfo.ini"

    if not gt_path.exists():
        print(f"Skipping {seq_dir} (no gt.txt)")
        return

    if not seqinfo_path.exists():
        print(f"Skipping {seq_dir} (no seqinfo.ini)")
        return

    last_frame = get_last_frame(gt_path)
    update_seqinfo(seqinfo_path, last_frame)

    print(f"{seq_dir.name}: seqLength updated → {last_frame}")


def process_dataset(root_dir):
    """
    Process all sequences inside a dataset directory.
    """
    root = Path(root_dir)

    for seq_dir in root.iterdir():
        if seq_dir.is_dir():
            process_sequence(seq_dir)


if __name__ == "__main__":
    # Example usage
    dataset_root = "gen/train"

    process_dataset(dataset_root)