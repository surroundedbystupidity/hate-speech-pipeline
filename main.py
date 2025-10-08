import argparse
import logging
from math import e

from hate_speech_pipeline.driver import run

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DCRNN Hate Speech Pipeline")
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Don't assume embeddings are available, generate them.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="diffusion",
        help="Mode to run the model in: diffusion or classification.",
    )
    parser.add_argument(
        "--train-file-path",
        type=str,
        default="val_dataset_with_emb.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--test-file-path",
        type=str,
        default="test_dataset_with_emb.csv",
        help="Path to the test CSV file.",
    )
    parser.add_argument(
        "--val-file-path",
        type=str,
        default="val_dataset_with_emb.csv",
        help="Path to the validation CSV file.",
    )
    parser.add_argument(
        "--subset-count",
        type=int,
        default=0,
        help="Number of samples to use from the dataset.",
    )
    parser.add_argument(
        "--window-size-hours",
        type=int,
        default=1,
        help="Number of hours to use for snapshot window.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    args = parser.parse_args()
    logger.info(
        "Arguments: %s",
        {
            "mode": args.mode,
            "generate_embeddings": args.generate_embeddings,
            "train_file_path": args.train_file_path,
            "val_file_path": args.val_file_path,
            "test_file_path": args.test_file_path,
            "subset_count": args.subset_count,
            "window_size_hours": args.window_size_hours,
            "epochs": args.epochs,
        },
    )
    run(
        generate_embeddings=args.generate_embeddings,
        train_file_path=args.train_file_path,
        test_file_path=args.test_file_path,
        val_file_path=args.val_file_path,
        subset_count=args.subset_count,
        window_size_hours=args.window_size_hours,
        epochs=args.epochs,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
