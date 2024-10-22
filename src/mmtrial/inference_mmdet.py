"""Demo script to test the model using mmdet 3.3.0.

Ref: https://github.com/open-mmlab/mmdetection/issues/10829
"""

import argparse
import os
from itertools import chain

import pandas as pd
from loguru import logger
from mmdet.apis import DetInferencer


def get_image_groups(
    folder: str,
    n: int,
    extensions: tuple,
) -> list[list[str]]:
    """Gets images from a folder and divides them into groups.

    Args:
        folder (str): The folder containing the images.
        n (int): The maximum number of images in each group.
        extensions (tuple): The valid image extensions.

    Returns:
        list[list[str]]: A list of `M` groups, each containing up to `n` image
          paths. Returns an empty list if no images are found.
    """
    if not os.path.isdir(folder):
        logger.error(f"The folder '{folder}' does not exist.")
        raise ValueError()

    # Gather all image files with the specified extensions
    image_paths = [
        os.path.join(root, file) for root, _, files in os.walk(folder)
        for file in files if file.lower().endswith(extensions)
    ]

    if not image_paths:
        logger.warning("No images found in the folder "
                       f"'{folder}' with extensions {extensions}.")
        return []

    # Split the list of images into chunks of size `n`
    return [image_paths[i:i + n] for i in range(0, len(image_paths), n)]


def infer(inferencer: DetInferencer, imgs: list[str]) -> pd.DataFrame:
    """Infers on the given images.

    Args:
        inferencer (DetInferencer): The inference object.
        imgs (list[str]): List of `N` image paths.

    Returns:
        pd.DataFrame: Dataframe containing the inference results
    """
    pred = inferencer(imgs)["predictions"]
    seq_n = [len(p["labels"]) for p in pred]
    seq_score = [p["scores"] for p in pred]
    seq_bboxe = [p["bboxes"] for p in pred]
    seq_label = [p["labels"] for p in pred]

    bboxes = list(chain(*seq_bboxe))
    images = list(chain(*[[i] * n for i, n in zip(imgs, seq_n, strict=True)]))
    scores = list(chain(*seq_score))
    x1 = [bbx[0] for bbx in bboxes]
    y1 = [bbx[1] for bbx in bboxes]
    x2 = [bbx[2] for bbx in bboxes]
    y2 = [bbx[3] for bbx in bboxes]
    labels = list(chain(*seq_label))
    return pd.DataFrame({
        "image": images,
        "score": scores,
        "label": labels,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    })


def main(  # noqa: PLR0913
    img_dir: str,
    tsv_path: str,
    cfg_path: str,
    ckpt_path: str,
    device: str,
    *,
    n: int = 10,
) -> None:
    """Main function to perform inference.

    Args:
        img_dir (str): Folder containing the images to run inference on.
        tsv_path (str): Path to the output TSV file.
        cfg_path (str): Path to the model configuration file.
        ckpt_path (str): Path to the model checkpoint file.
        device (str): Device to run the inference on.
        n (int): Maximum number of images in each group. Defaults to 10.
    """
    # Get the base directory of the current script
    imgs = get_image_groups(img_dir, n, (".jpg", ".jpeg"))

    if not imgs:
        return

    inferencer = DetInferencer(cfg_path, ckpt_path, device=device)

    for i, grp in enumerate(imgs):
        logger.info(f"Processing image group {i + 1}...")
        df = infer(inferencer, grp)  # noqa: PD901
        df.to_csv(tsv_path, sep="\t", index=False, mode="a", header=not i)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run inference using MMDetection.")

    parser.add_argument(
        "img_dir",
        type=str,
        help="Folder containing the images to run inference on",
    )
    parser.add_argument(
        "tsv_path",
        type=str,
        help="Path to the output TSV file",
    )
    parser.add_argument(
        "cfg_path",
        type=str,
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "ckpt_path",
        type=str,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the inference on (default: cuda:0)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Maximum number of images in each group (default: 10)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        args.img_dir,
        args.tsv_path,
        args.cfg_path,
        args.ckpt_path,
        args.device,
        n=args.n,
    )
