import random
import time
import os
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import imageio
import torch

from lib.data_loader import data_loader
from lib.bbox import compute_bounding_box_frustrum_cam, compute_bounding_box_coarse
from lib.dvgo import DirectVoxGO, scene_rep_reconstruction
from lib.rendering import render_viewpoints
from lib.train_levels import (
    coarse_train,
    fine_train,
    coarse_model_and_render,
    fine_model_and_render,
)


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def format_time(seconds):
    return f"{int(seconds // 3600):02}:{int(seconds % 3600 // 60):02}:{int(seconds % 60):02}"


def setup_experiment_directory(path, args):
    os.makedirs(path, exist_ok=True)
    args_path = path / "args.txt"
    with args_path.open("w") as file:
        for arg, value in sorted(vars(args).items()):
            file.write(f"{arg} = {value}\n")


def run_reconstruction(stage, bbox_compute_func, model_attrs, train_attrs, **kwargs):
    start_time = time.time()
    xyz_min, xyz_max = bbox_compute_func(**kwargs)
    scene_rep_reconstruction(
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        model_attrs=model_attrs,
        train_attrs=train_attrs,
        **kwargs,
        stage=stage,
    )
    print(f"train: {stage} reconstruction in {format_time(time.time() - start_time)}")


def train(args, data, exp_path, device, coarse_ckpt_path=None):
    print("train: start")
    setup_experiment_directory(exp_path, args)

    # Coarse geometry reconstruction
    run_reconstruction(
        stage="coarse",
        bbox_compute_func=lambda **kwargs: compute_bounding_box_frustrum_cam(
            HW=data["HW"],
            Ks=data["Ks"],
            poses=data["poses"],
            i_train=data["i_train"],
            near=data["near"],
            far=data["far"],
        ),
        model_attrs=coarse_model_and_render,
        train_attrs=coarse_train,
        args=args,
        data_dict=data,
        exp_path=exp_path,
        device=device,
    )

    # Fine detail reconstruction
    run_reconstruction(
        stage="fine",
        bbox_compute_func=lambda **kwargs: compute_bounding_box_coarse(
            model_class=DirectVoxGO,
            model_attrs=fine_model_and_render,
            model_path=coarse_ckpt_path,
        ),
        model_attrs=fine_model_and_render,
        train_attrs=fine_train,
        args=args,
        data_dict=data,
        exp_path=exp_path,
        coarse_ckpt_path=coarse_ckpt_path,
    )

    total_time = format_time(time.time() - exp_path.start_time)
    print(f"train: finish (total time {total_time})")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--object", type=str, help="Object to train on")
    return parser.parse_args()


def setup_paths_and_seed(object_name, base_path="./data/"):
    BASE_PATH = Path(base_path)
    DATA_PATH = BASE_PATH / object_name
    EXP_PATH = BASE_PATH / "results" / object_name
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    return BASE_PATH, DATA_PATH, EXP_PATH


def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_class = DirectVoxGO
    model = model_class(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, device


def render_and_save_views(data, model, device, exp_path, args):
    render_kwargs = {
        "near": data["near"],
        "far": data["far"],
        "bg": 1,
        "render_depth": True,
        "model": model,
        "device": device,
    }

    for subset in ["train", "test"]:
        indices_key = f"i_{subset}"
        poses = data["poses"][data[indices_key]]
        HW = data["HW"][data[indices_key]]
        Ks = data["Ks"][data[indices_key]]
        gt_imgs = [data["images"][i].cpu().numpy() for i in data[indices_key]]

        savedir = exp_path / f"render_{subset}_{model.ckpt_name}"
        os.makedirs(savedir, exist_ok=True)

        print(f"All results are dumped into {savedir}")
        rgbs, depths, bgmaps = render_viewpoints(
            render_poses=poses,
            HW=HW,
            Ks=Ks,
            gt_imgs=gt_imgs,
            savedir=savedir,
            **render_kwargs,
        )

        # Save RGB and depth videos
        video_path_rgb = savedir / "video.rgb.mp4"
        video_path_depth = savedir / "video.depth.mp4"
        imageio.mimwrite(video_path_rgb, to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(
            video_path_depth, to8b(1 - depths / np.max(depths)), fps=30, quality=8
        )


def main():
    args = parse_arguments()
    BASE_PATH, DATA_PATH, EXP_PATH = setup_paths_and_seed(args.object)

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data = data_loader(DATA_PATH, device=device)
    train(
        args,
        data,
        EXP_PATH,
        device=device,
        coarse_ckpt_path=BASE_PATH / "results" / args.object / "coarse_last.tar",
    )

    model, device = load_model(EXP_PATH / "fine_last.tar")
    render_and_save_views(data, model, device, EXP_PATH, args)


if __name__ == "__main__":
    main()
