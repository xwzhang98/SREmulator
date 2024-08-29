import os
import sys
import warnings
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FieldDataset
from .data import norms
from . import models
from .models import narrow_cast
from .utils import import_attr, load_model_state_dict
from .models import narrow_like
from .data import norms


def test(args):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            warnings.warn("Not parallelized but given more than 1 GPUs")

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda", 0)

        torch.backends.cudnn.benchmark = True
    else:  # CPU multithreading
        device = torch.device("cpu")

        if args.num_threads is None:
            args.num_threads = int(os.environ["SLURM_CPUS_ON_NODE"])

        torch.set_num_threads(args.num_threads)

    print("pytorch {}".format(torch.__version__))
    pprint(vars(args))
    sys.stdout.flush()

    test_dataset = FieldDataset(
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        style_pattern=args.test_style_pattern,
        noise_patterns=args.test_noise_patterns,
        noise_style_pattern=args.test_noise_style_pattern,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=False,
        aug_shift=None,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    in_chan = test_dataset.in_chan
    out_chan = test_dataset.tgt_chan
    style_size = test_dataset.style_size

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(
        2 * sum(in_chan),
        sum(out_chan),
        style_size=style_size,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    model.to(device)

    generator = G(6, 6, 1, 8)
    state = torch.load(
        "/hildafs/home/xzhangn/xzhangn/emulator_sr/5-training/test_pretrained/GAN_state/state_710.pt",
        map_location=device,
    )
    G_state = state["model"]
    generator.load_state_dict(G_state)
    del state
    del G_state
    generator.to(device)
    generator.eval()

    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state["model"], strict=args.load_state_strict)
    print(
        "model state at epoch {} loaded from {}".format(state["epoch"], args.load_state)
    )
    del state

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input, target, style = data["input"], data["target"], data["style"]
            noise = data["noise"]

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            style = style.to(device, non_blocking=True)
            noise = noise.to(device, non_blocking=True)

            with torch.no_grad():
                sr_out = generator(input, style)
            sr_out = narrow_like(sr_out, noise)
            output = model(sr_out, style, noise)

            if i < 5:
                print("##### sample :", i)
                print("input shape :", input.shape)
                print("output shape :", output.shape)
                print("target shape :", target.shape)
                print("style shape :", style.shape)

            norms.cosmology.dis(output[:, :3], a=style.item(), undo=True)
            norms.cosmology.vel(target[:, 3:], a=style.item(), undo=True)

            # test_dataset.assemble('_in', in_chan, input,
            #                      data['input_relpath'])
            test_dataset.assemble("_out", out_chan, output, data["target_relpath"])
            # test_dataset.assemble('_tgt', out_chan, target,
            #                      data['target_relpath'])
