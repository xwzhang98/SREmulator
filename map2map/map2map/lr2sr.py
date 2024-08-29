import chunk
import numpy as np
import torch
from .models.unet import UNetModel as Unet
from .models.discriminator import Discriminator
from .models.styled_srsgan import G
from .utils.utils_lr2sr import *
import os
from bigfile import BigFile


def lr2sr(args):
    # try crop size 14
    upsample_fac = 8
    device = "cuda"

    style = np.load(args.style_path)
    noise_style = np.load("style.npy")

    lr_disp = np.load(args.lr_disp_input)
    lr_vel = np.load(args.lr_vel_input)

    noise_disp = np.load(args.noise_disp)
    noise_vel = np.load(args.noise_vel)

    size = lr_disp.shape[1:]
    size = np.asarray(size)
    ndim = len(size)

    chunk_size = np.asarray([16, 16, 16])
    crop = np.broadcast_to(chunk_size, size.shape)
    reps = np.rint(size / crop).astype(np.int64)
    tot_reps = int(np.prod(reps))
    pad = args.padding
    pad = np.broadcast_to(pad, (ndim, 2))
    noise_pad = args.noise_padding
    noise_pad = np.broadcast_to(noise_pad, (ndim, 2))

    tgt_size = crop[0] * upsample_fac
    tgt_chunk = np.broadcast_to(tgt_size, size.shape)

    noise_size = np.asarray(noise_disp.shape[1:])
    noise_chunk_size = crop[0] * upsample_fac
    noise_crop = np.broadcast_to(noise_chunk_size, noise_size.shape)
    noise_reps = np.rint(noise_size / noise_crop).astype(np.int64)
    noise_pad = np.broadcast_to(noise_pad, (ndim, 2))

    Ng_sr = size[0] * upsample_fac
    disp_field = np.zeros([3, Ng_sr, Ng_sr, Ng_sr])
    vel_field = np.zeros([3, Ng_sr, Ng_sr, Ng_sr])

    ######################### Load model ############################
    model = Unet(12, 6)
    state = torch.load(args.model_path, map_location=device)
    if args.ema:
        model.load_state_dict(state["ema"])
    else:
        model.load_state_dict(state["model"])
    print("load model state at epoch {}".format(state["epoch"]))
    epoch = state["epoch"]
    del state

    model.eval()
    model.to(device)

    generator = G(6, 6, 1, 8)
    state = torch.load(
        "state.pt",
        map_location=device,
    )
    G_state = state["model"]
    generator.load_state_dict(G_state)
    del state
    del G_state
    generator.eval()
    generator.to(device)

    for idx in range(0, tot_reps):
        lr_disp_chunk = cropfield(lr_disp, idx, reps, crop, pad)
        lr_vel_chunk = cropfield(lr_vel, idx, reps, crop, pad)
        noise_disp_chunk = cropfield(noise_disp, idx, noise_reps, noise_crop, noise_pad)
        noise_vel_chunk = cropfield(noise_vel, idx, noise_reps, noise_crop, noise_pad)
        style = style.reshape(1, 1)
        print("chunk {} / {}".format(idx + 1, tot_reps))
        sr_disp_chunk, sr_vel_chunk = superres_with_addon(
            lr_disp_chunk,
            lr_vel_chunk,
            style,
            noise_disp_chunk,
            noise_vel_chunk,
            noise_style,
            tgt_size=tgt_size,
            model=[generator, model],
            device=device,
        )

        ns = np.unravel_index(idx, reps) * tgt_chunk

        disp_field[
            :,
            ns[0] : ns[0] + tgt_size,
            ns[1] : ns[1] + tgt_size,
            ns[2] : ns[2] + tgt_size,
        ] = (
            sr_disp_chunk.numpy()
            if ns[0] + tgt_size <= Ng_sr
            and ns[1] + tgt_size <= Ng_sr
            and ns[2] + tgt_size <= Ng_sr
            else sr_disp_chunk.numpy()[
                :, : Ng_sr - ns[0], : Ng_sr - ns[1], : Ng_sr - ns[2]
            ]
        )
        vel_field[
            :,
            ns[0] : ns[0] + tgt_size,
            ns[1] : ns[1] + tgt_size,
            ns[2] : ns[2] + tgt_size,
        ] = (
            sr_vel_chunk.numpy()
            if ns[0] + tgt_size <= Ng_sr
            and ns[1] + tgt_size <= Ng_sr
            and ns[2] + tgt_size <= Ng_sr
            else sr_vel_chunk.numpy()[
                :, : Ng_sr - ns[0], : Ng_sr - ns[1], : Ng_sr - ns[2]
            ]
        )

        print("{}/{} done".format(idx + 1, tot_reps), flush=True)

    disp_field = np.float64(disp_field)
    vel_field = np.float64(vel_field)

    sr_pos = dis2pos(disp_field, boxsize=args.Lbox_kpc, Ng=Ng_sr)

    if args.require_id:
        IDs = gen_id(Ng_sr)

    sr_pos = sr_pos.reshape(3, Ng_sr * Ng_sr * Ng_sr).transpose()
    vel_field = vel_field.reshape(3, Ng_sr * Ng_sr * Ng_sr).transpose()

    path = args.sr_path
    os.makedirs(path, exist_ok=True)

    if path[-1] != "/":
        path += "/"

    dest = BigFile(path, create=1)

    blockname = "Position"
    dest.create_from_array(blockname, sr_pos)

    blockname = "Velocity"
    dest.create_from_array(blockname, vel_field)

    if args.require_id:
        blockname = "ID"
        dest.create_from_array(blockname, IDs)

    print("Generated SR column in ", path)
