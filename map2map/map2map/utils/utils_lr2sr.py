import numpy as np
import torch
import argparse
import os


def dis2pos(dis_field, boxsize, Ng):
    """Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
    and dis_field.shape = (3,Ng,Ng,Ng)
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos = dis_field.copy()

    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    pos[pos < 0] += boxsize
    pos[pos > boxsize] -= boxsize

    return pos


def narrow_like(sr_box, tgt_Ng):
    """sr_box in shape (Nc,Ng,Ng,Ng),trim to (Nc,tgt_Ng,tgt_Ng,tgt_Ng), better to be even"""
    width = np.shape(sr_box)[1] - tgt_Ng
    half_width = width // 2
    begin, stop = half_width, tgt_Ng + half_width
    return sr_box[:, begin:stop, begin:stop, begin:stop]


def narrow_as(a, b):
    """Narrow a to be like b.

    Try to be symmetric but cut more on the right for odd difference
    """
    for d in range(2, a.dim()):
        width = a.shape[d] - b.shape[d]
        half_width = width // 2
        a = a.narrow(d, half_width, a.shape[d] - width)
    return a


def cropfield(field, idx, reps, crop, pad):
    """input field in shape of (Nc,Ng,Ng,Ng),
    crop idx^th subbox in reps grid with padding"""
    start = np.unravel_index(idx, reps) * crop  # find coordinate of idx in reps grid
    x = field.copy()
    for d, (i, N, (p0, p1)) in enumerate(zip(start, crop, pad)):
        x = x.take(range(i - p0, i + N + p1), axis=1 + d, mode="wrap")

    return x


def crop_info(noise, n_split, pad):
    """return the crop info of the noise field"""
    size = noise.shape[1:]
    size = np.asarray(size)
    chunk_size = size // n_split
    crop = np.broadcast_to(chunk_size, size.shape)
    reps = size // crop
    tot_reps = int(np.prod(reps))
    ndim = len(size)
    pad = np.broadcast_to(pad, (ndim, 2))
    return tot_reps, reps, crop, pad


def gen_id(Ng):
    return np.arange(Ng**3) + 1


def superres_with_addon(
    lr_disp, lr_vel, style, noise_disp, noise_vel, noise_style, tgt_size, model, device
):
    if isinstance(model, list):
        G = model[0]
        model = model[1]

    lr_disp = dis(lr_disp, a=style)
    lr_vel = vel(lr_vel, a=style)

    noise_disp = dis(noise_disp, a=noise_style)
    noise_vel = vel(noise_vel, a=noise_style)

    lr_field = np.concatenate([lr_disp, lr_vel], axis=0)
    lr_field = torch.from_numpy(lr_field).float()
    lr_field.unsqueeze_(0)
    lr_field = lr_field.to(device)

    style = torch.from_numpy(style).float()
    style.unsqueeze_(0)
    style = style.view(1, -1)
    style = style.to(device)

    noise = np.concatenate([noise_disp, noise_vel], axis=0)
    noise = torch.from_numpy(noise).float()
    noise.unsqueeze_(0)
    noise = noise.to(device)

    print("lr_field.shape", lr_field.shape)
    print("style.shape", style.shape)
    print("noise.shape", noise.shape)
    print("noise_style.shape", noise_style.shape)
    with torch.no_grad():
        sr_out = G(lr_field, style)
        print(sr_out.shape)
        sr_out = narrow_as(sr_out, noise)
        sr_field = model(sr_out, style, noise)

    sr_field.squeeze_(0)
    print("sr_field.shape", sr_field.shape)
    if style > 1:
        style = style / 1000
    print(style)

    sr_disp = dis(sr_field[:3, :, :, :].cpu(), a=style[0].cpu().numpy(), undo=True)
    sr_vel = vel(sr_field[3:, :, :, :].cpu(), a=style[0].cpu().numpy(), undo=True)
    sr_disp = narrow_like(sr_disp, tgt_size)
    sr_vel = narrow_like(sr_vel, tgt_size)
    print("sr_disp.shape", sr_disp.shape)
    print("sr_vel.shape", sr_vel.shape)

    return sr_disp, sr_vel



def apply_emulator(
    input_disp, input_vel, style, noise_disp, noise_vel, noise_style, tgt_size, model, device
):

    input_disp = dis(input_disp, a=style)
    input_vel = vel(input_vel, a=style)

    noise_disp = dis(noise_disp, a=noise_style)
    noise_vel = vel(noise_vel, a=noise_style)

    input_field = np.concatenate([input_disp, input_vel], axis=0)
    input_field = torch.from_numpy(input_field).float()
    input_field.unsqueeze_(0)
    input_field = input_field.to(device)

    style = torch.from_numpy(style).float()
    style.unsqueeze_(0)
    style = style.view(1, -1)
    style = style.to(device)

    noise = np.concatenate([noise_disp, noise_vel], axis=0)
    noise = torch.from_numpy(noise).float()
    noise.unsqueeze_(0)
    noise = noise.to(device)

    print("input field.shape", input_field.shape)
    print("style shape", style.shape)
    print("noise shape", noise.shape)
    print("noise style shape", noise_style.shape)
    with torch.no_grad():
        output_field = model(input_field, style, noise)

    output_field.squeeze_(0)
    print("output_field.shape", output_field.shape)
    # if style > 1:
    #     style = style / 1000
    # print(style)

    out_disp = dis(output_field[:3, :, :, :].cpu(), a=style[0].cpu().numpy(), undo=True)
    out_vel = vel(output_field[3:, :, :, :].cpu(), a=style[0].cpu().numpy(), undo=True)
    out_disp = narrow_like(out_disp, tgt_size)
    out_vel = narrow_like(out_vel, tgt_size)
    print("out_disp.shape", out_disp.shape)
    print("out_vel.shape", out_vel.shape)

    return out_disp, out_vel


import numpy as np
from scipy.special import hyp2f1


def dis(x, undo=False, a=0.3333, dis_std=6000.0, **kwargs):
    z = 1 / a - 1
    dis_norm = dis_std * D(z)  # [Kpc/h]

    if not undo:
        dis_norm = 1 / dis_norm

    x *= dis_norm

    return x


def vel(x, undo=False, a=0.3333, dis_std=6.0, **kwargs):
    z = 1 / a - 1
    vel_norm = dis_std * D(z) * H(z) * f(z) / (1 + z)  # [km/s]

    if not undo:
        vel_norm = 1 / vel_norm

    x *= vel_norm
    return x


def D(z, Om=0.31):
    """linear growth function for flat LambdaCDM, normalized to 1 at redshift zero"""
    OL = 1 - Om
    a = 1 / (1 + z)
    return (
        a
        * hyp2f1(1, 1 / 3, 11 / 6, -OL * a**3 / Om)
        / hyp2f1(1, 1 / 3, 11 / 6, -OL / Om)
    )


def f(z, Om=0.31):
    """linear growth rate for flat LambdaCDM"""
    OL = 1 - Om
    a = 1 / (1 + z)
    aa3 = OL * a**3 / Om
    return 1 - 6 / 11 * aa3 * hyp2f1(2, 4 / 3, 17 / 6, -aa3) / hyp2f1(
        1, 1 / 3, 11 / 6, -aa3
    )


def H(z, Om=0.31):
    """Hubble in [h km/s/Mpc] for flat LambdaCDM"""
    OL = 1 - Om
    a = 1 / (1 + z)
    return 100 * np.sqrt(Om / a**3 + OL)
