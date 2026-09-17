import os
import socket
import time
import sys
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .data import FieldDataset, DistFieldSampler
from . import models
from .models import (
    narrow_like,
    resample,
    lag2eul,
    G,
)
from .utils import import_attr, load_model_state_dict, plt_slices, plt_power, score


ckpt_link = "checkpoint.pt"


def node_worker(args):
    if "SLURM_STEP_NUM_NODES" in os.environ:
        args.nodes = int(os.environ["SLURM_STEP_NUM_NODES"])
    elif "SLURM_JOB_NUM_NODES" in os.environ:
        args.nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    else:
        raise KeyError("missing node counts in slurm env")
    args.gpus_per_node = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus_per_node

    node = int(os.environ["SLURM_NODEID"])

    if args.gpus_per_node < 1:
        raise RuntimeError("GPU not found on node {}".format(node))
    print(args.gpus_per_node, args.world_size, node, flush=True)
    spawn(gpu_worker, args=(node, args), nprocs=args.gpus_per_node)


def gpu_worker(local_rank, node, args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    device = torch.device("cuda", 0)
    torch.cuda.empty_cache()

    rank = args.gpus_per_node * node + local_rank

    # Need randomness across processes, for sampler, augmentation, noise etc.
    # Note DDP broadcasts initial model states from rank 0
    torch.manual_seed(args.seed + rank)
    # good practice to disable cudnn.benchmark if enabling cudnn.deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    dist_init(rank, args)

    dtype = torch.float32
    torch.set_default_dtype(dtype)

    train_dataset = FieldDataset(
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        style_pattern=args.train_style_pattern,
        noise_patterns=args.train_noise_patterns,
        noise_style_pattern=args.train_noise_style_pattern,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        noise_norms=args.noise_norms,
        callback_at=args.callback_at,
        augment=args.augment,
        aug_shift=args.aug_shift,
        aug_add=args.aug_add,
        aug_mul=args.aug_mul,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        noise_pad=args.noise_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    train_sampler = DistFieldSampler(
        train_dataset,
        shuffle=True,
        div_data=args.div_data,
        div_shuffle_dist=args.div_shuffle_dist,
    )
    # random_sampler =
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.loader_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    if args.val:
        val_dataset = FieldDataset(
            in_patterns=args.val_in_patterns,
            tgt_patterns=args.val_tgt_patterns,
            style_pattern=args.val_style_pattern,
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
        val_sampler = DistFieldSampler(
            val_dataset,
            shuffle=False,
            div_data=args.div_data,
            div_shuffle_dist=args.div_shuffle_dist,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.loader_workers,
            pin_memory=True,
        )

    args.in_chan = train_dataset.in_chan
    args.out_chan = train_dataset.tgt_chan
    args.style_size = train_dataset.style_size
    args.noise_style_size = train_dataset.noise_style_size

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(
        2 * sum(args.in_chan),
        sum(args.out_chan),
        style_size=args.style_size,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    model.float()
    model.to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], process_group=dist.new_group()
    )

    criterion = import_attr(args.criterion, nn, models, callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)

    optimizer = optimizer(
        model.parameters(),
        lr=args.lr,
        **args.optimizer_args,
    )
    if args.warmup:
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, **args.scheduler_args
        )

        number_warmup_epochs = args.warmup_epochs

        def warmup(current_step: int):
            return 1 / (2 ** (float(number_warmup_epochs - current_step)))

        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, train_scheduler], [number_warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, **args.scheduler_args
        )
    adv_model = adv_criterion = adv_optimizer = adv_scheduler = None
    if args.adv:
        adv_model = import_attr(args.adv_model, models, callback_at=args.callback_at)
        adv_model = adv_model(
            sum(args.in_chan + args.out_chan),
            1,
            style_size=args.style_size,
            scale_factor=args.scale_factor,
            **args.misc_kwargs,
        )
        adv_model.to(device)
        adv_model = DistributedDataParallel(
            adv_model,
            device_ids=[device],
            process_group=dist.new_group(),
        )

        adv_criterion = import_attr(
            args.adv_criterion, nn, models, callback_at=args.callback_at
        )
        adv_criterion = adv_criterion()
        adv_criterion.to(device)

        adv_optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
        adv_optimizer = adv_optimizer(
            adv_model.parameters(),
            lr=args.adv_lr,
            **args.adv_optimizer_args,
        )
        adv_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            adv_optimizer, T_max=args.epochs, **args.scheduler_args
        )

    if (
        args.load_state == ckpt_link
        and not os.path.isfile(ckpt_link)
        or not args.load_state
    ):
        if args.init_weight_std is not None:
            model.apply(init_weights)

        start_epoch = 0
        pretrained_layers = None

        if rank == 0:
            min_loss = None
    else:
        state = torch.load(args.load_state, map_location=device)

        if "epoch" in state:
            start_epoch = state["epoch"]
        else:
            start_epoch = 0

        load_model_state_dict(
            model.module, state["model"], strict=args.load_state_strict
        )

        if "optimizer" in state:
            if args.lr != state["optimizer"]["param_groups"][0]["lr"]:
                state["optimizer"]["param_groups"][0]["lr"] = args.lr
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler_state = state["scheduler"]
            if args.lr != scheduler_state["base_lrs"][0]:
                scheduler_state["base_lrs"] = [args.lr]
                scheduler_state["_last_lr"] = [args.lr]
            scheduler.load_state_dict(state["scheduler"])

        if args.adv:
            if "adv_model" in state:
                load_model_state_dict(
                    adv_model.module, state["adv_model"], strict=args.load_state_strict
                )

            if "adv_optimizer" in state:
                adv_optimizer.load_state_dict(state["adv_optimizer"])
            if "adv_scheduler" in state:
                adv_scheduler.load_state_dict(state["adv_scheduler"])

        torch.set_rng_state(state["rng"].cpu())  # move rng state back

        if rank == 0:
            min_loss = state["min_loss"]
            if args.adv and "adv_model" not in state:
                min_loss = None  # restarting with adversary wipes the record

            print(
                "state at epoch {} loaded from {}".format(
                    state["epoch"], args.load_state
                ),
                flush=True,
            )

        if args.ema:
            if "ema" in state:
                ema_state = state["ema"]
            else:
                ema_state = state["model"]

        del state

    torch.backends.cudnn.benchmark = True

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    logger = None
    if rank == 0:
        logger = SummaryWriter()

    if rank == 0:
        print("pytorch {}".format(torch.__version__))
        pprint(vars(args))
        sys.stdout.flush()

    generator = G(6, 6, 1, 8)
    state = torch.load(
        "state.pt",
        map_location=device,
    )
    generator.load_state_dict(state["model"])
    del state
    generator.to(device)
    generator.eval()

    discriminator = None

    # tf32 stuff on A100
    if not torch.backends.cuda.matmul.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.backends.cudnn.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train(
            epoch,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            adv_model,
            adv_criterion,
            adv_optimizer,
            adv_scheduler,
            logger,
            device,
            args,
            generator=generator,
        )

        # prof.step()

        epoch_loss = train_loss

        if args.reduce_lr_on_plateau:
            scheduler.step(epoch_loss[2] * epoch_loss[6])
        else:
            scheduler.step()
        if rank == 0:
            logger.flush()

            if (
                min_loss is None or epoch_loss[7] < min_loss
            ) and epoch >= args.adv_start:
                min_loss = epoch_loss[7]
            pretrained_layers = None
            state = {
                "epoch": epoch + 1,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "min_loss": min_loss,
                "pretrained_layers": pretrained_layers,
                "mtl": mtl.module.state_dict() if args.multitaskloss else None,
            }
            if args.adv:
                state.update(
                    {
                        "adv_model": adv_model.module.state_dict(),
                        "adv_optimizer": adv_optimizer.state_dict(),
                        "adv_scheduler": adv_scheduler.state_dict(),
                    }
                )

            if args.ema:
                ema_state = update_ema(model.module.state_dict(), ema_state)
                state.update({"ema": ema_state})

            state_file = "state_{}.pt".format(epoch + 1)
            torch.save(state, state_file)
            del state

            tmp_link = "{}.pt".format(time.time())
            os.symlink(state_file, tmp_link)  # workaround to overwrite
            os.rename(tmp_link, ckpt_link)
    # prof.stop()
    dist.destroy_process_group()


def train(
    epoch,
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
    adv_model,
    adv_criterion,
    adv_optimizer,
    adv_scheduler,
    logger,
    device,
    args,
    generator=None,
):
    torch.cuda.reset_peak_memory_stats(device=device)
    eul_scale_factor = 2
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        epoch_start = torch.cuda.Event(enable_timing=True)
        training_start = torch.cuda.Event(enable_timing=True)
        epoch_logging_start = torch.cuda.Event(enable_timing=True)
        epoch_making_plots_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)
        epoch_start.record()

    model.train()
    if args.adv:
        adv_model.train()
    set_requires_grad(model, requires_grad=True)
    if generator is not None:
        generator.eval()
        set_requires_grad(generator, requires_grad=False)

    epoch_loss = torch.zeros(100, dtype=torch.float32, device=device)

    if rank == 0:
        training_start.record()

    for i, data in enumerate(loader):
        batch = epoch * len(loader) + i + 1

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

        if i <= 1 and rank == 0:
            print("##### batch :", batch)
            print("total batch :", len(loader))
            print("input shape :", input.shape)
            print("output shape :", output.shape)
            print("target shape :", target.shape)
        if output.shape != target.shape:
            output = narrow_like(output, target)
        if i <= 1 and rank == 0:
            print("narrowed shape :", output.shape)
            print("lag loss weight :", args.lag_loss_weight)
            print("eul loss weight :", args.eul_loss_weight)
            print("adv loss weight :", args.adv_loss_weight)
            print("#####", flush=True)
        if (i + 1) % 200 == 0 and rank == 0:
            print("##### current batch :", i + 1)
            print("total batch :", len(loader))
            print("input shape :", input.shape)
            print("output shape :", output.shape)
            print("target shape :", target.shape)
            print("narrowed shape :", output.shape)
            print("#####", flush=True)
        if args.adv:
            input = resample(input, scale_factor=args.scale_factor)
            input = narrow_like(input, target)

        else:
            del input
        optimizer.zero_grad(set_to_none=True)

        disp_lag_out, disp_lag_tgt = output[:, :3], target[:, :3]
        vel_lag_out, vel_lag_tgt = output[:, 3:], target[:, 3:]

        if args.adv and epoch >= args.adv_start:
            # discriminator
            set_requires_grad(adv_model, True)
            adv_optimizer.zero_grad(set_to_none=True)
            score_out = adv_model(output.detach(), style=style, cond=input)
            score_tgt = adv_model(target, style=style, cond=input)

            adv_loss_real, adv_loss_fake = adv_criterion(score_out, score_tgt)
            epoch_loss[8] += adv_loss_real.detach()
            epoch_loss[9] += adv_loss_fake.detach()
            adv_loss = adv_loss_fake + adv_loss_real
            adv_loss.backward()
            adv_optimizer.step()
        if args.eul:
            disp_eul_out, disp_eul_tgt = lag2eul(
                dis=[disp_lag_out, disp_lag_tgt],
                a=np.float64(style),
                eul_scale_factor=2,
                rm_dis_mean=True,
            )
            disp_eul_loss = criterion(disp_eul_out, disp_eul_tgt)

        # ---------- displacement loss ----------
        disp_lag_loss = criterion(disp_lag_out, disp_lag_tgt)

        epoch_loss[0] += disp_lag_loss.detach()
        if args.eul:
            epoch_loss[1] += disp_eul_loss.detach()

        if args.eul:
            dis_loss = 1.0 * disp_lag_loss + 1.0 * disp_eul_loss
        else:
            dis_loss = disp_lag_loss

        # ---------- velocity loss ----------
        if args.vel_eul:
            vel_eul_out, vel_eul_tgt = (
                lag2eul(
                    disp_lag_tgt,
                    val=vel_lag_out,
                    a=np.float64(style),
                    eul_scale_factor=eul_scale_factor,
                )[0],
                lag2eul(
                    disp_lag_tgt,
                    val=vel_lag_tgt,
                    a=np.float64(style),
                    eul_scale_factor=eul_scale_factor,
                )[0],
            )

        vel_lag_loss = criterion(vel_lag_out, vel_lag_tgt)

        if args.vel_eul:
            vel_eul_loss = criterion(vel_eul_out, vel_eul_tgt)
            vel_loss = args.lag_loss_weight * (vel_lag_loss + vel_eul_loss)
        else:
            vel_loss = args.lag_loss_weight * vel_lag_loss

        epoch_loss[2] += vel_lag_loss.detach()
        if args.vel_eul:
            epoch_loss[3] += vel_eul_loss.detach()

        total_loss = dis_loss + vel_loss

        if args.adv and epoch >= args.adv_start:
            set_requires_grad(adv_model, False)
            score_out = adv_model(output, style=style, cond=input)
            loss_adv = adv_criterion(score_out)
            epoch_loss[12] += loss_adv.detach()
            total_loss += 3e-2 * loss_adv
        total_loss.backward()
        optimizer.step()
        if i + 1 != len(loader):
            try:
                del (
                    disp_lag_out,
                    disp_lag_tgt,
                    disp_lag_loss,
                    disp_eul_out,
                    disp_eul_tgt,
                    disp_eul_loss,
                    dis_loss,
                )
                del (
                    vel_lag_out,
                    vel_lag_tgt,
                    vel_eul_out,
                    vel_eul_tgt,
                    vel_eul_loss,
                    vel_lag_loss,
                    vel_eul2_out,
                    vel_eul2_tgt,
                    vel_eul2_loss,
                    vel_loss,
                )
                del style, noise, sr_out, output, target, total_loss
            except:
                pass

    if rank == 0:
        epoch_logging_start.record()
    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    lr = scheduler.get_last_lr()[0]
    if rank == 0:
        # ---------- learning rate log ----------
        logger.add_scalar("hyperparam/lr", lr, global_step=epoch + 1)

        # ---------- loss log ----------
        logger.add_scalar(
            "epoch/train/loss/disp/lag", epoch_loss[0], global_step=epoch + 1
        )
        logger.add_scalar(
            "epoch/train/loss/disp/eul", epoch_loss[1], global_step=epoch + 1
        )
        logger.add_scalar(
            "epoch/train/loss/vel/lag", epoch_loss[2], global_step=epoch + 1
        )
        logger.add_scalar(
            "epoch/train/loss/vel/eul", epoch_loss[3], global_step=epoch + 1
        )
        logger.add_scalar(
            "epoch/train/loss/vel/eul2", epoch_loss[4], global_step=epoch + 1
        )

        if args.adv and epoch >= args.adv_start:
            logger.add_scalars(
                "epoch/train/loss/adv/",
                {
                    "real": epoch_loss[8],
                    "fake": epoch_loss[9],
                },
                global_step=epoch + 1,
            )
            logger.add_scalars(
                "epoch/train/weight",
                {
                    "lag": args.lag_loss_weight,
                    "eul": args.eul_loss_weight,
                    "adv": args.adv_loss_weight,
                },
                global_step=epoch + 1,
            )
            logger.add_scalar(
                "epoch/train/loss/adv", epoch_loss[12], global_step=epoch + 1
            )

        logger.add_scalars(
            "stat/mem/",
            {
                "max_alloc": epoch_loss[-1],
                "max_reserved": epoch_loss[-2],
            },
            global_step=epoch + 1,
        )

        epoch_making_plots_start.record()
        if (epoch + 1) % 1 == 0:
            disp_sr_out = sr_out[:, :3].detach()
            vel_sr_out = sr_out[:, 3:].detach()

            with torch.no_grad():
                sr_eul = lag2eul(
                    sr_out[:, :3],
                    a=np.float64(style),
                    eul_scale_factor=eul_scale_factor,
                )[0]
                disp_eul_out = lag2eul(
                    disp_lag_out,
                    a=np.float64(style),
                    eul_scale_factor=eul_scale_factor,
                )[0]
                disp_eul_tgt = lag2eul(
                    disp_lag_tgt,
                    a=np.float64(style),
                    eul_scale_factor=eul_scale_factor,
                )[0]
            try:
                fig = plt_slices(
                    disp_sr_out[-1],
                    disp_lag_out[-1],
                    disp_lag_tgt[-1],
                    disp_lag_out[-1] - disp_lag_tgt[-1],
                    sr_eul[-1],
                    disp_eul_out[-1],
                    disp_eul_tgt[-1],
                    disp_eul_out[-1] - disp_eul_tgt[-1],
                    vel_sr_out[-1],
                    vel_lag_out[-1],
                    vel_lag_tgt[-1],
                    vel_lag_out[-1] - vel_lag_tgt[-1],
                    title=[
                        "ai3 disp",
                        "out disp",
                        "tgt disp",
                        "disp diff",
                        "ai3 eul",
                        "out eul",
                        "tgt eul",
                        "eul diff",
                        "ai3 vel",
                        "out vel",
                        "tgt vel",
                        "vel diff",
                    ],
                    **args.misc_kwargs,
                )
                logger.add_figure("fig/train", fig, global_step=epoch + 1)
                fig.clf()
            except Exception as error:
                print(error)
                pass

            try:
                del (
                    disp_sr_out,
                    disp_lag_out,
                    disp_lag_tgt,
                    disp_eul_out,
                    disp_eul_tgt,
                    sr_eul,
                    vel_sr_out,
                    vel_lag_out,
                    vel_lag_tgt,
                )
            except:
                pass
        epoch_end.record()
        torch.cuda.synchronize()
        logger.add_scalars(
            "stat/time/train",
            {
                "prepare": epoch_start.elapsed_time(training_start) / 1000,
                "training": training_start.elapsed_time(epoch_logging_start) / 1000,
                "loss logging": epoch_logging_start.elapsed_time(
                    epoch_making_plots_start
                )
                / 1000,
                "making plots": epoch_making_plots_start.elapsed_time(epoch_end) / 1000,
            },
            global_step=epoch + 1,
        )
        max_memory_alloc = torch.cuda.max_memory_allocated(device=device)
        max_momory_reserved = torch.cuda.max_memory_reserved(device=device)
        max_memory_alloc = round(max_memory_alloc / (1024**3), 2)
        max_momory_reserved = round(max_momory_reserved / (1024**3), 2)
        logger.add_scalars(
            "stat/mem/",
            {
                "max_alloc": max_memory_alloc,
                "max_reserved": max_momory_reserved,
            },
            global_step=epoch + 1,
        )
    return epoch_loss


def dist_init(rank, args):
    dist_file = "dist_addr"

    if rank == 0:
        addr = socket.gethostname()

        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, 0))
            _, port = s.getsockname()

        args.dist_addr = "tcp://{}:{}".format(addr, port)

        with open(dist_file, mode="w") as f:
            f.write(args.dist_addr)
    else:
        while not os.path.exists(dist_file):
            time.sleep(1)

        with open(dist_file, mode="r") as f:
            args.dist_addr = f.read()

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_addr,
        world_size=args.world_size,
        rank=rank,
    )
    dist.barrier()

    if rank == 0:
        os.remove(dist_file)


def init_weights(m):
    if isinstance(
        m,
        (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        m.weight.data.normal_(0.0, args.init_weight_std)
    elif isinstance(
        m,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    ):
        if m.affine:
            m.weight.data.normal_(1.0, args.init_weight_std)
            m.bias.data.fill_(0)


def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_grads(model):
    """gradients of the weights of the first and the last layer"""
    grads = list(p.grad for n, p in model.named_parameters() if ".weight" in n)
    grads = [grads[0], grads[-1]]
    grads = [g.detach().norm() for g in grads]
    return grads


def get_weight(model):
    """weights of the first and the last layer"""
    for n, p in model.named_parameters():
        if "log" in n:
            log_vars = p.detach().clone()
    stds = torch.exp(log_vars) ** (1 / 2)
    weights = 1 / (stds**2)
    return weights


def update_ema(model, ema_model, alpha=0.9999):
    for ema_param, param in zip(ema_model.values(), model.values()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    return ema_model
