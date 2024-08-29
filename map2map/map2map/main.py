from .args import get_args
from . import train
from . import test


def main():
    args = get_args()

    if args.mode == "train":
        train.node_worker(args)
        
    elif args.mode == "test":
        test.test(args)
        
    elif args.mode == "lr2sr":
        from . import lr2sr
        lr2sr.lr2sr(args)
        
    elif args.mode == "emu_sr":
        from . import emu_sr
        emu_sr.emu_sr(args)
        
    elif args.mode == "lr2sr_gradcam":
        from . import lr2sr_gradcam

        lr2sr_gradcam.lr2sr(args)


if __name__ == "__main__":
    main()
