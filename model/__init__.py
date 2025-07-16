from .vnet import VNet
from .unet import UNet2D, UNet3D
from monai.networks.nets import SwinUNETR

def get_model(args):
    if args.model == 'unet':
        model = UNet3D()
    elif args.model == 'unet2d':
        model = UNet2D()
    elif args.model == 'vnet':
        model = VNet(outChans=args.out_channel)
    elif args.model == 'swin':
        model = SwinUNETR( img_size = args.patch_size, in_channels = args.in_channel, out_channels = args.out_channel)
    else:
        print('Model %s not implemented, return None' % args.model)
        model = None
    return model
