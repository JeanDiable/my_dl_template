import argparse

def get_args():
    parser = argparse.ArgumentParser(description='template training')
    parser.add_argument('--model',type = str, default = 'resnet50')
    parser.add_argument('--dataset',type = str, default = 'imagenet')

    parser.add_argument('--batch',type = int, default = 128)
    parser.add_argument('--epoch',type = int, default = 100)
    parser.add_argument('--optimizer',type=str, default='sgd')
    parser.add_argument('--lr',type=float, default = 1e-1)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--resume','-r',action='store_true')
    parser.add_argument('--pretrained','-p',action='store_true')
    parser.add_argument('--continue','-c',action='store_true')
    parser.add_argument('--debug','-nd',action='store_false')

    parser.add_argument('--logfile',type=str,default='./log')
    #TODO set check point path
    parser.add_argument('--ckpt',type=str,default='./ckpt.pth')
    parser.add_argument('--gpuid',typr=int,default='0')
    parser.add_argument('--wandb','-w',action='store_true')
    parser.add_argument('--print_log',type=bool,default=True)
    args = parser.parse_args()
    return args