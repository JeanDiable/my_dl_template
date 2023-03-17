import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import optimizer
import os
import wandb

from utils import *
from args import get_args
from architectures import get_architecture
from datasets import get_dataloader
config = load_config()

def train(epoch,net,trainloader):
    net.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx,(inputs,targets) in enumerate(trainloader):
        if epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001 + (args.lr - 0.001) * (batch_idx+1)/len(trainloader)
            
        inputs,targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total += targets.size(0)
        correct += compute_correct(outputs,targets)

        log.refresh(batch_idx,len(trainloader),'Train','Loss:{:.3f} | Acc: {:.3f}%'.format(total_loss/(batch_idx+1),100.* correct/total))
    wandb.log({'train_loss':total_loss/(batch_idx+1),'train_acc':100.*correct/total},step=epoch)

@torch.no_grad()
def  valid_test(net,dataloader):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0
    toPIL = transforms.ToPILImage()
    for batch_idx,(inputs,targets) in enumerate(dataloader):     
        inputs,targets = inputs.to(device), targets.to(device)
        save_tensor_to_png(inputs[0],'test.png')
        outputs = net(inputs)
        loss = criterion(outputs,targets)
        total_loss += loss.item()
        total += targets.size(0)
        correct += compute_correct(outputs,targets)
        log.refresh(batch_idx,len(dataloader),'Loss:{:.3f} | Acc: {:.3f}%'.format(total_loss/(batch_idx+1),100.* correct/total))
    return 100.* correct / total, total_loss/(batch_idx + 1)

def test(epoch,net,testloader):
    acc,loss = valid_test(net,testloader)
    wandb.log({'test_clean_loss':loss,'test_clean_acc':acc},step=epoch)
    global best_acc
    if acc< best_acc:
        print('saving.....')
        state = {
            'net' : net.state_dict(),
            'acc':acc,
            'epoch':epoch,
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
        }
        if not os.path.exists(os.path.dirname(args.ckpt)):
            os.mkdir(os.path.dirname(args.ckpt))
        torch.save(state,args.ckpt)
        best_acc = acc

if __name__ == "__main__":
    args = get_args()
    wandb.init(project="Diable-" + args.dataset,entity="Diable",save_code=True,mode='online' if args.wandb else'disabled',name=f'{args.model}-{args.dataset}')
    wandb.config.update(args)

    params = f'{args.model}_{args.dataset}'
    log_path = f'{args.logfile}/log_{params}_{str(datetime.datetime.now())[:-7]}.log'
    log = LogProcessBar(log_path,args,args.print_log)

    if torch.cuda.is_available():
        device = 'cuda'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        cudnn.benchmark = True
    else:
        device = 'cpu'

    best_acc = 0
    start_epoch = 0

    print('====>building model...')
    net = get_architecture(arch=args.model,dataset = args.dataset,pretrained=args.pretrained,normalize=True)
    net = net.to(device)

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer,milestoned=[75,90],gamma=0.1)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.5)
    else:
        optimizer = None
        raise NotImplementedError
    
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        print('=====>Resuming from checkpoint....')
        assert os.path.exists(args.ckpt),'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.ckpt)
        print('ckpt folder:',args.ckpt)
        net.load_state_dict(checkpoint['net'])
        if args.ct:
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler'])

trainloader,testloader = get_dataloader(dataset=args.dataset,batch_size=args.batch,shuffle=True,normalize=False,num_workers=args.workers)

valid_test(net,testloader)

wandb.watch(net,log='all')
for epoch in range(start_epoch, args.epoch):
    if epoch==1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    print('\nEpoch: %d' % epoch)
    print(args.model, 'lr now:', optimizer.state_dict()['param_groups'][0]['lr'])
    train(epoch, net, trainloader)
    # valid(epoch, net, validloader, args)
    test(epoch, net, testloader)
    scheduler.step()
    
