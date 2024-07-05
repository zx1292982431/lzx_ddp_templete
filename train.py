import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3,6,'
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from utils.progressbar import progressbar as pb
import soundfile as sf
import logging as log
from utils.Checkpoint import Checkpoint
import yaml
from pathlib import Path
import numpy as np
import json
from torch.nn.utils import clip_grad_norm_
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter   
import shutil
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import soundfile as sf

from dataloaders.RandomData import RandomDataset
from models.TempleteModel import TempleteModel
from models.Loss import TempleteLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def get_avg(self):
        self.all_reduce()
        return self.avg

def ddp_setup():
    init_process_group(backend="nccl",timeout=timedelta(minutes=300))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
 
 
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        eval_data: DataLoader,
        criterion,
        optimizer: torch.optim.Optimizer,
        lr_scheduler :torch.optim.lr_scheduler.ReduceLROnPlateau,
        #warmup_scheduler,
        save_every: int,
        model_dir: str,
        model_name: str,
        log_dir: str,
        sample_rate: int,
        test_run: bool,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.epochs_run = 0
        self.model_dir = model_dir
        self.lr_scheduler = lr_scheduler
        #self.warmup_scheduler = warmup_scheduler
        self.best_loss = float('inf')
        self.model_name = model_name
        if self.global_rank == 0:
            self.tensorboard = SummaryWriter(log_dir)
        self.sample_rate =sample_rate
        self.test_run = test_run

        if self.model_name is not None:
            self.checkpoint_path = self.model_dir + model_name 
        else:
            self.checkpoint_path = None
        self.step = 0
 
        self.scaler = torch.cuda.amp.GradScaler()
 
        if os.path.exists(self.checkpoint_path):
            print("Loading snapshot")
            self.load_checkpoints(self.checkpoint_path)
 
        self.model = DDP(self.model,
                        device_ids=[self.local_rank],
                        #find_unused_parameters=True
                        )
        # self.criterion = self.criterion.to(self.local_rank)
 
    def train(self,config):     ############################################################ your train framework
        if self.global_rank == 0:
            log.info('#' * 20 + ' START TRAINING ' + '#' * 20)
        cnt = 0.
        step = self.step
        for epoch in range(0,2) if self.test_run else range(self.epochs_run, config["MAX_EPOCH"]):

            if self.global_rank == 0:
                self.tensorboard.add_scalar(f'optim/learning_rate',self.optimizer.state_dict()['param_groups'][0]['lr'],global_step=epoch)

            b_sz = len(next(iter(self.train_data))[0])
 
            log.info(f'[GPU{self.global_rank}] Epoch:{epoch+1} BatchSize:{b_sz} Step:{len(self.train_data)}')
 
            accu_train_loss = 0.0
            self.model.train()
            tbar = pb(0, 10, 20) if self.test_run else pb(0, len(self.train_data), 20)
            tbar.start()
            eval_cnt = 0
            step = 0
 
            for i,batch_info in enumerate(self.train_data):
                if self.test_run and i > 10:
                    break
                step += 1 
                eval_cnt += 1
                x,y = batch_info[0].to(self.local_rank),batch_info[1].to(self.local_rank)

                for param in self.model.parameters():
                    param.grad = None

                with torch.cuda.amp.autocast():
                    outputs = self.model(x) 
                    loss = self.criterion(x,outputs)

                if torch.isnan(loss.data).any():
                    print('backward nan!')
                    del loss, outputs, batch_info
                    continue
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), 5.)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                #self.warmup_scheduler.step()
 
                running_loss = loss.data.item()

                if self.global_rank == 0:
                    self.tensorboard.add_scalar('train/step/loss',running_loss,global_step=step+len(self.train_data)*epoch)

                accu_train_loss += running_loss
                cnt += 1
                del loss, outputs, batch_info
 
                tbar.update_progress(i, 'Train', 'epoch:{}/{}, loss:{:.5f}/{:.5f}'.format(epoch + 1,
                                                                                      config['MAX_EPOCH'], running_loss,
                                                                                      accu_train_loss / cnt)
                                                                                      )
                
            if (epoch+1) % self.save_every == 0 :
                train_losses = AverageMeter('train/epoch/loss')
                train_loss = accu_train_loss / cnt
                train_losses.update(train_loss)

                eval_losses = AverageMeter('eval/loss')
                eval_loss = self.validate(epoch)
                eval_losses.update(eval_loss)

                avg_train_loss = train_losses.get_avg()
                avg_eval_loss = eval_losses.get_avg()

                if self.global_rank == 0:
                    self.tensorboard.add_scalar('train/epoch/avg_loss',avg_train_loss,global_step=epoch)
                    self.tensorboard.add_scalar('eval/avg_loss',avg_eval_loss,global_step=epoch)

                is_best = True if avg_eval_loss < self.best_loss else False
                self.best_loss = avg_eval_loss if is_best else self.best_loss
                if self.global_rank == 0:
                    log.info('Epoch [%d/%d], ( TrainLoss: %.4f  | EvalLoss: %.4f | SISDR: %.4f )' % (
                        epoch + 1, config['MAX_EPOCH'], avg_train_loss, avg_eval_loss))
                    self.save_checkpoints(self.model_dir,epoch,i,avg_train_loss,is_best)

                accu_train_loss = 0.0
                self.model.train()
                cnt = 0.

                self.lr_scheduler.step(avg_eval_loss) 

        if self.global_rank == 0:
            self.tensorboard.close()
 
    def validate(self,epoch): ################################################################### your val framework
        self.model.eval()
        with torch.no_grad():
            cnt = 0.
            accu_eval_loss = 0.0
            accu_sisdr = 0.0
            ebar = pb(0, 10, 20) if self.test_run else pb(0, len(self.eval_data), 20)
            ebar.start()
            for j, batch_eval in enumerate(self.eval_data):
                if self.test_run and j>10:
                    break
                x,y = batch_eval[0].to(self.local_rank),batch_eval[1].to(self.local_rank)

                with torch.cuda.amp.autocast():
                    outputs = self.model(x)

                    loss1 = self.criterion(x,y)

                    # if j < 3 and self.global_rank == 0:
                    #     self.tensorboard.add_audio(f'{epoch}/{j}/mix',mix.cpu().detach().numpy()[0],sample_rate=self.sample_rate)
                    #     self.tensorboard.add_audio(f'{epoch}/{j}/target',target.cpu().detach().numpy()[0],sample_rate=self.sample_rate)
                    #     self.tensorboard.add_audio(f'{epoch}/{j}/pred',outputs.cpu().detach().numpy()[0][0],sample_rate=self.sample_rate)
                    #     self.tensorboard.add_audio(f'{epoch}/{j}/pre_speaker2',outputs.cpu().detach().numpy()[0][1],sample_rate=self.sample_rate)

                if torch.isnan(loss.data).any():
                    print('nan!')
                    del loss, outputs, batch_eval
                    continue
                
                eval_loss = loss.data.item()
                accu_eval_loss += eval_loss
                cnt += 1.
 
                ebar.update_progress(j, 'CV   ', 'loss:{:.5f}'.format(eval_loss))
           
            avg_eval_loss = accu_eval_loss / cnt
            avg_sisdr = accu_sisdr / cnt
            print()
        print()
 
        self.model.train()
        return avg_eval_loss
    
    def load_checkpoints(self,model_path):
        checkpoint = Checkpoint()
        checkpoint.load(model_path)
        self.epochs_run = checkpoint.start_epoch
        self.best_loss = checkpoint.best_loss
        self.model.load_state_dict(checkpoint.state_dict,strict=True)
        self.optimizer.load_state_dict(checkpoint.optimizer)
        self.scaler.load_state_dict(checkpoint.scaler)
        for param_group in self.optimizer.param_groups:
           param_group['lr']= 0.001
        log.info('#' * 18 + 'Finish Resume Model ' + '#' * 18)
        print('#' * 18 + 'Finish Resume Model ' + '#' * 18)
 
    def save_checkpoints(self,checkpoints_dir,epoch,i,avg_train_loss,is_best):
        checkpoint = Checkpoint(start_epoch=epoch+1,
                                train_loss=avg_train_loss,
                                best_loss=self.best_loss,
                                state_dict=self.model.module.state_dict(),
                                optimizer=self.optimizer.state_dict(),
                                scaler=self.scaler.state_dict(),
                                )
        model_name = checkpoints_dir + '{}-{}-val.ckpt'.format(epoch + 1, i + 1)
        best_model = checkpoints_dir + 'best.ckpt'
        if is_best:
            checkpoint.save(is_best, best_model)
        if not config['SAVE_BEST_ONLY']:
            checkpoint.save(False, model_name)


def load_train_objs(config):    ### define your train obj 如果有需要补充的，如：warmup_scheduler，可增加返回值，在初始化Trainer时传入
 
    train_set = RandomDataset()  ########################################## load your train dataset
    eval_set = RandomDataset()

    model = TempleteModel()  ####################################### load your model
    # model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['LR']) ############ chooose your optimzer 
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( ##################### init lr_scheduler
        optimizer,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        threshold=0.001,
        verbose=True,
    )
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.99)
    criterion = TempleteLoss() ########## your loss function object
    
    return train_set,eval_set, model,  criterion ,optimizer,lr_scheduler
 
def prepare_dataloader(train_set,eval_set,config):
    trainloader = DataLoader(
        train_set,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORK"],
        pin_memory=True,
        sampler=DistributedSampler(train_set,seed=config['SEEDS']['train'],shuffle=True),
        persistent_workers=False,
        worker_init_fn = lambda x: np.random.seed(config['SEEDS']['train'] + x),
        #collate_fn=TrainCollate()
    )
    evalloader = DataLoader(
        eval_set,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORK"],
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(eval_set,seed=config['SEEDS']['eval'],shuffle=False),
        persistent_workers=False,
        worker_init_fn = lambda x: np.random.seed(config['SEEDS']['eval'] + x),
        #collate_fn=TrainCollate()
    )

    return trainloader,evalloader
 
def set_seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config, model_name,test_run=False):
    seed_everything = config['seed_everything']
    if seed_everything:
        set_seed_everything(seed_everything)
    ddp_setup()
    train_set,eval_set,model, criterion,optimizer,lr_scheduler = load_train_objs(config)
    train_data,eval_data = prepare_dataloader(train_set,eval_set,config)
    trainer = Trainer(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        criterion = criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_every=config["SAVE_EVERY"], ############################# val and save checkpoints every {config["SAVE_EVERY"]} steps
        model_dir = model_dir,
        model_name = model_name,
        sample_rate= config['sample_rate'],
        log_dir=log_dir,
        test_run = test_run
    )
    
    trainer.train(config)
    destroy_process_group()
    
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("-m", "--model_name", help="trained model name, retrain if no input", default='none')
    parser.add_argument("-y", "--yaml_name", help="config file name")
    parser.add_argument("-t", "--test_run", help="fast dev run",type=bool,default=False)
    args = parser.parse_args()
 
    # Loading configs 
    _abspath = Path(os.path.abspath(__file__)).parent
    _yaml_path = os.path.join(_abspath, 'configs/' + args.yaml_name)
    try:
        with open(_yaml_path, 'r') as f_yaml:
            config = yaml.load(f_yaml, Loader=yaml.FullLoader)
    except:
        raise ValueError('No config file found at "%s"' % _yaml_path)
    
    model_dir = config['OUTPUT_DIR'] + config['WORKSPACE'] + "/checkponints/"
    log_dir = config['OUTPUT_DIR'] + config['WORKSPACE'] + "/logs/"
    if not os.path.exists(model_dir) and int(os.environ["RANK"]) == 0:
        os.makedirs(model_dir)
    if not os.path.exists(log_dir) and int(os.environ["RANK"]) == 0:
        os.makedirs(log_dir)
    shutil.copy(_yaml_path,config['OUTPUT_DIR'] + config['WORKSPACE']+"/config.yaml")

    main(config,model_name=args.model_name,test_run=args.test_run)
 
 



'''
### 运行命令 ###
    torchrun --nproc_per_node={该机器上要用多少张卡} --nnodes={机器总数} --node_rank={主机为0，从机为n} --master_addr="{主机ip(需要主机和从机在同一网段)}" --master_port={主机空闲端口} train_ddp.py --train_script_arg1 value_of_arg1 --train_script_arg2 value_of_arg2 ...
'''