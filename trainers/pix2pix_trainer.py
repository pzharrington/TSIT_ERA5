import os, sys, time
import torch
from models.pix2pix_model import Pix2PixModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.functional as TF
import wandb
from utils.data_loader_multifiles import get_data_loader
from utils.weighted_acc_rmse import weighted_acc_torch_channels, unlog_tp_torch
from utils.spectra_metrics import spectra_metrics_rfft, \
    spectra_metrics_fft_input
from utils.viz import viz_fields, viz_spectra
import numpy as np
import matplotlib.pyplot as plt
import logging
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

class Pix2PixTrainer():
    """
    Trainer object creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, params, args):

        self.sweep_id = args.sweep_id
        self.root_dir = args.root_dir
        self.config = args.config
        self.group = args.group
        if self.group is not None:
            self.root_dir = os.path.join(self.root_dir, self.group)

        params.amp = args.amp
        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        if self.world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(self.local_rank)
        torch.backends.cudnn.benchmark = True

        if self.world_rank==0:
            params.log()
        self.log_to_screen = params.log_to_screen and self.world_rank==0
        self.log_to_wandb = params.log_to_wandb and self.world_rank==0
        params.name = args.config

        # load climatology
        self.tp_tm = torch.as_tensor(np.load(params.precip_time_means))
        if self.tp_tm.shape != params.img_size:
            self.tp_tm = TF.resize(self.tp_tm, params.img_size)

        self.device = torch.cuda.current_device()
        self.params = params


    def build_and_launch(self):
        self.build()
        # launch training
        self.train()


    def build(self):

        # init wandb
        if self.log_to_wandb:
            if self.sweep_id:
                jid = os.environ['SLURM_JOBID']
                wandb.init()
                hpo_config = wandb.config
                self.params.update_params(hpo_config)
                logging.info('HPO sweep %s, job ID %d, trial params:'%(self.sweep_id, jid))
                logging.info(self.params.log())
            else:
                exp_dir = os.path.join(*[self.root_dir, 'expts', self.config])
                if not os.path.isdir(exp_dir):
                    os.makedirs(exp_dir)
                    os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                self.params.experiment_dir = os.path.abspath(exp_dir)
                self.params.checkpoint_path = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
                self.params.resuming = True if os.path.isfile(self.params.checkpoint_path) else False
                wandb.init(config=self.params.params, name=self.params.name, project=self.params.project, 
                           entity=self.params.entity, resume=self.params.resuming, group=self.group)

        # setup output dir
        if self.sweep_id:
            exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.config, jid])
        else:
            exp_dir = os.path.join(*[self.root_dir, 'expts', self.config])
        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))

        self.params.experiment_dir = os.path.abspath(exp_dir)
        self.params.checkpoint_path = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
        self.params.resuming = True if os.path.isfile(self.params.checkpoint_path) else False

        if self.sweep_id and dist.is_initialized():
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            assert self.world_rank == rank
            if rank != 0: 
                self.params = None
            # Broadcast sweep config -- after here params should be finalized
            self.params = comm.bcast(self.params, root=0)

        if self.world_rank == 0:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in self.params.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(self.params.experiment_dir, 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams,  hpfile)

        self.params.global_batch_size = self.params.batch_size
        self.params.local_batch_size = int(self.params.batch_size//self.world_size)

        logging.info('rank %d, begin data loader init'%self.world_rank)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, distributed=dist.is_initialized(), train=True)
        self.valid_data_loader, self.valid_dataset, self.valid_sampler = get_data_loader(self.params, distributed=dist.is_initialized(), train=False)
        logging.info('rank %d, data loader initialized'%self.world_rank)

        self.pix2pix_model = Pix2PixModel(self.params, dist.is_initialized(), self.local_rank, self.device)

        self.generated = None
        self.optimizerG, self.optimizerD = self.pix2pix_model.create_optimizers(self.params)
        # constant, then linear LR decay: chain schedules together
        constG = lr_scheduler.ConstantLR(self.optimizerG, factor=1., total_iters=self.params.niter)
        linearG = lr_scheduler.LinearLR(self.optimizerG, start_factor=1., end_factor=0., total_iters=self.params.niter_decay)
        self.schedulerG = lr_scheduler.SequentialLR(self.optimizerG, schedulers=[constG, linearG], milestones=[self.params.niter])

        if self.optimizerD is not None:
            linearD = lr_scheduler.LinearLR(self.optimizerD, start_factor=1., end_factor=0., total_iters=self.params.niter_decay)
            constD = lr_scheduler.ConstantLR(self.optimizerD, factor=1., total_iters=self.params.niter)
            self.schedulerD = lr_scheduler.SequentialLR(self.optimizerD, schedulers=[constD, linearD], milestones=[self.params.niter])
        else:
            self.schedulerD = None

        if self.params.amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()
        
        self.iters = 0
        self.startEpoch = 0

        if self.params.resuming:
            logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)

        self.epoch = self.startEpoch

        self.logs = {}


    def train(self):
        if self.log_to_screen:
            logging.info("Starting Training Loop...")
        best = 0.
        for epoch in range(self.startEpoch, self.params.niter+self.params.niter_decay):
            self.epoch = epoch
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)

            start = time.time()
            tr_time = self.train_one_epoch()
            valid_time, fields, spectra = self.validate_one_epoch()
            self.schedulerG.step()
            if self.schedulerD is not None:
                self.schedulerD.step()

            is_best = self.logs['acc'] >= best
            best = max(self.logs['acc'], best)

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    #checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path, is_best=is_best)

            if self.log_to_wandb:
                fig = viz_fields(fields)
                self.logs['viz'] = wandb.Image(fig)
                plt.close(fig)
                # fig = viz_spectra(spectra)
                # self.logs['viz_spec'] = wandb.Image(fig)
                # plt.close(fig)
                self.logs['learning_rate_G'] = self.optimizerG.param_groups[0]['lr']
                wandb.log(self.logs, step=self.epoch+1)

            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(self.epoch+1, time.time()-start))
                logging.info('Train time = {}, Valid time = {}'.format(tr_time, valid_time))
                logging.info('G losses = '+str(self.g_losses))
                logging.info('D losses = '+str(self.d_losses))
                logging.info('ACC = %f'%self.logs['acc'])
                logging.info('RMSE amp = %f'%self.logs['rmse_amp'])
                logging.info('RMSE phase = %f'%self.logs['rmse_phase'])
                logging.info('FFL = %f'%self.logs['ffl'])
                logging.info('FCL = %f'%self.logs['fcl'])

        if self.log_to_wandb:
            wandb.finish()

    def train_one_epoch(self):
        tr_time = 0
        self.pix2pix_model.set_train()
        batch_size = self.params.local_batch_size # batch size per gpu

        tr_start = time.time()
        g_time = 0.
        d_time = 0.
        data_time = 0.
        for i, (image, target) in enumerate(self.train_data_loader, 0):
            self.iters += 1
            timer = time.time()
            data = (image.to(self.device), target.to(self.device))
            data_time += time.time() - timer
            self.pix2pix_model.zero_all_grad()

            # Training
            # train generator
            timer = time.time()
            self.run_generator_one_step(data)
            g_time += time.time() - timer
            timer = time.time()
            if not self.params.no_gan_loss:
                self.run_discriminator_one_step(data)
            else:
                self.d_losses = {}
            d_time += time.time() - timer

            if self.params.DEBUG and (i % 375) == 0:
                logging.info(f'Rank {self.world_rank} | B: {i+1}/{len(self.train_data_loader)} | '
                             f'G: {self.g_losses} | D: {self.d_losses}')

        tr_time = time.time() - tr_start

        if self.log_to_screen: logging.info('Total=%f, G=%f, D=%f, data=%f, next=%f'%(tr_time, g_time, d_time, data_time, tr_time - (g_time+ d_time + data_time)))
        self.logs =  {**self.g_losses, **self.d_losses} 

        if dist.is_initialized():
            for key in self.logs.keys():
                if not torch.is_tensor(self.logs[key]):
                    continue
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        return tr_time


    def validate_one_epoch(self):
        self.pix2pix_model.set_eval()
        valid_start = time.time()
        preds = []
        targets = []
        acc = []
        ffts = {}
        spec_metrics = {}
        inps = []
        nc, iw ,ih = self.params.output_nc, self.params.img_size[0], self.params.img_size[1]
        loop = time.time()
        acctime = 0.
        g_time = 0.
        data_time = 0.
        with torch.no_grad():
            for idx, (image, target) in enumerate(self.valid_data_loader):
                timer = time.time()
                data = (image.to(self.device), target.to(self.device))
                data_time += time.time() - timer
                timer = time.time()
                gen = self.generate_validation(data)
                g_time += time.time() - timer
                timer = time.time()
                gen_unlog = unlog_tp_torch(gen, self.params.precip_eps)
                tar_unlog = unlog_tp_torch(data[1], self.params.precip_eps)
                tp_tm = self.tp_tm.to(tar_unlog.device)
                acc.append(weighted_acc_torch_channels(gen_unlog - tp_tm,
                                                       tar_unlog - tp_tm))
                acctime += time.time() - timer
                timer = time.time()
                spec_dict = spectra_metrics_rfft(gen, data[1])
                ffts.setdefault('pred', []).append(spec_dict.pop('pred_fft'))
                ffts.setdefault('tar', []).append(spec_dict.pop('tar_fft'))
                weighted_spec_dict = spectra_metrics_fft_input(ffts['pred'][-1],
                                                               ffts['tar'][-1],
                                                               freq_weighting=True,
                                                               lat_weighting=True)
                for key, metric in spec_dict.items():
                    spec_metrics.setdefault(key, []).append(metric)
                    spec_metrics.setdefault('weighted_'+key, []).append(weighted_spec_dict[key])
                spectime = time.time() - timer
                preds.append(gen.detach())
                targets.append(data[1].detach())
                inps.append(image.detach())

        timer = time.time()
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        pred_ffts = torch.cat(ffts['pred'])
        target_ffts = torch.cat(ffts['tar'])
        acc = torch.cat(acc)
        inps = torch.cat(inps)
        for key, metric in spec_metrics.items():
            spec_metrics[key] = torch.cat(metric).mean().item()
        '''
        # All-gather for full validation set currently OOMs
        if self.world_size > 1:
            # gather the sizes
            sz = torch.tensor(preds.shape[0]).float().to(self.device)
            sz_gather = [torch.zeros((1,)).float().to(self.device) for _ in range(self.world_size)]
            dist.all_gather(sz_gather, sz)
            # gather all the preds 
            preds_global = [torch.zeros(int(sz_loc.item()), nc, ih, iw).float().to(self.device) for sz_loc in sz_gather]
            dist.all_gather(preds_global, preds)
            preds = torch.cat([x for x in preds_global])
            targets_global = [torch.zeros(int(sz_loc.item()), nc, ih, iw).float().to(self.device) for sz_loc in sz_gather]
            dist.all_gather(targets_global, targets)
            targets = torch.cat([x for x in targets_global])
            acc_global = [torch.zeros(int(sz_loc.item()), nc).float().to(self.device) for sz_loc in sz_gather]
            dist.all_gather(acc_global, acc)
            acc = torch.cat([x for x in acc_global])
        '''
        sample_idx = np.random.randint(max(preds.size()[0], targets.size()[0]))
        fields = [preds[sample_idx].detach().cpu().numpy(),
                  targets[sample_idx].detach().cpu().numpy(),
                  inps[sample_idx].detach().cpu().numpy()]
        spectra = [pred_ffts[sample_idx].detach().cpu().numpy(),
                   target_ffts[sample_idx].detach().cpu().numpy()]

        valid_time = time.time() - valid_start
        self.logs.update({'acc': acc.mean().item()})
        self.logs.update(spec_metrics)
        agg = time.time() - timer 
        if self.log_to_screen: logging.info('Total=%f, G=%f, data=%f, acc=%f, spec=%f, agg=%f, next=%f'%(valid_time, g_time, data_time, acctime, spectime, agg, valid_time - (g_time+ data_time + acctime + spectime + agg)))

        return valid_time, fields, spectra


    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.pix2pix_model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 
                    'model_state_G': model.save_state('generator'), 'model_state_D': model.save_state('discriminator'), 'model_state_E': model.save_state('encoder'),
                    'optimizerG_state_dict': self.optimizerG.state_dict(), 'schedulerG_state_dict': self.schedulerG.state_dict(),
                    'optimizerD_state_dict': self.optimizerD.state_dict() if self.optimizerD is not None else None,
                    'schedulerD_state_dict': self.schedulerD.state_dict() if self.schedulerD is not None else None},
                   checkpoint_path)
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 
                        'model_state_G': model.save_state('generator'), 'model_state_D': model.save_state('discriminator'), 'model_state_E': model.save_state('encoder'),
                        'optimizerG_state_dict': self.optimizerG.state_dict(), 'schedulerG_state_dict': self.schedulerG.state_dict(),
                        'optimizerD_state_dict': self.optimizerD.state_dict() if self.optimizerD is not None else None,
                        'schedulerD_state_dict': self.schedulerD.state_dict() if self.schedulerD is not None else None},
                       checkpoint_path.replace('.tar', '_best.tar'))

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank))
        if not dist.is_initialized():
            # remove DDP 'module' prefix if not distributed
            for key in checkpoint.keys():
                if 'model_state' in key and checkpoint[key] is not None:
                    consume_prefix_in_state_dict_if_present(checkpoint[key], 'module.')
        self.pix2pix_model.load_state(checkpoint)
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch'] + 1
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.schedulerG.load_state_dict(checkpoint['schedulerG_state_dict'])
        if not self.params.no_gan_loss:
            self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
            self.schedulerD.load_state_dict(checkpoint['schedulerD_state_dict'])

    def run_generator_one_step(self, data):
        self.optimizerG.zero_grad()
        with torch.cuda.amp.autocast(self.params.amp):
            g_losses, generated = self.pix2pix_model.compute_generator_loss(data[0], data[1])
            g_loss = sum(g_losses.values()).mean()

            self.g_losses = {k: v.item() for k,v in g_losses.items()}
            self.generated = generated
            if self.params.amp:
                self.grad_scaler.scale(g_loss).backward()
                self.grad_scaler.step(self.optimizerG)
                self.grad_scaler.update()
            else:
                g_loss.backward()
                self.optimizerG.step()

    def run_discriminator_one_step(self, data):
        self.optimizerD.zero_grad()
        with torch.cuda.amp.autocast(self.params.amp):
            d_losses = self.pix2pix_model.compute_discriminator_loss(data[0], data[1])
            d_loss = sum(d_losses.values()).mean()
            self.d_losses = {k: v.item() for k,v in d_losses.items()}
            if self.params.amp:
                self.grad_scaler.scale(d_loss).backward()
                self.grad_scaler.step(self.optimizerG)
                self.grad_scaler.update()
            else:
                d_loss.backward()
                self.optimizerD.step()


    def get_latest_generated(self):
        return self.generated

    def generate_validation(self, data):
        generated, _ = self.pix2pix_model.generate_fake(data[0], data[1])
        return generated
