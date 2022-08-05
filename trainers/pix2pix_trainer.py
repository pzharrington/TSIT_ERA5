import os, sys, time
import torch
from models.pix2pix_model import Pix2PixModel
from models.afnonet import AFNONet, PrecipNet, load_afno
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.functional as TF
import wandb
from utils.data_loader_multifiles import get_data_loader
from utils.weighted_acc_rmse import weighted_acc_torch_channels, \
    weighted_rmse_torch_channels, unlog_tp_torch
from utils.spectra_metrics import spectra_metrics_rfft, \
    spectra_metrics_fft_input
from utils.viz import *
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
        self.sub_dir = args.sub_dir
        self.config = args.config
        self.best_acc_overall = 0.
        if self.sub_dir is not None:
            self.root_dir = os.path.join(self.root_dir, self.sub_dir)

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
        self.afno_validate = params.afno_validate and self.world_rank==0
        params.name = args.config

        self.device = torch.cuda.current_device()

        # load climatology
        self.tp_tm = torch.as_tensor(np.load(params.precip_time_means)).to(self.device)
        if self.tp_tm.shape[-2] != params.img_size[0] or \
           self.tp_tm.shape[-1] != params.img_size[1]:
            old_shape = self.tp_tm
            self.tp_tm = TF.resize(self.tp_tm, params.img_size)
            if self.log_to_screen:
                logging.info(f'Resized precip means from {old_shape} to {self.tp_tm.shape}')

        # add grid channels to input_nc
        if params.add_grid:
            params.input_nc += params.N_grid_channels

        if params.orography:
            params.input_nc += 1

        self.params = params


    def build_and_launch(self):
        self.build()
        # launch training
        self.train()


    def build(self):

        jid = os.environ['SLURM_JOBID']

        # init wandb
        if self.log_to_wandb:
            if self.sweep_id:
                wandb_dir = os.environ['WANDB_DIR']
                run_id = os.environ['WANDB_RUN_ID']
                exp_dir = os.path.join(*[wandb_dir, run_id])
                if not os.path.isdir(exp_dir):
                    os.makedirs(exp_dir)
                wandb.init(config=self.params.params, resume=self.params.resuming, dir=exp_dir)
                assert run_id == wandb.run.id, f'$WANDB_RUN_ID = {run_id} but wandb.run.id = {wandb.run.id}'
                hpo_config = wandb.config
                self.params.update_params(hpo_config)
                logging.info('HPO sweep %s, run ID %s, trial params:'%(self.sweep_id, run_id))
                logging.info(self.params.log())
                if self.params.resuming:
                    wandb.mark_preempting()
            else:
                exp_dir = os.path.join(*[self.root_dir, 'expts', self.config])
                if not os.path.isdir(exp_dir):
                    os.makedirs(exp_dir)
                wandb.init(config=self.params.params, name=self.params.name, project=self.params.project,
                           entity=self.params.entity, resume=self.params.resuming, dir=exp_dir)
        elif self.sweep_id:
            exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.config, jid])
        else:
            exp_dir = os.path.join(*[self.root_dir, 'expts', self.config])

        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
            if not os.path.isdir(os.path.join(exp_dir, 'checkpoints/')):
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))

        self.params.experiment_dir = os.path.abspath(exp_dir)
        self.params.checkpoint_path = os.path.join(exp_dir, 'checkpoints/ckpt.tar')

        if self.sweep_id and dist.is_initialized():
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            assert self.world_rank == rank
            if rank != 0:
                self.params = None
            # Broadcast sweep config
            self.params = comm.bcast(self.params, root=0)

        # need initial value of resuming to be True even if it's the first run,
        # otherwise no wandb-resume.json gets created
        self.params.resuming = self.params.resuming and os.path.isfile(self.params.checkpoint_path)

        # -- after here params should be finalized
        self.params.update_params({ 'afno_validate': self.afno_validate })

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

        if self.afno_validate or self.params.train_on_afno_wind:

            # backbone model
            self.params.N_in_channels = self.params.afno_wind_N_channels
            self.params.N_out_channels = self.params.afno_wind_N_channels
            afno_wind = AFNONet(self.params).to(self.device)
            afno_wind = load_afno(afno_wind,
                                  self.params,
                                  self.params.afno_model_wind_path) # ,
                                       # map_location=torch.device('cpu'))
            self.afno_wind = afno_wind.to(self.device)

            # precip model
            if self.afno_validate:
                self.params.N_out_channels = len(self.params.out_channels)
                afno_precip = AFNONet(self.params).to(self.device)
                afno_precip = PrecipNet(self.params, backbone=afno_precip).to(self.device)
                afno_precip = load_afno(afno_precip,
                                        self.params,
                                        self.params.afno_model_precip_path) # ,
                                        # map_location=torch.device('cpu'))
                self.afno_precip = afno_precip.to(self.device)

        if self.params.amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()

        self.iters = 0
        self.startEpoch = 0

        # self.params.resuming is False if there's no checkpoint file, in which
        # case we can load the pretrained model, if using one
        if self.params.resuming:
            logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)
        elif self.params.pretrained:
            assert os.path.isfile(self.params.pretrained_model_path), \
                f'There is no pretrained model file {self.params.pretrained_model_path}'
            logging.info("Loading pretrained model %s"%self.params.pretrained_model_path)
            self.restore_checkpoint(self.params.pretrained_model_path,
                                    pretrained_model=True,
                                    pretrained_same_arch=self.params.pretrained_same_arch)

        self.epoch = self.startEpoch

        self.logs = {}


    def train(self):
        if self.log_to_screen:
            logging.info("Starting Training Loop...")
        for epoch in range(self.startEpoch, self.params.niter+self.params.niter_decay):
            self.epoch = epoch
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)
            # for keeping track of niters_l1_pretrain
            self.pix2pix_model.set_epoch(epoch)

            start = time.time()
            tr_time = self.train_one_epoch()
            valid_time, fields, spectra = self.validate_one_epoch()
            self.schedulerG.step()
            if self.schedulerD is not None:
                self.schedulerD.step()

            if self.world_rank == 0:
                is_best = self.logs['acc_overall'] >= self.best_acc_overall
                if is_best:
                    self.best_acc_overall = self.logs['acc_overall']

                if self.params.save_checkpoint:
                    #checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path, is_best=is_best)

            if self.log_to_wandb:
                try:
                    fig = viz_fields(fields)
                    self.logs['viz'] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as inst:
                    logging.warning(f'viz_fields threw {type(inst)}!\n{str(inst)}')
                try:
                    fig = viz_density(fields)
                    self.logs['viz_density'] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as inst:
                    logging.warning(f'viz_density threw {type(inst)}!\n{str(inst)}')
                try:
                    fig = viz_spectra(spectra)
                    self.logs['viz_spec'] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as inst:
                    logging.warning(f'viz_spectra threw {type(inst)}!\n{str(inst)}')

                self.logs['learning_rate_G'] = self.optimizerG.param_groups[0]['lr']
                self.logs['best_acc_overall'] = self.best_acc_overall
                self.logs['epoch'] = self.epoch + 1
                wandb.log(self.logs)

            if self.params.DEBUG:
                logging.info(f'Rank {self.world_rank} | acc = {self.logs["acc"]}')

            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(self.epoch+1, time.time()-start))
                logging.info('Train time = {}, Valid time = {}'.format(tr_time, valid_time))
                logging.info('G losses = '+str(self.g_losses))
                logging.info('D losses = '+str(self.d_losses))
                logging.info('ACC = %f'%self.logs['acc'])
                logging.info('ACC overall = %f'%self.logs['acc_overall'])
                logging.info('ACC overall best = %f'%self.best_acc_overall)
                logging.info('RMSE = %f'%self.logs['rmse'])
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
        afno_time = 0.
        n_wind = self.params.afno_wind_N_channels
        for i, (image, target) in enumerate(self.train_data_loader, 0):
            self.iters += 1
            timer = time.time()
            data = (image.to(self.device), target.to(self.device))
            data_time += time.time() - timer
            self.pix2pix_model.zero_all_grad()

            timer = time.time()
            if self.params.train_on_afno_wind:
                with torch.no_grad():
                    afno_pred = self.afno_wind(data[0][:, :n_wind])
                    if self.params.add_grid:
                        afno_pred = torch.cat([afno_pred, data[0][:, n_wind:]], dim=1)
                    data = (afno_pred, data[1])
            afno_time += time.time() - timer

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

            if self.params.amp:
                self.grad_scaler.update()

            if self.params.log_steps_to_screen and (i % self.params.log_every_n_steps) == 0:
                logging.info(f'Rank {self.world_rank} | B: {i+1}/{len(self.train_data_loader)} | '
                             f'G: {self.g_losses} | D: {self.d_losses}')

            if self.params.DEBUG and i > 0 and (i % (self.params.log_every_n_steps * 4)) == 0:
                break

        tr_time = time.time() - tr_start

        if self.log_to_screen: logging.info('Total=%f, G=%f, D=%f, afno=%f, data=%f, next=%f'%(tr_time, g_time, d_time, afno_time, data_time, tr_time - (g_time+ d_time + afno_time + data_time)))
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
        afno_preds = []
        acc = []
        rmse = []
        afno_acc = []
        afno_rmse = []
        amps = {}
        spec_metrics = {}
        # inps = []
        n_wind = self.params.afno_wind_N_channels
        nc, iw ,ih = self.params.output_nc, self.params.img_size[0], self.params.img_size[1]
        loop = time.time()
        acctime = 0.
        g_time = 0.
        data_time = 0.
        spectime = 0.
        afnotime = 0.
        # with torch.no_grad():
        with torch.inference_mode():
            for idx, (image, target) in enumerate(self.valid_data_loader):
                timer = time.time()
                assert image.shape[1] == self.params.input_nc, f'image.shape: {image.shape}'
                data = (image.to(self.device), target.to(self.device))
                tar_unlog = unlog_tp_torch(data[1], self.params.precip_eps)
                data_time += time.time() - timer

                timer = time.time()
                if self.afno_validate or self.params.train_on_afno_wind:
                    afno_wind_pred = self.afno_wind(data[0][:, :n_wind])
                    if self.afno_validate:
                        afno_pred = self.afno_precip(afno_wind_pred)
                        afno_preds.append(afno_pred.detach().cpu())
                        amps.setdefault('afno', []).append(
                            torch.fft.rfft(afno_pred, dim=-1, norm='ortho')[:, 0].abs().mean(dim=-2).detach().cpu()
                        )
                        afno_unlog = unlog_tp_torch(afno_pred, self.params.precip_eps)
                        afno_acc.append(weighted_acc_torch_channels(afno_unlog - self.tp_tm,
                                                                    tar_unlog - self.tp_tm))
                        afno_rmse.append(weighted_rmse_torch_channels(afno_unlog, tar_unlog))
                    if self.params.train_on_afno_wind:
                        if self.params.add_grid:
                            afno_wind_pred = torch.cat([afno_wind_pred, data[0][:, n_wind:]], dim=1)
                        data = (afno_wind_pred, data[1])
                afnotime += time.time() - timer

                timer = time.time()
                gen = self.generate_validation(data)
                assert gen.shape[1] == 1, f'gen.shape: {gen.shape}'

                g_time += time.time() - timer
                timer = time.time()
                gen_unlog = unlog_tp_torch(gen, self.params.precip_eps)
                acc.append(weighted_acc_torch_channels(gen_unlog - self.tp_tm,
                                                       tar_unlog - self.tp_tm))
                rmse.append(weighted_rmse_torch_channels(gen_unlog, tar_unlog))
                acctime += time.time() - timer

                timer = time.time()
                spec_dict = spectra_metrics_rfft(gen, data[1])
                amps.setdefault('tsit', []).append(spec_dict.pop('pred_fft'))
                amps.setdefault('era5', []).append(spec_dict.pop('tar_fft'))
                weighted_spec_dict = spectra_metrics_fft_input(amps['tsit'][-1],
                                                               amps['era5'][-1],
                                                               freq_weighting=True,
                                                               lat_weighting=True)
                # mean fft across latitude
                amps['tsit'][-1] = amps['tsit'][-1][:, 0].abs().mean(dim=-2).detach().cpu()
                amps['era5'][-1] = amps['era5'][-1][:, 0].abs().mean(dim=-2).detach().cpu()
                for key, metric in spec_dict.items():
                    spec_metrics.setdefault(key, []).append(metric)
                    spec_metrics.setdefault('weighted_'+key, []).append(weighted_spec_dict[key])
                spectime += time.time() - timer

                preds.append(gen.detach().cpu())
                targets.append(data[1].detach().cpu())
                # inps.append(inp.detach().cpu())

        timer = time.time()
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        acc = torch.cat(acc)
        rmse = torch.cat(rmse)
        # inps = torch.cat(inps)
        for key, metric in spec_metrics.items():
            spec_metrics[key] = torch.cat(metric).mean().item()

        if self.afno_validate:
            afno_preds = torch.cat(afno_preds)
            afno_acc = torch.cat(afno_acc)
            afno_rmse = torch.cat(afno_rmse)

        if dist.is_initialized():
            # average acc across ranks
            sz = torch.tensor(preds.shape[0]).float().to(self.device)
            sz_overall = torch.clone(sz)
            dist.all_reduce(sz_overall)
            acc_overall = acc.mean() * sz
            dist.all_reduce(acc_overall)
            acc_overall = acc_overall.item() / sz_overall.item()
        else:
            acc_overall = acc.mean().item()

        sample_idx = np.random.randint(max(preds.size()[0], targets.size()[0]))

        if self.afno_validate:
            afno_samp = afno_preds[sample_idx].detach().cpu().numpy()
        else:
            afno_samp = None

        fields = [preds[sample_idx].detach().cpu().numpy(),
                  targets[sample_idx].detach().cpu().numpy(),
                  # inps[sample_idx].detach().cpu().numpy(),
                  afno_samp]

        spectra_mean = {}
        spectra_std = {}

        # summarize spectra
        for key, amp in amps.items():
            amp = torch.cat(amp)
            # calc mean and standard error across validation obs
            std_mean = torch.std_mean(amp, dim=0)
            spectra_mean[key] = std_mean[1].detach().cpu().numpy()
            spectra_std[key] = std_mean[0].detach().cpu().numpy() # / np.sqrt(amp.shape[0])

        spectra = [spectra_mean, spectra_std, acc.shape[0]]

        valid_time = time.time() - valid_start
        self.logs.update({'acc': acc.mean().item()})
        self.logs.update({'rmse': rmse.mean().item()})
        self.logs.update({'acc_overall': acc_overall})
        if self.afno_validate:
            self.logs.update({'acc_afno': afno_acc.mean().item()})
            self.logs.update({'rmse_afno': afno_rmse.mean().item()})
            # log fixed overall afno validation acc
            self.logs.update({'acc_afno_overall': self.params.afno_acc_overall})
        self.logs.update(spec_metrics)
        agg = time.time() - timer
        if self.log_to_screen: logging.info('Total=%f, G=%f, data=%f, acc=%f, spec=%f, afno=%f, agg=%f, next=%f'%(valid_time, g_time, data_time, acctime, spectime, afnotime, agg, valid_time - (g_time+ data_time + acctime + spectime + afnotime + agg)))

        return valid_time, fields, spectra


    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.pix2pix_model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 'acc_overall': self.logs['acc_overall'], 'best_acc_overall': self.best_acc_overall,
                    'model_state_G': model.save_state('generator'), 'model_state_D': model.save_state('discriminator'), 'model_state_E': model.save_state('encoder'),
                    'optimizerG_state_dict': self.optimizerG.state_dict(),
                    'schedulerG_state_dict': self.schedulerG.state_dict(),
                    'optimizerD_state_dict': self.optimizerD.state_dict() if self.optimizerD is not None else None,
                    'schedulerD_state_dict': self.schedulerD.state_dict() if self.schedulerD is not None else None,
                    'scaler_state_dict': self.grad_scaler.state_dict() if self.params.amp else None},
                   checkpoint_path)
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'acc_overall': self.logs['acc_overall'],
                        'model_state_G': model.save_state('generator'), 'model_state_D': model.save_state('discriminator'), 'model_state_E': model.save_state('encoder'),
                        'optimizerG_state_dict': self.optimizerG.state_dict(),
                        'schedulerG_state_dict': self.schedulerG.state_dict(),
                        'optimizerD_state_dict': self.optimizerD.state_dict() if self.optimizerD is not None else None,
                        'schedulerD_state_dict': self.schedulerD.state_dict() if self.schedulerD is not None else None,
                        'scaler_state_dict': self.grad_scaler.state_dict() if self.params.amp else None},
                       checkpoint_path.replace('.tar', '_best.tar'))

    def restore_checkpoint(self, checkpoint_path, pretrained_model=False, pretrained_same_arch=True):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank))
        if 'best_acc_overall' in checkpoint.keys():
            self.best_acc_overall = checkpoint['best_acc_overall']
        if not dist.is_initialized():
            # remove DDP 'module' prefix if not distributed
            for key in checkpoint.keys():
                if 'model_state' in key and checkpoint[key] is not None:
                    consume_prefix_in_state_dict_if_present(checkpoint[key], 'module.')
        self.pix2pix_model.load_state(checkpoint, pretrained_model, pretrained_same_arch)
        if not pretrained_model:
            self.iters = checkpoint['iters']
            self.startEpoch = checkpoint['epoch'] + 1
            self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
            self.schedulerG.load_state_dict(checkpoint['schedulerG_state_dict'])
            if not self.params.no_gan_loss:
                self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
                self.schedulerD.load_state_dict(checkpoint['schedulerD_state_dict'])
            if self.params.amp and 'scaler_state_dict' in checkpoint.keys() and checkpoint['scaler_state_dict'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['scaler_state_dict'])

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
            # self.grad_scaler.update()
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
            self.grad_scaler.step(self.optimizerD)
            # self.grad_scaler.update()
        else:
            d_loss.backward()
            self.optimizerD.step()


    def get_latest_generated(self):
        return self.generated

    def generate_validation(self, data):
        generated, _ = self.pix2pix_model.generate_fake(data[0], data[1], validation=True)
        return generated
