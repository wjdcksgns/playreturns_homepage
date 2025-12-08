import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import requests
import numpy as np
import torch
import torch.distributed as dist
import yaml

from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_amp, check_file, check_img_size,
                           check_suffix, check_yaml, colorstr, increment_path, init_seeds, 
                           intersect_dicts, labels_to_class_weights, methods, strip_optimizer)
from utils.loggers import Loggers
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class Train():
    
    def __init__(self):
        self.epochs = 0                 # 전체 epochs 수
        self.train_loader_length = 0    # batch_size가 적용된 train_loader의 length
        self.batch_size = 0             # batch_size
        self.present_progress = 0       # 현재 epoch에 맞게 누적된 train data length
        self.admin_path = ''            # train에 사용될 전체적인 base 경로


    # def get_train_info(self, data):
    #     '''
    #     self.total_length = self.train_loader_length * self.batch_size * self.epochs
    #     present_percent : (현재 사용된 데이터 누적 개수 / total_length) 값을 소수점 셋째 자리까지 변환 후 * 100 진행
    #     self.present_progress : 매 epoch마다 train_loader_length 누적해서 보관하는 변수
    #     '''
    #     data = data
    #     present_percent = round(self.present_progress / self.total_length, 3) * 100

    #     self.present_progress += self.train_loader_length * self.batch_size

    #     return present_percent, data


    def train(self, hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary

        save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, train_path, val_path, nc, names, maps_url, pid_url, obj_uuid = \
            Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.train_path, opt.valid_path, opt.nc, opt.names, opt.maps_url, opt.pid_url, opt.obj_uuid
        callbacks.run('on_pretrain_routine_start')

        proc_pid = os.getpid()
        print('train PID:', proc_pid)

        obj_uuid = obj_uuid.split(',')

        requests.post(pid_url, json={'pid': proc_pid, 'uuid': obj_uuid})

        val_path = val_path.split(',')
        train_path = train_path.split(',')
        names = names.split(',')

        names_dict = dict()
        for i in range(len(names)):
            names_dict[i] = names[i]

        self.batch_size = batch_size
        self.epochs = epochs

        # Directories
        w = save_dir / 'weights'  # weights dir
        (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
        last, best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        opt.hyp = hyp.copy()  # for saving hyps to checkpoints

        # Loggers
        data_dict = None
        if RANK in {-1, 0}:
            loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

            # Register actions
            for k in methods(loggers):
                callbacks.register_action(k, callback=getattr(loggers, k))

            # Process custom dataset artifact link
            data_dict = loggers.remote_dataset

        # Config
        # data.yaml에 대한 정보를 확인 / 이 부분에서 object_data.csv에 따른 yaml 파일 수정이 필요
        # util.general.py 492 line으로 인해 nc는 yaml 파일에서 따로 수정이 가능한 것이 아닌, names의 개수에 맞춰 설정된다.
        # plots = not evolve and not opt.noplots  # create plots
        plots = False
        cuda = device.type != 'cpu'
        init_seeds(opt.seed + 1 + RANK, deterministic=True)

        data_dict = dict() 
        data_dict['nc'] = nc
        
        # Model
        check_suffix(weights, '.pt')  # check weights

        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report

        amp = check_amp(model)  # check AMP

        # Freeze
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers

        # Image size
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

        # Scheduler
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        ema = ModelEMA(model) if RANK in {-1, 0} else None

        # Resume
        best_fitness, start_epoch = 0.0, 0

        del ckpt, csd

        # DP mode
        if cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        if opt.sync_bn and cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        train_loader, dataset = create_dataloader(train_path,
                                                imgsz,
                                                batch_size // WORLD_SIZE,
                                                gs,
                                                single_cls,
                                                hyp=hyp,
                                                augment=True,
                                                cache=None if opt.cache == 'val' else opt.cache,
                                                rect=opt.rect,
                                                rank=LOCAL_RANK,
                                                workers=workers,
                                                image_weights=opt.image_weights,
                                                quad=opt.quad,
                                                prefix=colorstr('train: '),
                                                shuffle=True,
                                                user_uuid_list = obj_uuid,
                                                save_dir=str(save_dir).replace('result', 'cache\\train'))

        self.train_loader_length = len(train_loader)
        self.total_length = self.train_loader_length * self.batch_size * self.epochs

        labels = np.concatenate(dataset.labels, 0)
        mlc = int(labels[:, 0].max())  # max label class
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

        # Process 0
        if RANK in {-1, 0}:
            val_loader = create_dataloader(val_path,
                                        imgsz,
                                        batch_size // WORLD_SIZE * 2,
                                        gs,
                                        single_cls,
                                        hyp=hyp,
                                        cache=None if noval else opt.cache,
                                        rect=True,
                                        rank=-1,
                                        workers=workers * 2,
                                        pad=0.5,
                                        prefix=colorstr('val: '),
                                        user_uuid_list = obj_uuid,
                                        save_dir=str(save_dir).replace('result', 'cache\\valid'))[0]

            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor

            model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end', labels, names_dict)

        # DDP mode
        if cuda and RANK != -1:
            model = smart_DDP(model)

        # Model attributes
        nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights

        model.names = names_dict

        # Start training
        t0 = time.time()
        nb = len(train_loader)  # number of batches
        nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)

        last_opt_step = -1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        stopper, stop = EarlyStopping(patience=opt.patience), False
        compute_loss = ComputeLoss(model)  # init loss class
        callbacks.run('on_train_start')
        LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                    f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                    f"Logging results to {colorstr('bold', save_dir)}\n"
                    f'Starting training for {epochs} epochs...')
        
        
        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------

            callbacks.run('on_train_epoch_start')

            model.train()

            mloss = torch.zeros(3, device=device)  # mean losses
            if RANK != -1:
                train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(train_loader)
            LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
            if RANK in {-1, 0}:
                pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

            optimizer.zero_grad()

            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                callbacks.run('on_train_batch_start')
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:

                    xi = [0, nw]  # x interp
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Forward
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

                # Backward
                scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= accumulate:
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    last_opt_step = ni

                # Log
                if RANK in {-1, 0}:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                        (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                    callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                    if callbacks.stop_training:
                        return
                # end batch ------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            scheduler.step()

            if RANK in {-1, 0}:
                # mAP
                callbacks.run('on_train_epoch_end', epoch=epoch)
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
                if not noval or final_epoch:  # Calculate mAP
                    results, maps, _ = validate.run(data_dict,
                                                    batch_size=batch_size // WORLD_SIZE * 2,
                                                    imgsz=imgsz,
                                                    half=amp,
                                                    model=ema.ema,
                                                    single_cls=single_cls,
                                                    dataloader=val_loader,
                                                    save_dir=save_dir,
                                                    plots=False,
                                                    callbacks=callbacks,
                                                    compute_loss=compute_loss)
                    
                    result = np.array(results).reshape(1, -1)[0][3]
                    # result: all 객체에 대한 mAP 값 / type: float

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                stop = stopper(epoch=epoch, fitness=fi)  # early stop check
                if fi > best_fitness:
                    best_fitness = fi
                log_vals = list(mloss) + list(results) + lr
                callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

                # Save model
                if (not nosave) or (final_epoch and not evolve):  # if save
                    ckpt = {
                        'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'opt': vars(opt),
                        'date': datetime.now().isoformat()}

                    # Save last, best and delete
                    torch.save(ckpt, last)

                    if best_fitness == fi:
                        torch.save(ckpt, best)
                    if opt.save_period > 0 and epoch % opt.save_period == 0:
                        torch.save(ckpt, w / f'epoch{epoch}.pt')
                    del ckpt
                    callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # EarlyStopping
            if RANK != -1:  # if DDP training
                broadcast_list = [stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    stop = broadcast_list[0]
            if stop:
                break  # must break all DDP ranks

            # end epoch ----------------------------------------------------------------------------------------------------
        # end training -----------------------------------------------------------------------------------------------------
        if RANK in {-1, 0}:
            LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
                    if f is best:
                        LOGGER.info(f'\nValidating {f}...')
                        results, maps, _ = validate.run(
                            data_dict,
                            batch_size=batch_size // WORLD_SIZE * 2,
                            imgsz=imgsz,
                            model=attempt_load(f, device).half(),
                            iou_thres=0.60,
                            single_cls=single_cls,
                            dataloader=val_loader,
                            save_dir=save_dir,
                            verbose=True,
                            plots=plots,
                            callbacks=callbacks,
                            compute_loss=compute_loss)  # val best model with plots

            callbacks.run('on_train_end', last, best, epoch, results)

            # maps: 각 객체에 대한 최종 AP 값 / type: np.array / array([float, float, ...])
            
            requests.post(maps_url, json={"maps": maps.tolist(), 'uuid': obj_uuid})

        torch.cuda.empty_cache()
        return results


    def parse_opt(self, known=False):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
        parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
        parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
        parser.add_argument('--noplots', action='store_true', help='save no plot files')
        parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
        parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
        parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
        parser.add_argument('--seed', type=int, default=0, help='Global training seed')
        parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

        # Logger arguments
        parser.add_argument('--entity', default=None, help='Entity')
        parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
        parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
        parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

        parser.add_argument('--obj_uuid', type=str, help='object uuid list')
        parser.add_argument('--maps_url', type=str, help='maps post api url')
        parser.add_argument('--pid_url', type=str, help='pid post api url')

        parser.add_argument('--train_path', type=str, help='base path for train')
        parser.add_argument('--valid_path', type=str, help='base path for valid')
        parser.add_argument('--nc', type=int, help='object length')
        parser.add_argument('--names', type=str, help='object names')
        parser.add_argument('--admin_path', type=str, help='admin_path')
        opt = parser.parse_args()
        print('opt :', opt)

        return parser.parse_known_args()[0] if known else parser.parse_args()


    def main(self, opt, callbacks=Callbacks()):
        # Checks
        opt.data, opt.cfg, opt.hyp, opt.weights = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights) # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

        result_path = str(Path(opt.admin_path) / opt.name)
        if os.path.isdir(result_path):
            shutil.rmtree(result_path)
        
        opt.save_dir = result_path

        # DDP mode
        device = select_device(opt.device, batch_size=opt.batch_size)
        if LOCAL_RANK != -1:
            msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
            assert not opt.image_weights, f'--image-weights {msg}'
            assert not opt.evolve, f'--evolve {msg}'
            assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
            assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        # Train
        self.train(opt.hyp, opt, device, callbacks)


    def run(self, **kwargs):
        # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
        opt = self.parse_opt(True)
        for k, v in kwargs.items():
            setattr(opt, k, v)
        self.main(opt)

        return opt


if __name__ == "__main__":

    trainer = Train()
    opt = trainer.parse_opt()
    trainer.main(opt)
