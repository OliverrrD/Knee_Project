import numpy as np
import os

import pandas as pd
import torch.nn
from torch import optim
from tqdm import tqdm
from Utils.misc import get_logger, load_json
import yaml
import json
from torchvision import models
from Data.covid_cxr_dataloader import ChestDataLoader
from torch.autograd import Variable
from Utils.misc import AverageMeterSet


logger = get_logger(os.path.basename(__file__))


def load_yaml_config(yaml_config):
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    experiment_name = os.path.basename(yaml_config).replace('.yaml', '')
    save_dir = os.path.join(config['env']['root_dir'], experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    num_existing_runs = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, f'run_{num_existing_runs}')
    config['save_path'] = save_path

    return config


def set_loss(params):
    return torch.nn.MSELoss()


def set_device(params):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def set_inference_model(params, load_model_path):
    assert params['model']['arch'] == 'densenet121'

    logger.info(f'Load model {load_model_path}')
    model = models.densenet121()
    last_in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(last_in_features, params['model']['output_channel'])
    model.load_state_dict(torch.load(load_model_path)['model'])
    return model


def set_model(params):
    assert params['model']['arch'] == 'densenet121'

    pretrained_checkpoint = params['model']['resume_path']
    if pretrained_checkpoint == '':
        model = models.densenet121(pretrained=True)
        # Change the last layer
        last_in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(last_in_features, params['model']['output_channel'])
        start_epoch = 0
    else:
        # TODO - get the pretraining work.
        assert os.path.exists(pretrained_checkpoint)
        logger.info(f'Loading pretrained model {pretrained_checkpoint}')
        checkpoint = torch.load(pretrained_checkpoint)
        model = checkpoint['model']
        start_epoch = 0

    return model, start_epoch


def set_optimization(params, model):
    # optimizer = optim.SGD(model.parameters(), lr=params['model']['base_lr'], momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=params['model']['base_lr'])
    return optimizer


def set_dataloader(params):
    train_loader = None
    valid_loader = None
    test_loader = None
    external_loader = None

    if params['data']['mode'] == 'train':
        train_loader = set_spec_dataloader(params, 'train')
        valid_loader = set_spec_dataloader(params, 'valid')

    if params['data']['mode'] == 'test':
        test_loader = set_spec_dataloader(params, 'test')

    if params['data']['mode'] == 'external':
        external_loader = set_spec_dataloader(params, 'external')

    return train_loader, valid_loader, test_loader, external_loader

def set_dataloader_train(params):
    train_loader = None
    valid_loader = None
    test_loader = None
    external_loader = None

    if params['data']['mode'] == 'train':
        train_loader = set_spec_dataloader_train(params, 'train')
        valid_loader = set_spec_dataloader(params, 'valid')

    if params['data']['mode'] == 'test':
        test_loader = set_spec_dataloader(params, 'test')

    if params['data']['mode'] == 'external':
        external_loader = set_spec_dataloader(params, 'external')

    return train_loader, valid_loader, test_loader, external_loader

def set_spec_dataloader(params, mode):
    dataset = ChestDataLoader(params, mode)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=params['data']['batch_size'] if mode == 'train' else 1,
        shuffle=True if mode == 'train' else False,
        pin_memory=True, num_workers=params['data']['num_workers'])

    return dataset_loader

def set_spec_dataloader_train(params, mode):
    dataset = ChestDataLoader(params, mode)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        shuffle=True if mode == 'train' else False,
        pin_memory=True, num_workers=params['data']['num_workers'])

    return dataset_loader

def train_epoch(params, model, epoch, train_dataloader, optimizer, criterion, device):
    meters = AverageMeterSet()

    model.train()
    model = model.to(device)

    max_iter = params['model']['epoch_num'] * len(train_dataloader)
    p_bar = tqdm(range(len(train_dataloader)))

    for index, data in enumerate(train_dataloader):
        cur_iter = epoch * len(train_dataloader) + index
        lr = params['model']['base_lr'] * (1 - float(cur_iter) / max_iter) ** 0.9

        inputs = Variable(data['image'].float().cuda())
        target = Variable(data['gt'].float().cuda())

        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        meters.update("loss_mse", loss.item(), params['data']['batch_size'])

        p_bar.set_description(
            "Train Epoch: {epoch:4}/{total_epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {mse_loss:3f}. LR: {lr:3f}".format(
                epoch=epoch + 1,
                total_epochs=params['model']['epoch_num'],
                batch=index + 1,
                iter=len(train_dataloader),
                mse_loss=meters['loss_mse'].avg,
                lr=lr
            )
        )
        p_bar.update()

    p_bar.close()
    return meters['loss_mse'].avg, model, optimizer


def valid_epoch(params, model, epoch, valid_dataloader, criterion, device):
    meters = AverageMeterSet()

    with torch.no_grad():
        model.eval()
        model = model.to(device)

        p_bar = tqdm(range(len(valid_dataloader)))

        for index, data in enumerate(valid_dataloader):
            inputs = Variable(data['image'].float().cuda())
            target = Variable(data['gt'].float().cuda())

            outputs = model(inputs)
            loss = criterion(outputs, target)

            meters.update('loss_mse', loss.item(), params['data']['batch_size'])

            p_bar.set_description(
                "Valid Epoch: {epoch:4}/{total_epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {mse_loss:3f}".format(
                    epoch=epoch + 1,
                    total_epochs=params['model']['epoch_num'],
                    batch=index + 1,
                    iter=len(valid_dataloader),
                    mse_loss=meters['loss_mse'].avg
                )
            )
            p_bar.update()

        p_bar.close()
        return meters['loss_mse'].avg


def inference(model, inference_dataloader, device):
    with torch.no_grad():
        model.eval()
        model = model.to(device)

        record_list = []

        p_bar = tqdm(range(len(inference_dataloader)))
        for index, data in enumerate(inference_dataloader):
            inputs = Variable(data['image'].float().cuda())

            outputs = model(inputs).data.cpu().numpy().squeeze().astype(np.float)
            target = data['gt'].numpy().squeeze().astype(np.float)

            record_dict = {
                'cxr_file_name': data['cxr_file_name'][0],
                'pred_dfa': outputs[0],
                'pred_pta': outputs[1],
                'pred_hka': outputs[2],
                'gt_dfa': target[0],
                'gt_pta': target[1],
                'gt_hka': target[2]
            }

            record_list.append(record_dict)

            p_bar.set_description(
                'Test. Iter: {index:4}/{iter:4}'.format(
                    index=index,
                    iter=len(inference_dataloader)
                )
            )
            p_bar.update()

        p_bar.close()
        record_df = pd.DataFrame(record_list)
        return record_df


def save_state(
        epoch: int,
        model,
        optimizer,
        save_path: str,
        filename: str = "best_model.tar",
):
    state_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer
    }
    file_path = os.path.join(save_path, filename)
    logger.info("Save current state to {}".format(filename))
    torch.save(state_dict, file_path)