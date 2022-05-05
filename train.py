import os
import sys
from Utils.train_utils import *
from Utils.train_utils import load_yaml_config, set_model, set_loss, set_optimization
import shutil
import torch
#from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


def main():
    save_path = config['save_path']
    os.makedirs(save_path)

    yaml_archive = os.path.join(save_path, 'config.yaml')
    shutil.copyfile(yaml_file, yaml_archive)

    writer = SummaryWriter(save_path)

    # load CUDA
    #cuda = torch.cuda.is_available()
    #print(f'cuda: {cuda}')
    #cuda = False
    device = set_device(config)
    logger.info('Device: %s' % (device,))
    torch.manual_seed(1)

    criterion = set_loss(config)
    model, start_epoch = set_model(config)
    optimizer = set_optimization(config, model)

    train_dataloader, valid_dataloader, test_dataloader, external_loader = set_dataloader(config)

    if config['model']['mode'] in ['train', 'resume']:
        for epoch in range(start_epoch, config['model']['epoch_num']):
            train_loss, model, optimizer = train_epoch(
                config,
                model,
                epoch,
                train_dataloader,
                optimizer,
                criterion,
                device)

            valid_loss = valid_epoch(
                config,
                model,
                epoch,
                valid_dataloader,
                criterion,
                device)

            writer.add_scalar("Loss/train_mse_loss", train_loss, epoch)
            writer.add_scalar("Loss/valid_mse_loss", valid_loss, epoch)
            writer.flush()

            if epoch % config['model']['checkpoint_interval'] == 0:
                save_state(epoch, model, optimizer, save_path, 'checkpoint_%s.tar' % (epoch,))

        writer.close()
        save_state(epoch, model, optimizer, save_path, 'last_model.tar')


def load_pretrained_model(params):
    model = set_model(params)
    # summary(model, (3, 224, 224))
    print(model)


if __name__ == '__main__':
    yaml_file =sys.argv[1]
    config = load_yaml_config(yaml_file)
    config['data']['mode'] = 'train'
    config['model']['mode'] = 'train'

    # load_pretrained_model(config)
    main()