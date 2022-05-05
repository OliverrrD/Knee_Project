import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from Utils.misc import get_logger
import yaml
from Utils.train_utils import set_device, set_dataloader, set_dataloader_train, inference, set_inference_model
from scipy.stats import pearsonr
import numpy as np
import statsmodels.api as stat
from sklearn.metrics import mean_squared_error
from Utils.misc import tflog2pandas
from glob import glob


logger = get_logger(os.path.basename(__file__))


def get_train_valid_loss_csv():
    tflog_path = glob(f'{save_path}/events.out.tfevents*')[0]
    train_valid_loss_df = tflog2pandas(tflog_path)

    out_csv = os.path.join(save_path, 'train_valid_loss.csv')
    logger.info(f'Save to {out_csv}')
    train_valid_loss_df.to_csv(out_csv, index=False)


def plot_training_curve():
    train_valid_csv = os.path.join(save_path, 'train_valid_loss.csv')
    logger.info(f'Load {train_valid_csv}')
    train_valid_df = pd.read_csv(train_valid_csv)

    train_df = train_valid_df.loc[train_valid_df['metric'] == 'Loss/train_mse_loss']
    valid_df = train_valid_df.loc[train_valid_df['metric'] == 'Loss/valid_mse_loss']

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(train_df['step'].to_list(), train_df['value'].to_list(), label='training')
    ax.plot(valid_df['step'].to_list(), valid_df['value'].to_list(), label='validation')
    ax.legend(loc='best')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    out_png = os.path.join(save_path, 'training_curve.png')
    logger.info(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)


def run_inference():
    yaml_config = os.path.join(save_path, 'config.yaml')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    config['data']['mode'] = 'test'
    config['model']['mode'] = 'test'

    device = set_device(config)

    # Load model
    best_model_path = os.path.join(save_path, 'best_model.tar')
    model = set_inference_model(config, best_model_path)

    _, _, test_dataloader, _ = set_dataloader(config)

    inference_record_df = inference(model, test_dataloader, device)

    inference_record_csv = os.path.join(save_path, 'test.csv')
    logger.info(f'Save to {inference_record_csv}')
    inference_record_df.to_csv(inference_record_csv, index=False)


def plot_test_result():
    test_csv = os.path.join(save_path, 'test.csv')
    logger.info(f'Load {test_csv}')
    test_df = pd.read_csv(test_csv)

    gt_pta = test_df['gt_pta'].to_list()
    pred_pta = test_df['pred_pta'].to_list()
    gt_dfa = test_df['gt_dfa'].to_list()
    pred_dfa = test_df['pred_dfa'].to_list()
    gt_hka = test_df['gt_hka'].to_list()
    pred_hka = test_df['pred_hka'].to_list()

    for gt, pred, label, lim_max, lim_min in zip(
            [gt_pta, gt_dfa, gt_hka],
            [pred_pta, pred_dfa, pred_hka],
            ['pta', 'dfa', 'hka'],
            [11, 2, 8],
            [-4, -13, -8]):
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.scatter(gt, pred, color='r', alpha=1.0, s=5)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], c='b', linewidth=2, linestyle='dashed')
        ax.set_xlabel(f'Ground Truth {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        r_val, p_val = pearsonr(pred, gt)
        rmse_val = np.sqrt(mean_squared_error(gt, pred))
        annotate_str = f'r = {r_val:.3f} (p-value: {p_val:.3f})\n' \
                       f'RMSE = {rmse_val:.3f}'

        ax.annotate(
            annotate_str,
            xy=(0.01, 0.99),
            horizontalalignment='left',
            verticalalignment='top',
            xycoords='axes fraction'
        )

        out_png = os.path.join(save_path, f'test.regression.{label}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

def plot_test_result_ba():
    test_csv = os.path.join(save_path, 'test.csv')
    logger.info(f'Load {test_csv}')
    test_df = pd.read_csv(test_csv)

    gt_pta = test_df['gt_pta'].to_list()
    pred_pta = test_df['pred_pta'].to_list()
    gt_dfa = test_df['gt_dfa'].to_list()
    pred_dfa = test_df['pred_dfa'].to_list()
    gt_hka = test_df['gt_hka'].to_list()
    pred_hka = test_df['pred_hka'].to_list()

    for gt, pred, label, lim_max, lim_min in zip(
            [gt_pta, gt_dfa, gt_hka],
            [pred_pta, pred_dfa, pred_hka],
            ['pta', 'dfa', 'hka'],
            [11, 2, 8],
            [-4, -13, -8]):
        fig, ax = plt.subplots(figsize=(10, 9))
        stat.graphics.mean_diff_plot(np.array(gt), np.array(pred), ax=ax)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)


        out_png = os.path.join(save_path, f'test.Bland-Altman.{label}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()


def run_inference_validate():
    yaml_config = os.path.join(save_path, 'config.yaml')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    config['data']['mode'] = 'train'
    config['model']['mode'] = 'train'

    device = set_device(config)

    # Load model
    best_model_path = os.path.join(save_path, 'best_model.tar')
    model = set_inference_model(config, best_model_path)

    _, valid_dataloader, _, _ = set_dataloader(config)

    inference_record_df = inference(model, valid_dataloader, device)

    inference_record_csv = os.path.join(save_path, 'validate.csv')
    logger.info(f'Save to {inference_record_csv}')
    inference_record_df.to_csv(inference_record_csv, index=False)


def plot_test_result_validate():
    test_csv = os.path.join(save_path, 'validate.csv')
    logger.info(f'Load {test_csv}')
    test_df = pd.read_csv(test_csv)

    gt_pta = test_df['gt_pta'].to_list()
    pred_pta = test_df['pred_pta'].to_list()
    gt_dfa = test_df['gt_dfa'].to_list()
    pred_dfa = test_df['pred_dfa'].to_list()
    gt_hka = test_df['gt_hka'].to_list()
    pred_hka = test_df['pred_hka'].to_list()

    for gt, pred, label, lim_max, lim_min in zip(
            [gt_pta, gt_dfa, gt_hka],
            [pred_pta, pred_dfa, pred_hka],
            ['pta', 'dfa', 'hka'],
            [11, 2, 8],
            [-4, -13, -8]):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(gt, pred, color='r', alpha=1.0, s=5)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], c='b', linewidth=2, linestyle='dashed')
        ax.set_xlabel(f'Ground Truth {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        r_val, p_val = pearsonr(pred, gt)
        rmse_val = np.sqrt(mean_squared_error(gt, pred))
        annotate_str = f'r = {r_val:.3f} (p-value: {p_val:.3f})\n' \
                       f'RMSE = {rmse_val:.3f}'

        ax.annotate(
            annotate_str,
            xy=(0.01, 0.99),
            horizontalalignment='left',
            verticalalignment='top',
            xycoords='axes fraction'
        )

        out_png = os.path.join(save_path, f'validate.regression.{label}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

def plot_test_result__validate_ba():
    test_csv = os.path.join(save_path, 'validate.csv')
    logger.info(f'Load {test_csv}')
    test_df = pd.read_csv(test_csv)

    gt_pta = test_df['gt_pta'].to_list()
    pred_pta = test_df['pred_pta'].to_list()
    gt_dfa = test_df['gt_dfa'].to_list()
    pred_dfa = test_df['pred_dfa'].to_list()
    gt_hka = test_df['gt_hka'].to_list()
    pred_hka = test_df['pred_hka'].to_list()

    for gt, pred, label, lim_max, lim_min in zip(
            [gt_pta, gt_dfa, gt_hka],
            [pred_pta, pred_dfa, pred_hka],
            ['pta', 'dfa', 'hka'],
            [11, 2, 8],
            [-4, -13, -8]):
        fig, ax = plt.subplots(figsize=(10, 9))
        stat.graphics.mean_diff_plot(np.array(gt), np.array(pred), ax=ax)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)


        out_png = os.path.join(save_path, f'validate.Bland-Altman.{label}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

def run_inference_training():
    yaml_config = os.path.join(save_path, 'config.yaml')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    config['data']['mode'] = 'train'
    config['model']['mode'] = 'train'

    device = set_device(config)

    # Load model
    best_model_path = os.path.join(save_path, 'best_model.tar')
    model = set_inference_model(config, best_model_path)

    training_dataloader, _, _, _ = set_dataloader_train(config)

    inference_record_df = inference(model, training_dataloader, device)

    inference_record_csv = os.path.join(save_path, 'train.csv')
    logger.info(f'Save to {inference_record_csv}')
    inference_record_df.to_csv(inference_record_csv, index=False)


def plot_test_result_training():
    test_csv = os.path.join(save_path, 'train.csv')
    logger.info(f'Load {test_csv}')
    test_df = pd.read_csv(test_csv)

    gt_pta = test_df['gt_pta'].to_list()
    pred_pta = test_df['pred_pta'].to_list()
    gt_dfa = test_df['gt_dfa'].to_list()
    pred_dfa = test_df['pred_dfa'].to_list()
    gt_hka = test_df['gt_hka'].to_list()
    pred_hka = test_df['pred_hka'].to_list()

    for gt, pred, label, lim_max, lim_min in zip(
            [gt_pta, gt_dfa, gt_hka],
            [pred_pta, pred_dfa, pred_hka],
            ['pta', 'dfa', 'hka'],
            [11, 2, 8],
            [-4, -13, -8]):
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.scatter(gt, pred, color='r', alpha=1.0, s=5)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], c='b', linewidth=2, linestyle='dashed')
        ax.set_xlabel(f'Ground Truth {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        r_val, p_val = pearsonr(pred, gt)
        rmse_val = np.sqrt(mean_squared_error(gt, pred))
        annotate_str = f'r = {r_val:.3f} (p-value: {p_val:.3f})\n' \
                       f'RMSE = {rmse_val:.3f}'

        ax.annotate(
            annotate_str,
            xy=(0.01, 0.99),
            horizontalalignment='left',
            verticalalignment='top',
            xycoords='axes fraction'
        )

        out_png = os.path.join(save_path, f'train.regression.{label}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

def plot_test_result_train_ba():
    test_csv = os.path.join(save_path, 'train.csv')
    logger.info(f'Load {test_csv}')
    test_df = pd.read_csv(test_csv)

    gt_pta = test_df['gt_pta'].to_list()
    pred_pta = test_df['pred_pta'].to_list()
    gt_dfa = test_df['gt_dfa'].to_list()
    pred_dfa = test_df['pred_dfa'].to_list()
    gt_hka = test_df['gt_hka'].to_list()
    pred_hka = test_df['pred_hka'].to_list()

    for gt, pred, label, lim_max, lim_min in zip(
            [gt_pta, gt_dfa, gt_hka],
            [pred_pta, pred_dfa, pred_hka],
            ['pta', 'dfa', 'hka'],
            [11, 2, 8],
            [-4, -13, -8]):
        fig, ax = plt.subplots(figsize=(10, 9))
        stat.graphics.mean_diff_plot(np.array(gt), np.array(pred), ax=ax)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)


        out_png = os.path.join(save_path, f'train.Bland-Altman.{label}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
if __name__ == '__main__':
    # save_path = '/nfs/masi/dongc1/Desktop/Projects/COVID19_severity_score/Experiments/experiment_1/run_28'
    #save_path = '/home/local/VANDERBILT/dongc1/Desktop/Projects/Knee/Experiment_2_knee/experiment_2/run_5'
    save_path = '/home/local/VANDERBILT/dongc1/Desktop/Projects/Knee/Experiment_4_knee/experiment_4/run_0'
    # get_train_valid_loss_csv()
    # plot_training_curve()
    run_inference()
    plot_test_result()
    run_inference_validate()
    plot_test_result_validate()
    run_inference_training()
    plot_test_result_training()
    plot_test_result_ba()
    plot_test_result__validate_ba()
    plot_test_result_train_ba()