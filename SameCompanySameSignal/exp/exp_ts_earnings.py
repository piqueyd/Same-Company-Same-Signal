
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import shutil

import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, print_and_save, TSMixerWrapper
from utils.losses import mape_loss, mase_loss, smape_loss

warnings.filterwarnings('ignore')


class Exp_TimeSeries_Earnings(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimeSeries_Earnings, self).__init__(args)

    def _build_model(self):
        print('_build_model self.args.model: ', self.args.model)
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (ts_data, volaitlity, ts_data_mark, volaitlity_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                ts_data = ts_data.float().to(self.device)
                volaitlity = volaitlity.float().to(self.device)
                
                # In TSMixer only the first parameter us utilized
                pred_vol = self.model(ts_data, None, None, None)

                # f_dim= -1 for Earnings since only one target column
                f_dim = -1 if self.args.features == 'MS' else 0
                pred_vol = pred_vol[:, -self.args.pred_len:, f_dim:]
                volaitlity = volaitlity[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss_value = criterion(pred_vol, volaitlity)
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 50 == 0:
                    log_msg = f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}"
                    print_and_save(self.args.model_id, self.args.model, self.args.output_dir, log_msg)
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    log_msg = f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s'
                    print_and_save(self.args.model_id, self.args.model, self.args.output_dir, log_msg)
                    
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            epoch_duration = time.time() - epoch_time
            log_msg = f"Epoch: {epoch + 1} cost time: {epoch_duration:.2f}s"
            print_and_save(self.args.model_id, self.args.model, self.args.output_dir, log_msg)
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            log_msg = (f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                      f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            print_and_save(self.args.model_id, self.args.model, self.args.output_dir, log_msg)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print_and_save(self.args.model_id, self.args.model, self.args.output_dir,
                               "Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (ts_data, volaitlity, ts_data_mark, volaitlity_mark) in enumerate(vali_loader):
                ts_data = ts_data.float().to(self.device)

                pred_vol = self.model(ts_data, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                pred_vol = pred_vol[:, -self.args.pred_len:, f_dim:]
                volaitlity = volaitlity[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = pred_vol.detach().cpu()
                true = volaitlity.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
   

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('\nLoading model to test')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (ts_data, volaitlity, ts_data_mark, volaitlity_mark) in enumerate(test_loader):
                ts_data = ts_data.float().to(self.device)
                volaitlity = volaitlity.float().to(self.device)
               
                pred_vol = self.model(ts_data, None, None, None)

                pred_vol = pred_vol[:, -self.args.pred_len:, :]
                volaitlity = volaitlity[:, -self.args.pred_len:, :].to(self.device)

                pred_vol = pred_vol.detach().cpu().numpy()
                volaitlity = volaitlity.detach().cpu().numpy()

                # M: multivariate predict multivariate,
                # S: univariate predict univariate,
                # MS: multivariate predict univariate;
                f_dim = -1 if self.args.features == 'MS' else 0
                pred_vol = pred_vol[:, :, f_dim:]
                volaitlity = volaitlity[:, :, f_dim:]
                preds.append(pred_vol)
                trues.append(volaitlity)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # log_msg = f'The shapes of preds {preds.shape} and trues: {trues.shape}'
        # print_and_save(self.args.model_id, self.args.model, self.args.output_dir, log_msg)


        # f_dim = -1 if self.args.features == 'MS' else 0
        f_dim = -1
        mse = mean_squared_error(preds[:, 0, f_dim], trues[:, 0, f_dim])

        # result save as df
        folder_path = './earnings_results/' + self.args.model + '/' + self.args.output_dir + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pd.DataFrame({
            'id': test_loader.dataset.ids,
            'preds': np.round(preds[:, 0, f_dim], 6),
            'trues': np.round(trues[:, 0, f_dim], 6)
        })

        forecasts_df['abs_e'] = np.round(abs(forecasts_df['preds'] - forecasts_df['trues']), 6)
        
        forecasts_df_path = os.path.join(folder_path, f'{self.args.model_id}_{mse:.6f}.csv')
        forecasts_df.to_csv(forecasts_df_path, index=False)

        log_msg = f'\033[1mMSE for {self.args.model_id}: {mse:.6f}\033[0m'
        print_and_save(self.args.model_id, self.args.model, self.args.output_dir, log_msg)

        # Remove the directory and all its contents
        path = os.path.join(self.args.checkpoints, setting)
        try:
            if self.args.remove_model_path:
                shutil.rmtree(path)
        except Exception as error:
            print_and_save(self.args.model_id, self.args.model, self.args.output_dir, f"Error: {error}")
