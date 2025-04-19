from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, print_and_save
from utils.losses import mape_loss, mase_loss, smape_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import shutil

warnings.filterwarnings('ignore')


class Exp_Text_Earnings(Exp_Basic):
    def __init__(self, args):
        super(Exp_Text_Earnings, self).__init__(args)

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
            for i, (text_emb, volatility) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                text_emb = text_emb.float().to(self.device)
                volatility = volatility.float().to(self.device)

                pred_vol = self.model(text_emb)
                loss = criterion(pred_vol, volatility)
                train_loss.append(loss.item())

                if (i + 1) % 20 == 0:
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
            for i, (text_emb, volatility) in enumerate(vali_loader):
                text_emb = text_emb.float().to(self.device)
                volatility = volatility.float()

                pred_vol = self.model(text_emb)
                pred_vol = pred_vol.detach().cpu()

                loss = criterion(pred_vol, volatility)
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
            for i, (text_emb, volatility) in enumerate(test_loader):
                text_emb = text_emb.float().to(self.device)

                pred_vol = self.model(text_emb)
                pred_vol = pred_vol.detach().cpu().numpy()
  
                preds.append(pred_vol)
                trues.append(volatility)

        # preds = np.array(preds)
        # trues = np.array(trues)
        # preds = preds.reshape(preds.shape[0])
        # trues = trues.reshape(trues.shape[0])
        preds = np.array(preds).reshape(-1)
        trues = np.array(trues).reshape(-1)

        mse = mean_squared_error(preds, trues)
        # result save as df
        folder_path = './earnings_results/' + self.args.model + '/' + self.args.output_dir + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = {
            'id': test_loader.dataset.ids,
            'preds':  np.round(preds, 6),
            'trues': np.round(trues, 6),
        }
        forecasts_df = pandas.DataFrame(forecasts_df)

        forecasts_df['abs_e'] = np.round(abs(forecasts_df['preds'] - forecasts_df['trues']), 6)
        
        forecasts_df.sort_values(['abs_e'], inplace=True)
        
        forecasts_df_path = os.path.join(folder_path, f'{self.args.model_id}_{mse:.6f}.csv')
        forecasts_df.to_csv(forecasts_df_path, index=False)
      
        log_msg = f'\033[1mMSE for {self.args.model_id}: {mse:.6f}\033[0m'
        print_and_save(self.args.model_id, self.args.model, self.args.output_dir, log_msg)

        path = os.path.join(self.args.checkpoints, setting)
        try:
            if self.args.remove_model_path:
                shutil.rmtree(path)
        except Exception as error:
            print_and_save(self.args.model_id, self.args.model, self.args.output_dir, f"Error: {error}")

