from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_with_retrieval, save_metadata
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
from datetime import datetime
import warnings
import numpy as np
import copy
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        model.prepare_dataset(train_data, vali_data, test_data)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'RAFT' or self.args.model == 'D_RAFT':
                    outputs = self.model(batch_x, index, mode='valid', x_mark=batch_x_mark)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # Save experiment metadata
        save_metadata(self.args, path)

        # Create visualization folder for training (with timestamp subfolder)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        train_vis_path = os.path.join(path, 'train_visualizations', ts)
        if not os.path.exists(train_vis_path):
            os.makedirs(train_vis_path)

        time_now = time.time()

        train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_valid_loss = float('inf')
        best_model = None
            
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == 'RAFT':
                    f_pred, g_pred, patterns = self.model(batch_x, index, mode='train', x_mark=batch_x_mark)

                    # Train visualization (first batch of each epoch)
                    if i == 0:
                        with torch.no_grad():
                            batch_idx = self.model.retrieval_indices_dict['train'][index.cpu()]
                            ret_in = self.model.x_data_source[batch_idx]   # [B, M, Seq, C]
                            ret_out = self.model.y_data_source[batch_idx]  # [B, M, Pred, C]
                        input_np = batch_x[0].detach().cpu().numpy()
                        true_np = batch_y[0, -self.args.pred_len:, :].detach().cpu().numpy()
                        pred_np = (f_pred[0] + g_pred[0]).detach().cpu().numpy()
                        ret_input_np = ret_in[0].cpu().numpy()
                        ret_output_np = ret_out[0].cpu().numpy()

                        vis_name = os.path.join(train_vis_path, f'train_epoch_{epoch+1}_batch_{i}.png')
                        visual_with_retrieval(input_np, true_np, pred_np,
                                              ret_input_np, ret_output_np, name=vis_name)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    f_pred = f_pred[:, -self.args.pred_len:, f_dim:]
                    g_pred = g_pred[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # Loss F: MSE (Base Net이 Y를 직접 예측)
                    loss_f = criterion(f_pred, batch_y)

                    # Loss G: Additive Correction
                    # G는 F가 놓친 잔차(Y - F)를 학습
                    target_g = batch_y - f_pred.detach()

                    loss_g = criterion(g_pred, target_g)

                    # 디버깅: G의 출력 범위와 target 범위 확인
                    if (i + 1) % 100 == 0:
                        print(f"\t[DEBUG] g_pred range: [{g_pred.min().item():.3f}, {g_pred.max().item():.3f}], "
                              f"target_g range: [{target_g.min().item():.3f}, {target_g.max().item():.3f}]")
                        print(f"\t[DEBUG] loss_f: {loss_f.item():.4f}, loss_g: {loss_g.item():.4f}")

                    loss = loss_f + loss_g

                elif self.args.model == 'D_RAFT':
                    # Get outputs with retrieved patterns for visualization
                    if i == 0:  # Visualize first batch of each epoch
                        outputs, retrieved_input, retrieved_output = self.model(
                            batch_x, index, mode='train', return_patterns=True
                        )
                        # Visualize first sample in batch
                        input_np = batch_x[0].detach().cpu().numpy()
                        true_np = batch_y[0, -self.args.pred_len:, :].detach().cpu().numpy()
                        pred_np = outputs[0].detach().cpu().numpy()
                        ret_input_np = retrieved_input[0].detach().cpu().numpy()
                        ret_output_np = retrieved_output[0].detach().cpu().numpy()

                        vis_name = os.path.join(train_vis_path, f'train_epoch_{epoch+1}_batch_{i}.png')
                        visual_with_retrieval(input_np, true_np, pred_np,
                                            ret_input_np, ret_output_np, name=vis_name)
                    else:
                        outputs = self.model(batch_x, index, mode='train')

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # We do not use early stopping
            
            if vali_loss < best_valid_loss:
                best_model = copy.deepcopy(self.model)
                best_valid_loss = vali_loss
                
        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(best_model.state_dict(), best_model_path)
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model
        return best_model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_path = './test_results/' + setting + '/' + ts + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Create retrieval visualization folder
        retrieval_vis_path = os.path.join(folder_path, 'retrieval_visualizations')
        if not os.path.exists(retrieval_vis_path):
            os.makedirs(retrieval_vis_path)

        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'RAFT' or self.args.model == 'D_RAFT':
                    # Get outputs with retrieved patterns for visualization
                    if i % 20 == 0:
                        outputs, retrieved_input, retrieved_output = self.model(
                            batch_x, index, mode='test', x_mark=batch_x_mark, return_patterns=True
                        )
                    else:
                        outputs = self.model(batch_x, index, mode='test', x_mark=batch_x_mark)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs_np.shape
                    outputs_np = test_data.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs_np = outputs_np[:, :, f_dim:]
                batch_y_np = batch_y_np[:, :, f_dim:]

                pred = outputs_np
                true = batch_y_np

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pd_vis = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd_vis, os.path.join(folder_path, f'{i}.png'))

                    # Save retrieval visualization for RAFT models
                    if self.args.model == 'RAFT' or self.args.model == 'D_RAFT':
                        ret_input_np = retrieved_input[0].detach().cpu().numpy()
                        ret_output_np = retrieved_output[0].detach().cpu().numpy()
                        vis_name = os.path.join(retrieval_vis_path, f'eval_sample_{i}.png')
                        visual_with_retrieval(input_np[0], true[0], pred[0],
                                            ret_input_np, ret_output_np, name=vis_name)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
