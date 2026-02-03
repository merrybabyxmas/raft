import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
#         self.decompsition = series_decomp(configs.moving_avg)
#         self.individual = individual
        self.channels = configs.enc_in
        
        self.residual_net = nn.Sequential(
            nn.Linear(self.seq_len + 2 * self.pred_len, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.pred_len),
            nn.Tanh()
        )
        

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm
        
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
        )
        
        self.period_num = self.rt.period_num[-1 * self.n_period:]
        
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

#         if self.task_name == 'classification':
#             self.projection = nn.Linear(
#                 configs.enc_in * configs.seq_len, configs.num_class)


    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)
        self.retrieval_dict = {}
        self.retrieval_indices_dict = {}
        
        print('Doing Train Retrieval')
        tr_pred, tr_idx = self.rt.retrieve_all_with_indices(train_data, train=True, device=self.device)
        print('Doing Valid Retrieval')
        va_pred, va_idx = self.rt.retrieve_all_with_indices(valid_data, train=False, device=self.device)
        print('Doing Test Retrieval')
        te_pred, te_idx = self.rt.retrieve_all_with_indices(test_data, train=False, device=self.device)

        # Store source patterns (targets) for residual net stats
        self.y_data_source = self.rt.y_data_all

        del self.rt
        torch.cuda.empty_cache()
            
        self.retrieval_dict['train'] = tr_pred.detach()
        self.retrieval_dict['valid'] = va_pred.detach()
        self.retrieval_dict['test'] = te_pred.detach()
        
        self.retrieval_indices_dict['train'] = tr_idx.detach().cpu()
        self.retrieval_indices_dict['valid'] = va_idx.detach().cpu()
        self.retrieval_indices_dict['test'] = te_idx.detach().cpu()

    def encoder(self, x, index, mode, return_patterns=False):
        index = index.to(self.device)

        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len and channels == self.channels)

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1) # B, P, C

        pred_from_retrieval = self.retrieval_dict[mode][:, index.cpu()].to(self.device)

        retrieval_pred_list = []

        # Compress repeating dimensions
        for i, pr in enumerate(pred_from_retrieval):
            assert((bsz, self.pred_len, channels) == pr.shape)
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]

            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)

            retrieval_pred_list.append(pr)

        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)
        retrieval_pred_list = retrieval_pred_list.sum(dim=1)

        pred = torch.cat([x_pred_from_x, retrieval_pred_list], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)

        pred = pred + x_offset

        if return_patterns:
            # Get retrieved pattern indices
            retrieval_indices = self.retrieval_indices_dict[mode][index.cpu()]  # B, M
            # Extract actual patterns
            retrieved_input_patterns = self.train_data_all[retrieval_indices]  # B, M, S, C
            retrieved_output_patterns = self.y_data_all[retrieval_indices]  # B, M, P, C
            return pred, retrieved_input_patterns, retrieved_output_patterns

        return pred

    def forecast(self, x_enc, index, mode):
        # Retrieve indices for the current batch
        batch_indices = self.retrieval_indices_dict[mode][index.cpu()] # [B, M]

        # Retrieve patterns (Targets) for statistics
        # self.y_data_source is typically on CPU
        patterns = self.y_data_source[batch_indices].to(self.device) # [B, M, Pred, C]
        
        # Calculate Statistics for G Network
        p_mean = patterns.mean(dim=1) # [B, Pred, C]
        p_std = patterns.std(dim=1)   # [B, Pred, C]

        # G Network Input Construction
        # x_enc: [B, Seq, C]
        x_offset = x_enc[:, -1:, :].detach()
        x_norm = x_enc - x_offset

        # Concatenate: x, p_mean, p_std along the sequence/time dimension (after permuting to [B, C, T])
        g_input = torch.cat([
            x_norm.permute(0, 2, 1),
            p_mean.permute(0, 2, 1),
            p_std.permute(0, 2, 1)
        ], dim=-1) # [B, C, Seq + 2*Pred]

        g_pred = self.residual_net(g_input).permute(0, 2, 1) # [B, Pred, C]

        # F Network (Standard RAFT Logic)
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        
        pred_from_retrieval = self.retrieval_dict[mode][:, index.cpu()].to(self.device) # [G, B, Pred, C]

        bsz = x_enc.shape[0]
        retrieval_pred_list = []
        for i, pr in enumerate(pred_from_retrieval):
             g_period = self.period_num[i]

             pr = pr.reshape(bsz, self.pred_len // g_period, g_period, self.channels)
             pr = pr[:, :, 0, :]
             pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
             pr = pr.reshape(bsz, self.pred_len, self.channels)
             retrieval_pred_list.append(pr)

        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1).sum(dim=1)
        
        pred_combined = torch.cat([x_pred_from_x, retrieval_pred_list], dim=1)
        f_pred = self.linear_pred(pred_combined.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)
        f_pred = f_pred + x_offset

        return f_pred, g_pred, patterns

    def imputation(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        # Encoder
        enc_out = self.encoder(x_enc, index, mode)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode='train', return_patterns=False):
        if self.task_name == 'long_term_forecast':
            f_pred, g_pred, patterns = self.forecast(x_enc, index, mode)

            # Apply Multiplicative Correction (F * (1 + G))
            if mode != 'train':
                return f_pred * (1 + g_pred)

            return f_pred, g_pred, patterns
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, index, mode)
            return dec_out  # [B, N]
        return None
