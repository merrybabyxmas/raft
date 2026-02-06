
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from layers.Retrieval import RetrievalTool
import numpy as np

class QuantumSelector(nn.Module):
    """
    입력 쿼리를 받아 '어떤 패턴을 선택할지' 결정하는 양자 회로
    """
    def __init__(self, n_wires, n_patterns):
        super().__init__()
        self.n_wires = n_wires
        self.n_patterns = n_patterns

        self.encoder_gates = tq.GeneralEncoder(
            [ {'input_idx': [i], 'func': 'ry', 'wires': [i]} for i in range(n_wires) ]
        )

        self.ansatz = tq.RandomLayer(n_ops=n_wires * 3, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.prob_head = nn.Linear(n_wires, n_patterns)

    def forward(self, x_query, device):
        # x_query: [Batch, n_wires]
        bsz = x_query.shape[0]
        q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz)
        q_device.to(device)

        self.encoder_gates(q_device, x_query)
        self.ansatz(q_device)

        exp_val = self.measure(q_device) # [Batch, n_wires]
        logits = self.prob_head(exp_val)
        return logits, exp_val

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # --- 1. Quantum Settings ---
        self.topm = configs.topm
        self.n_wires = 6
        self.query_proj = nn.Linear(self.seq_len, self.n_wires)
        self.quantum_selector = QuantumSelector(n_wires=self.n_wires, n_patterns=self.topm)

        # --- 2. Base Predictor ---
        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        # --- 3. Retrieval Settings ---
        self.n_period = configs.n_period
        self.rt = RetrievalTool(
            seq_len=self.seq_len, pred_len=self.pred_len,
            channels=self.channels, n_period=self.n_period, topm=self.topm,
            temperature=1e-5
        )

        all_periods = self.rt.period_num[-1 * self.n_period:]
        self.period_num = [g for g in all_periods if self.pred_len % g == 0]
        if not self.period_num: self.period_num = [1]
        self.period_indices = [i for i, g in enumerate(all_periods) if self.pred_len % g == 0]

        self.retrieval_pred = nn.ModuleList([
            nn.Linear(self.pred_len // g, self.pred_len) for g in self.period_num
        ])

        # --- 4. Final Combination Layer ---
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

    def forecast(self, x_enc, index, mode, return_diag=False):
        # 1. Base Prediction & Norm
        x_offset = x_enc[:, -1:, :].detach()
        x_norm = x_enc - x_offset
        base_pred = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. Quantum Selection
        q_in = self.query_proj(x_norm.permute(0, 2, 1)) # [B, C, n_wires]
        B, C, W = q_in.shape
        q_in_flat = q_in.reshape(B * C, W)

        logits, exp_val = self.quantum_selector(q_in_flat, self.device) # [B*C, topm], [B*C, n_wires]

        # Gumbel Softmax Hard → one-hot selection
        soft_weights = F.softmax(logits, dim=-1)
        weights = F.gumbel_softmax(logits, tau=0.5, hard=True)
        weights = weights.reshape(B, C, self.topm).permute(0, 2, 1) # [B, topm, C]

        # 3. Retrieval Patterns: [n_period, B, topm, pred_len, C]
        pred_from_retrieval_all = self.retrieval_dict[mode][:, index.cpu()].to(self.device)
        pred_from_retrieval = pred_from_retrieval_all[self.period_indices]

        ret_list = []
        for i, pr in enumerate(pred_from_retrieval):
             # pr: [B, topm, pred_len, C]
             g = self.period_num[i]
             pr = pr.reshape(B * self.topm, self.pred_len, C)
             pr = pr.reshape(B * self.topm, self.pred_len // g, g, C)[:, :, 0, :]
             pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
             ret_list.append(pr.reshape(B, self.topm, self.pred_len, C))

        raw_candidates = torch.stack(ret_list, dim=0).sum(dim=0) # [B, topm, Pred, C]

        # 4. Collapse & Selection
        weights_expanded = weights.unsqueeze(2) # [B, topm, 1, C]
        selected_pattern = (raw_candidates * weights_expanded).sum(dim=1) # [B, Pred, C]

        # 5. Combine via linear_pred
        combined = torch.cat([base_pred, selected_pattern], dim=1) # [B, 2*Pred, C]
        pred_norm = self.linear_pred(combined.permute(0, 2, 1)).permute(0, 2, 1) # [B, Pred, C]

        final_pred = pred_norm + x_offset

        if return_diag:
            # Which pattern index was selected per (B, C)
            selected_idx = weights.reshape(B, self.topm, C).argmax(dim=1) # Actually for hard one-hot
            # For hard gumbel, the one-hot tells us which was picked
            hard_selection = weights.reshape(B, self.topm, C) # [B, topm, C]
            pattern_counts = hard_selection.sum(dim=(0, 2)) # [topm] - how many times each pattern selected

            diag = {
                'quantum/logits_mean': logits.mean().item(),
                'quantum/logits_std': logits.std().item(),
                'quantum/exp_val_mean': exp_val.mean().item(),
                'quantum/exp_val_std': exp_val.std().item(),
                'quantum/soft_entropy': -(soft_weights * (soft_weights + 1e-8).log()).sum(dim=-1).mean().item(),
                'quantum/pattern_counts': pattern_counts.detach().cpu().numpy(),
                'pred/base_pred_norm': base_pred.norm().item() / B,
                'pred/selected_pattern_norm': selected_pattern.norm().item() / B,
                'pred/combined_pred_norm': pred_norm.norm().item() / B,
                'pred/base_ratio': base_pred.norm().item() / (base_pred.norm().item() + selected_pattern.norm().item() + 1e-8),
            }
            return final_pred, diag

        return final_pred

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)
        self.retrieval_dict, self.retrieval_indices_dict = {}, {}
        print('Doing Retrieval...')
        tr_pred, tr_idx = self.rt.retrieve_all_with_indices(train_data, train=True, device=self.device)
        va_pred, va_idx = self.rt.retrieve_all_with_indices(valid_data, train=False, device=self.device)
        te_pred, te_idx = self.rt.retrieve_all_with_indices(test_data, train=False, device=self.device)

        self.y_data_source, self.x_data_source = self.rt.y_data_all, self.rt.train_data_all
        del self.rt
        torch.cuda.empty_cache()

        self.retrieval_dict.update({'train': tr_pred.detach(), 'valid': va_pred.detach(), 'test': te_pred.detach()})
        self.retrieval_indices_dict.update({'train': tr_idx.detach().cpu(), 'valid': va_idx.detach().cpu(), 'test': te_idx.detach().cpu()})

    def forward(self, x_enc, index, mode='train', return_patterns=False, return_diag=False, *args, **kwargs):
        if self.task_name == 'long_term_forecast':
            if return_diag:
                final_pred, diag = self.forecast(x_enc, index, mode, return_diag=True)
            else:
                final_pred = self.forecast(x_enc, index, mode, return_diag=False)

            if mode == 'train':
                if return_diag:
                    return final_pred, torch.zeros_like(final_pred), None, diag
                return final_pred, torch.zeros_like(final_pred), None
            if return_patterns:
                batch_indices = self.retrieval_indices_dict[mode][index.cpu()]
                return final_pred, self.x_data_source[batch_indices], self.y_data_source[batch_indices]
            return final_pred
        return None

    def imputation(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        enc_out = self.encoder(x_enc, index, mode)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output
