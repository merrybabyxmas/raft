import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.use_trust_gate = getattr(configs, 'use_trust_gate', False)
        self.use_time_emb = getattr(configs, 'use_time_emb', False)

        # --- Residual Network (G) 입력 차원 ---
        # 입력: x_norm(Seq) + f_pred(Pred)
        dim = self.seq_len + self.pred_len

        self.residual_net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.pred_len)
        )

        # --- [추가] Trust Gate 정의 ---
        if self.use_trust_gate:
            self.trust_gate = nn.Sequential(
                nn.Linear(self.pred_len, 128),
                nn.ReLU(),
                nn.Linear(128, self.pred_len),
                nn.Sigmoid() # 0 ~ 1 사이의 신뢰도 가중치
            )
        
        # --- [추가] Time Embedding 관련 정의 ---
        if self.use_time_emb:
            mark_dim = configs.enc_in_mark if hasattr(configs, 'enc_in_mark') else 4
            self.time_embedding = nn.Linear(mark_dim, configs.d_model)
            self.time_projection = nn.Linear(configs.d_model, self.channels)

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        
        # Retrieval 설정
        self.n_period = configs.n_period
        self.topm = configs.topm
        self.rt = RetrievalTool(
            seq_len=self.seq_len, pred_len=self.pred_len,
            channels=self.channels, n_period=self.n_period, topm=self.topm
        )
        
        # Period 관련 인덱스 계산 로직 (기존 유지)
        all_periods = self.rt.period_num[-1 * self.n_period:]
        self.period_num = [g for g in all_periods if self.pred_len % g == 0]
        if len(self.period_num) == 0: self.period_num = [1]
        self.period_indices = [i for i, g in enumerate(all_periods) if self.pred_len % g == 0]
        if len(self.period_indices) == 0: self.period_indices = [len(all_periods) - 1]

        self.retrieval_pred = nn.ModuleList([
            nn.Linear(self.pred_len // g, self.pred_len) for g in self.period_num
        ])
        
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

        # Store source patterns for residual net stats and visualization
        self.y_data_source = self.rt.y_data_all
        self.x_data_source = self.rt.train_data_all  # 입력 패턴 저장 (시각화용)

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

    def forecast(self, x_enc, index, mode, x_mark=None):
        # 1. Main Network (F) 연산을 먼저 수행하여 베이스라인 생성
        x_offset = x_enc[:, -1:, :].detach()
        x_norm = x_enc - x_offset # [B, Seq, C]

        # Time Embedding: x_mark [B, Seq, mark_dim] -> [B, Seq, C]
        if self.use_time_emb and x_mark is not None:
            t_emb = self.time_embedding(x_mark)       # [B, Seq, d_model]
            t_emb = self.time_projection(t_emb)        # [B, Seq, C]
            x_norm = x_norm + t_emb

        # F의 예측 로직 (기존 RAFT 로직)
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        pred_from_retrieval_all = self.retrieval_dict[mode][:, index.cpu()].to(self.device)

        # 필터링된 period에 해당하는 retrieval 결과만 사용
        pred_from_retrieval = pred_from_retrieval_all[self.period_indices]

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
        
        # F의 정규화된 예측값 (G의 입력으로 사용됨)
        f_pred_norm = self.linear_pred(pred_combined.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)
        f_pred = f_pred_norm + x_offset

        # 2. 검색 패턴 획득
        batch_indices = self.retrieval_indices_dict[mode][index.cpu()]
        patterns = self.y_data_source[batch_indices].to(self.device) # [B, M, Pred, C]

        # Trust 계산
        if self.use_trust_gate:
            p_std = patterns.std(dim=1)
            trust_score = self.trust_gate(p_std.permute(0, 2, 1)).permute(0, 2, 1) # [B, Pred, C]
        else:
            trust_score = None

        # 3. Residual Network (G) 계산 - Additive Correction
        # g_input: x_norm(Seq) + f_pred(Pred)
        g_input = torch.cat([
            x_norm.permute(0, 2, 1),
            f_pred_norm.detach().permute(0, 2, 1),
        ], dim=-1) # [B, C, Seq + Pred]

        # G 출력: (Y - F)를 예측
        g_pred = self.residual_net(g_input)
        g_pred = g_pred.permute(0, 2, 1)  # [B, Pred, C]

        return f_pred, g_pred, patterns, trust_score

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

    def forward(self, x_enc, index, mode='train', x_mark=None, return_patterns=False):
        if self.task_name == 'long_term_forecast':
            f_pred, g_pred, patterns, trust_score = self.forecast(x_enc, index, mode, x_mark)

            # Apply Additive Correction
            if trust_score is not None:
                final_pred = f_pred + trust_score * g_pred
            else:
                final_pred = f_pred + g_pred
            
            if mode != 'train':
                if return_patterns:
                    # 시각화를 위해 검색된 입력/출력 패턴 반환
                
                    batch_indices = self.retrieval_indices_dict[mode][index.cpu()]
                    retrieved_input = self.x_data_source[batch_indices]   # [B, M, Seq, C]
                    retrieved_output = self.y_data_source[batch_indices]  # [B, M, Pred, C]
                    return final_pred, retrieved_input, retrieved_output
                return final_pred

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
