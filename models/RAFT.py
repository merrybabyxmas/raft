import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        
        # --- Residual Network (G) ---
        # 입력: x_norm(Seq) + f_pred_norm(Pred)
        self.dim = self.seq_len + self.pred_len
        self.residual_net = nn.Sequential(
            nn.Linear(self.dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.pred_len)
        )

        # --- Scaling Gain Network ---
        self.gain_net = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1), # 채널별 Gain 값 1개 추출
            nn.Sigmoid() 
        )

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        
        # Retrieval 설정
        self.n_period = configs.n_period
        self.topm = configs.topm
        self.rt = RetrievalTool(
            seq_len=self.seq_len, pred_len=self.pred_len,
            channels=self.channels, n_period=self.n_period, topm=self.topm,
            temperature=0.01 # Blurring 방지를 위해 낮은 값 설정
        )
        
        # Period 관련 인덱스 계산 로직
        all_periods = self.rt.period_num[-1 * self.n_period:]
        self.period_num = [g for g in all_periods if self.pred_len % g == 0]
        if len(self.period_num) == 0: self.period_num = [1]
        self.period_indices = [i for i, g in enumerate(all_periods) if self.pred_len % g == 0]
        
        self.retrieval_pred = nn.ModuleList([
            nn.Linear(self.pred_len // g, self.pred_len) for g in self.period_num
        ])

    def forecast(self, x_enc, index, mode, x_mark=None):
        # 1. 베이스라인 및 정규화
        x_offset = x_enc[:, -1:, :].detach()
        x_norm = x_enc - x_offset # [B, Seq, C]

        # F의 기본 예측 (입력 데이터 기반)
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 검색된 패턴들 가져오기
        pred_from_retrieval_all = self.retrieval_dict[mode][:, index.cpu()].to(self.device)
        pred_from_retrieval = pred_from_retrieval_all[self.period_indices]

        bsz = x_enc.shape[0]
        retrieval_pred_list = []
        for i, pr in enumerate(pred_from_retrieval):
             g_period = self.period_num[i]
             pr = pr.reshape(bsz, self.pred_len // g_period, g_period, self.channels)
             pr = pr[:, :, 0, :]
             pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
             retrieval_pred_list.append(pr.reshape(bsz, self.pred_len, self.channels))

        # 검색된 패턴 결합
        retrieval_out = torch.stack(retrieval_pred_list, dim=1).sum(dim=1) # [B, Pred, C]
        
        # --- [핵심 추가] 진폭 매칭 (Amplitude Matching) ---
        # 1) 현재 입력(Query)의 진폭(표준편차) 계산
        query_std = torch.std(x_norm, dim=1, keepdim=True) + 1e-5 # [B, 1, C]
        # 2) 검색된 패턴(Retrieval)의 진폭 계산
        ret_std = torch.std(retrieval_out, dim=1, keepdim=True) + 1e-5 # [B, 1, C]
        # 3) 스케일 조정: 검색된 패턴을 입력의 진폭에 맞춰 확장
        retrieval_out_scaled = retrieval_out * (query_std / ret_std)
        # --------------------------------------------------

        # 확장된 검색 패턴을 사용하여 결합 예측
        pred_combined = torch.cat([x_pred_from_x, retrieval_out_scaled], dim=1)
        f_pred_norm = self.linear_pred(pred_combined.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)

        # 2. Residual 및 Gain 계산
        g_input = torch.cat([
            x_norm.permute(0, 2, 1),
            f_pred_norm.detach().permute(0, 2, 1),
        ], dim=-1) # [B, C, Seq + Pred]

        # Gain 계산: 0.5~2.0 범위
        gain = (self.gain_net(g_input) * 1.5 + 0.5).permute(0, 2, 1) # [B, 1, C]

        # Gain 적용 후 offset 복원 (F 예측값)
        f_pred_scaled = f_pred_norm * gain + x_offset

        # Residual 예측 (G)
        g_pred = self.residual_net(g_input).permute(0, 2, 1) # [B, Pred, C]

        return f_pred_scaled, g_pred, gain

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

        self.y_data_source = self.rt.y_data_all
        self.x_data_source = self.rt.train_data_all 

        del self.rt
        torch.cuda.empty_cache()
            
        self.retrieval_dict['train'] = tr_pred.detach()
        self.retrieval_dict['valid'] = va_pred.detach()
        self.retrieval_dict['test'] = te_pred.detach()
        
        self.retrieval_indices_dict['train'] = tr_idx.detach().cpu()
        self.retrieval_indices_dict['valid'] = va_idx.detach().cpu()
        self.retrieval_indices_dict['test'] = te_idx.detach().cpu()

    def forward(self, x_enc, index, mode='train', x_mark=None, return_patterns=False):
        if self.task_name == 'long_term_forecast':
            f_pred, g_pred, gain = self.forecast(x_enc, index, mode, x_mark)

            # 최종 예측: F(보정된 베이스) + G(잔차)
            final_pred = f_pred + g_pred
            
            if mode != 'train':
                if return_patterns:
                    batch_indices = self.retrieval_indices_dict[mode][index.cpu()]
                    retrieved_input = self.x_data_source[batch_indices]
                    retrieved_output = self.y_data_source[batch_indices]
                    return final_pred, retrieved_input, retrieved_output
                return final_pred

            # 학습을 위해 분리된 결과 반환
            return f_pred, g_pred, None
        
        return None

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

