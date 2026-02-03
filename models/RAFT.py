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
            nn.Linear(self.seq_len, 512),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(512, self.pred_len)
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
        self.pattern_dict = {} # 패턴 저장용 딕셔너리 추가
        
        # retrieve_all이 이제 (pred, patterns)를 반환
        print('Doing Train Retrieval')
        tr_pred, tr_pat = self.rt.retrieve_all(train_data, train=True, device=self.device)
        print('Doing Valid Retrieval')
        va_pred, va_pat = self.rt.retrieve_all(valid_data, train=False, device=self.device)
        print('Doing Test Retrieval')
        te_pred, te_pat = self.rt.retrieve_all(test_data, train=False, device=self.device)

        # 메모리 관리를 위해 rt 삭제 (선택사항, 패턴 데이터가 크면 주의)
        del self.rt
        torch.cuda.empty_cache()
            
        self.retrieval_dict['train'] = tr_pred.detach()
        self.retrieval_dict['valid'] = va_pred.detach()
        self.retrieval_dict['test'] = te_pred.detach()
        
        # 패턴 데이터 저장
        self.pattern_dict['train'] = tr_pat.detach()
        self.pattern_dict['valid'] = va_pat.detach()
        self.pattern_dict['test'] = te_pat.detach()

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
        # 1. 기존 RAFT (F network) 로직 수행
        # ... (encoder 함수 내용과 동일하게 F_pred 계산) ...
        # (편의상 encoder 내용을 여기에 통합하거나 encoder가 F_pred를 리턴하게 수정)
        
        # (기존 encoder 로직 복사 시작)
        x_offset = x_enc[:, -1:, :].detach()
        x_norm = x_enc - x_offset
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        
        pred_from_retrieval = self.retrieval_dict[mode][:, index].to(self.device)
        # ... (retrieval_pred_list 계산 로직) ...
        retrieval_pred_list = []
        bsz = x_enc.shape[0]
        for i, pr in enumerate(pred_from_retrieval):
             g = self.period_num[i]
             pr = pr.reshape(bsz, self.pred_len // g, g, self.channels)
             pr = pr[:, :, 0, :]
             pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
             pr = pr.reshape(bsz, self.pred_len, self.channels)
             retrieval_pred_list.append(pr)
        
        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1).sum(dim=1)
        pred_combined = torch.cat([x_pred_from_x, retrieval_pred_list], dim=1)
        
        # f_pred: [B, Pred_Len, C]
        # patterns: [B, TopM, Pred_Len, C] (retrieval에서 가져온 원본 패턴들)

        # [수정] 1. G에게 "F가 검색했던 패턴들의 통계량"을 제공
        # 패턴들 사이의 평균과 표준편차 계산 (TopM 차원에 대해)
        p_mean = patterns.mean(dim=1) # [B, Pred_Len, C]
        p_std = patterns.std(dim=1)   # [B, Pred_Len, C]
        
        # x_norm과 패턴 통계량을 결합하여 G의 입력 생성
        # 마지막 채널(C) 기준으로 처리하거나 Flatten 처리
        x_norm = x_enc - x_enc[:, -1:, :]
        
        # 단순화를 위해 각 배치/채널별로 G 수행 (채널 독립적 가정 시)
        # g_input: [B, C, seq_len + 2*pred_len]
        g_input = torch.cat([
            x_norm.permute(0, 2, 1), 
            p_mean.permute(0, 2, 1), 
            p_std.permute(0, 2, 1)
        ], dim=-1)
        
        g_pred = self.residual_net(g_input).permute(0, 2, 1) # [B, Pred_Len, C]

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
            def forward(self, x_enc, index, mode='train'):
        f_pred, g_pred, patterns = self.forecast(x_enc, index, mode)
        
        # [수정] 2. Multiplicative (곱하기) 방식 적용
        # y = f * (1 + g) -> g가 0.1이면 진폭 10% 증가, -0.1이면 10% 감소
        if mode != 'train':
            return f_pred * (1 + g_pred)
        
        return f_pred, g_pred, patterns
            f_pred, g_pred, patterns = self.forecast(x_enc, index, mode)
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
