import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Retrieval import RetrievalTool

class moving_avg(nn.Module):
    """트렌드 성분 추출을 위한 이동 평균 블록 [cite: 59, 61]"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 패딩을 통해 시퀀스 길이 유지
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        back = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, back], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """시계열을 Seasonal과 Trend로 분해하는 블록 [cite: 28, 930]"""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # 1. 시계열 분해 도구 정의 [cite: 982]
        self.decomposition = series_decomp(configs.moving_avg)

        # 2. 각 성분을 위한 개별 선형 예측층 (RAFT 방식 결합) [cite: 968, 1014]
        self.linear_x_s = nn.Linear(self.seq_len, self.pred_len)
        self.linear_x_t = nn.Linear(self.seq_len, self.pred_len)
        
        # 3. Retrieval 설정
        self.n_period = configs.n_period
        self.topm = configs.topm
        
        # 성분별 검색을 위해 두 개의 RetrievalTool 인스턴스 생성 [cite: 959, 1014]
        self.rt_s = RetrievalTool(seq_len=self.seq_len, pred_len=self.pred_len,
                                  channels=self.channels, n_period=self.n_period, topm=self.topm)
        self.rt_t = RetrievalTool(seq_len=self.seq_len, pred_len=self.pred_len,
                                  channels=self.channels, n_period=self.n_period, topm=self.topm)
        
        self.period_num = self.rt_s.period_num[-1 * self.n_period:]
        
        # 성분별 검색 결과 처리를 위한 독립적인 ModuleList
        self.ret_pred_s = nn.ModuleList([nn.Linear(self.pred_len // g, self.pred_len) for g in self.period_num])
        self.ret_pred_t = nn.ModuleList([nn.Linear(self.pred_len // g, self.pred_len) for g in self.period_num])
        
        # 최종 결합층
        self.final_s = nn.Linear(2 * self.pred_len, self.pred_len)
        self.final_t = nn.Linear(2 * self.pred_len, self.pred_len)

    def prepare_dataset(self, train_data, valid_data, test_data):
        """
        1단계: 오리지널 데이터를 분해하여 개별 검색 DB 구축
        데이터셋 객체 내부의 .data_x를 직접 교체하여 RetrievalTool이 
        분해된 성분을 '원본'으로 인식하게 만듭니다.
        """
        # 원본 데이터 백업 (나중에 복구하기 위함)
        original_train_x = train_data.data_x
        original_valid_x = valid_data.data_x
        original_test_x = test_data.data_x

        # 전체 시퀀스를 분해하는 헬퍼 함수
        def get_decomp_np(data_np):
            tensor = torch.from_numpy(data_np).float().to(self.device).unsqueeze(0)
            s, t = self.decomposition(tensor)
            return s.squeeze(0).cpu().numpy(), t.squeeze(0).cpu().numpy()

        # Step 1 & 2: 학습/검증/테스트 데이터를 Trend와 Seasonal로 분해
        s_train, t_train = get_decomp_np(original_train_x)
        s_valid, t_valid = get_decomp_np(original_valid_x)
        s_test, t_test = get_decomp_np(original_test_x)

        # --- Seasonal 경로 구축 ---
        print('Doing Seasonal Retrieval')
        # 데이터셋 객체의 데이터를 seasonal 성분으로 교체 (Step 4를 위한 준비)
        train_data.data_x = s_train
        valid_data.data_x = s_valid
        test_data.data_x = s_test
        
        # 이제 객체를 전달하므로 .pred_len에 접근 가능하며, 
        # RetrievalTool은 교체된 seasonal 데이터를 기반으로 인덱스를 만듭니다.
        self.rt_s.prepare_dataset(train_data)
        
        self.r_dict_s = {}
        self.r_dict_s['train'] = self.rt_s.retrieve_all(train_data, train=True, device=self.device).detach()
        self.r_dict_s['valid'] = self.rt_s.retrieve_all(valid_data, train=False, device=self.device).detach()
        self.r_dict_s['test'] = self.rt_s.retrieve_all(test_data, train=False, device=self.device).detach()

        # --- Trend 경로 구축 ---
        print('Doing Trend Retrieval')
        # 데이터셋 객체의 데이터를 trend 성분으로 교체 (Step 3를 위한 준비)
        train_data.data_x = t_train
        valid_data.data_x = t_valid
        test_data.data_x = t_test
        
        self.rt_t.prepare_dataset(train_data)
        
        self.r_dict_t = {}
        self.r_dict_t['train'] = self.rt_t.retrieve_all(train_data, train=True, device=self.device).detach()
        self.r_dict_t['valid'] = self.rt_t.retrieve_all(valid_data, train=False, device=self.device).detach()
        self.r_dict_t['test'] = self.rt_t.retrieve_all(test_data, train=False, device=self.device).detach()

        # 데이터셋 원본 상태로 복구 (학습 루프에서 원본 데이터가 필요할 수 있음)
        train_data.data_x = original_train_x
        valid_data.data_x = original_valid_x
        test_data.data_x = original_test_x

        del self.rt_s, self.rt_t
        torch.cuda.empty_cache()

    def _process_raft(self, x, index, mode, lin_x, r_dict, r_preds, fin_layer):
        """개별 성분에 대해 RAFT 연산 수행 [cite: 90, 166]"""
        bsz = x.shape[0]
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        x_pred = lin_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        # Retrieval indexing (Device mismatch 해결)
        ret_val = r_dict[mode][:, index.cpu()].to(self.device)
        
        ret_list = []
        for i, pr in enumerate(ret_val):
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, self.channels)[:, :, 0, :]
            pr = r_preds[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            ret_list.append(pr)

        ret_sum = torch.stack(ret_list, dim=1).sum(dim=1)
        combined = torch.cat([x_pred, ret_sum], dim=1)
        return fin_layer(combined.permute(0, 2, 1)).permute(0, 2, 1) + x_offset

    def encoder(self, x, index, mode):
        # 2단계: Query 데이터를 분해 
        seasonal_q, trend_q = self.decomposition(x)
        
        # 3단계 & 4단계: 성분별 Retrieval 및 예측 (predict_trend, predict_seasonal) [cite: 1013, 1020]
        predict_seasonal = self._process_raft(seasonal_q, index, mode, self.linear_x_s, 
                                             self.r_dict_s, self.ret_pred_s, self.final_s)
        
        predict_trend = self._process_raft(trend_q, index, mode, self.linear_x_t, 
                                          self.r_dict_t, self.ret_pred_t, self.final_t)
        
        # 5단계: Residual 합산 [cite: 931, 954]
        return predict_seasonal + predict_trend
    def forecast(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def forward(self, x_enc, index, mode='train', *args, **kwargs):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, mode)
            # 최종 출력 길이를 pred_len에 맞게 슬라이싱하여 반환합니다.
            return dec_out[:, -self.pred_len:, :]
        
        # 다른 태스크(imputation 등)가 필요한 경우 여기에 추가
        return None