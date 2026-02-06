import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key
        
    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            
            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])
            
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        
        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C
        seq_len = data_all.shape[1]

        mg = []
        for g in self.period_num:
            # seq_len이 g로 나누어떨어지지 않으면 패딩 후 자르기
            n_chunks = seq_len // g
            if n_chunks == 0:
                # period가 seq_len보다 크면 전체를 하나의 평균으로
                cur = data_all.mean(dim=1, keepdim=True).repeat(1, seq_len, 1)
            else:
                usable_len = n_chunks * g
                cur = data_all[:, :usable_len, :].unfold(dimension=1, size=g, step=g).mean(dim=-1)
                cur = cur.repeat_interleave(repeats=g, dim=1)
                # 남은 부분은 마지막 값으로 채움
                if cur.shape[1] < seq_len:
                    padding = cur[:, -1:, :].repeat(1, seq_len - cur.shape[1], 1)
                    cur = torch.cat([cur, padding], dim=1)

            mg.append(cur)
#             data_all = data_all - cur

        mg = torch.stack(mg, dim=0) # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)
            
        return mg, offset
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        bx = key - torch.mean(key, dim=2, keepdim=True)
        
        iters = math.ceil(train_len / in_bsz)
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)
        
        return sim
        
    def retrieve(self, x, index, train=True, return_patterns=False):
        index = index.to(x.device)

        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)

        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T

        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)

            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)

            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T

        topm_index = torch.topk(sim, self.topm, dim=1).indices
        ranking_sim = torch.ones_like(sim) * float('-inf')

        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]

        sim = sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        topm_index = topm_index.reshape(self.n_period, bsz, self.topm) # G, B, M

        data_len, seq_len, channels = self.train_data_all.shape

        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu() # G, B, T

        y_data_all = self.y_data_all_mg.flatten(start_dim=2) # G, T, P * C

        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(self.n_period, bsz, -1, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)

        if return_patterns:
            # Extract top-M retrieved patterns (input sequences)
            # topm_index: G, B, M -> use first period group's indices
            first_period_indices = topm_index[0]  # B, M
            retrieved_input_patterns = []
            retrieved_output_patterns = []
            for b in range(bsz):
                indices = first_period_indices[b].cpu()  # M
                input_patterns = self.train_data_all[indices]  # M, S, C
                output_patterns = self.y_data_all[indices]  # M, P, C
                retrieved_input_patterns.append(input_patterns)
                retrieved_output_patterns.append(output_patterns)
            retrieved_input_patterns = torch.stack(retrieved_input_patterns, dim=0)  # B, M, S, C
            retrieved_output_patterns = torch.stack(retrieved_output_patterns, dim=0)  # B, M, P, C
            return pred_from_retrieval, retrieved_input_patterns, retrieved_output_patterns

        return pred_from_retrieval
    
    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert(self.train_data_all_mg != None)

        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )

        retrievals = []
        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                pred_from_retrieval = self.retrieve(batch_x.float().to(device), index, train=train)
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)

        retrievals = torch.cat(retrievals, dim=1)

        return retrievals

    def retrieve_with_indices(self, x, index, train=True):
        """Retrieve predictions and return top-M indices for visualization."""
        index = index.to(x.device)

        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)

        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T

        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)

            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)

            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T

        topm_index = torch.topk(sim, self.topm, dim=1).indices
        ranking_sim = torch.ones_like(sim) * float('-inf')

        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]

        sim = sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        topm_index_reshaped = topm_index.reshape(self.n_period, bsz, self.topm) # G, B, M

        data_len, seq_len, channels = self.train_data_all.shape

        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu() # G, B, T

        y_data_all = self.y_data_all_mg.flatten(start_dim=2) # G, T, P * C

        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(self.n_period, bsz, -1, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)

        # Return first period's top-M indices for visualization (B, M)
        first_period_indices = topm_index_reshaped[0].cpu()  # B, M

        return pred_from_retrieval, first_period_indices

    def retrieve_all_with_indices(self, data, train=False, device=torch.device('cpu')):
        """Retrieve all predictions with top-M indices for visualization."""
        assert(self.train_data_all_mg != None)

        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )

        retrievals = []
        all_indices = []
        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                pred_from_retrieval, topm_indices = self.retrieve_with_indices(
                    batch_x.float().to(device), index, train=train
                )
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                all_indices.append(topm_indices)

        retrievals = torch.cat(retrievals, dim=1)
        all_indices = torch.cat(all_indices, dim=0)  # N, M

        return retrievals, all_indices