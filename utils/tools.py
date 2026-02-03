import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.png'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def visual_with_retrieval(input_seq, true_seq, pred_seq, retrieved_input_patterns,
                          retrieved_output_patterns, name='./pic/test.png', channel_idx=-1):
    """
    Visualization with prediction and retrieved patterns side by side.

    Args:
        input_seq: Input sequence (seq_len, channels) or (seq_len,)
        true_seq: Ground truth prediction (pred_len, channels) or (pred_len,)
        pred_seq: Model prediction (pred_len, channels) or (pred_len,)
        retrieved_input_patterns: Retrieved input patterns (M, seq_len, channels)
        retrieved_output_patterns: Retrieved output patterns (M, pred_len, channels)
        name: Save path
        channel_idx: Which channel to visualize (-1 for last channel)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Handle multi-channel data - select one channel for visualization
    if len(input_seq.shape) > 1:
        input_seq = input_seq[:, channel_idx]
    if len(true_seq.shape) > 1:
        true_seq = true_seq[:, channel_idx]
    if len(pred_seq.shape) > 1:
        pred_seq = pred_seq[:, channel_idx]

    seq_len = len(input_seq)
    pred_len = len(true_seq)

    # Left Plot: Prediction
    ax1 = axes[0]
    # Combine input and ground truth for full sequence
    full_true = np.concatenate([input_seq, true_seq], axis=0)
    full_pred = np.concatenate([input_seq, pred_seq], axis=0)

    x_total = np.arange(len(full_true))
    x_input = np.arange(seq_len)
    x_pred = np.arange(seq_len, seq_len + pred_len)

    ax1.plot(x_total, full_true, label='Ground Truth', linewidth=2, color='blue')
    ax1.plot(x_pred, pred_seq, label='Prediction', linewidth=2, color='red', linestyle='--')
    ax1.axvline(x=seq_len, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.set_title('Prediction vs Ground Truth')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Right Plot: Retrieved Patterns
    ax2 = axes[1]
    M = retrieved_input_patterns.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, M))

    for m in range(M):
        # Get pattern for selected channel
        if len(retrieved_input_patterns.shape) > 2:
            pattern_input = retrieved_input_patterns[m, :, channel_idx]
            pattern_output = retrieved_output_patterns[m, :, channel_idx]
        else:
            pattern_input = retrieved_input_patterns[m, :]
            pattern_output = retrieved_output_patterns[m, :]

        full_pattern = np.concatenate([pattern_input, pattern_output], axis=0)
        alpha = 0.3 + 0.7 * (M - m) / M  # Higher rank = more visible
        ax2.plot(x_total[:len(full_pattern)], full_pattern,
                color=colors[m], alpha=alpha, linewidth=1.5,
                label=f'Pattern {m+1}' if m < 5 else None)

    ax2.axvline(x=seq_len, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Top-{M} Retrieved Patterns')
    if M <= 5:
        ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight', dpi=150)
    plt.close(fig)


def save_metadata(args, save_path):
    """
    Save experiment metadata (hyperparameters) to JSON file.

    Args:
        args: Argument namespace containing experiment settings
        save_path: Directory path to save metadata.json
    """
    import json

    # Convert args to dictionary
    if hasattr(args, '__dict__'):
        args_dict = vars(args).copy()
    else:
        args_dict = dict(args)

    # Remove non-serializable items
    for key in list(args_dict.keys()):
        try:
            json.dumps(args_dict[key])
        except (TypeError, ValueError):
            args_dict[key] = str(args_dict[key])

    metadata_path = os.path.join(save_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(args_dict, f, indent=4, default=str)

    print(f'Metadata saved to {metadata_path}')