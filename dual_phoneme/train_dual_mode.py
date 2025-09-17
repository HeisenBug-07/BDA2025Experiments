import os
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from data_load import *
import scoring
import subprocess
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_

def get_atten_mask(seq_lens, batch_size):
    max_len = max(seq_lens)
    mask = torch.zeros(batch_size, max_len, max_len)
    for i, length in enumerate(seq_lens):
        mask[i, :length, :length] = 1
    return mask.bool()

def get_atten_mask_student(seq_lens, batch_size, mask_type='fix', win_len=15):
    max_len = max(seq_lens)
    mask = torch.ones(batch_size, max_len, max_len)
    for i in range(batch_size):
        seq_len = seq_lens[i]
        if mask_type == 'fix':
            window = min(win_len, seq_len)
            mask[i, :window, :window] = 0
        elif mask_type == 'random':
            if seq_len > win_len:
                start = random.randint(0, seq_len - win_len)
                mask[i, start:start+win_len, start:start+winlen] = 0
            else:
                mask[i, :seq_len, :seq_len] = 0
    return mask.bool()

def evaluate(model, test_data, device, loss_func_CRE):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for utt, labels, seq_len in test_data:
            utt = utt.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            atten_mask = get_atten_mask(seq_len, utt.size(0)).to(device)

            with autocast():
                outputs = model(utt, seq_len, atten_mask)
                loss = loss_func_CRE(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_data)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--dim', type=int, default=392, help='dim of input features')
    parser.add_argument('--model', type=str, default='XSA_E2E', help='model name')
    parser.add_argument('--train', type=str, required=True, help='training data')
    parser.add_argument('--test', type=str, required=True, help='test data')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--optim', type=str, default='warmcosine', help='optimizer')
    parser.add_argument('--warmup', type=int, default=2400, help='warmup steps')
    parser.add_argument('--epochs', type=int, default=20, help='num of epochs')
    parser.add_argument('--lang', type=int, default=12, help='num of languages')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--device', type=int, default=0, help='device ID')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--temperature', type=float, default=2.0, help='temperature')
    parser.add_argument('--window', type=str, default='fix', help='window type')
    parser.add_argument('--winlen', type=int, default=15, help='window length')
    parser.add_argument('--alpha', type=float, default=0.33, help='teacher weight')
    parser.add_argument('--beta', type=float, default=0.33, help='student weight')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='gradient accumulation steps')
    parser.add_argument('--min_batch', type=int, default=4, help='minimum batch size for dynamic adjustment')
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()
    grad_accum_steps = args.grad_accum_steps
    min_batch = args.min_batch

    if args.model == 'XSA_E2E':
        model = X_Transformer_E2E_LID(
            input_dim=args.dim,
            feat_dim=64,
            d_k=64,
            d_v=64,
            d_ff=2048,
            n_heads=4,
            dropout=0.1,
            n_lang=args.lang,
            max_seq_len=10000,
            device=device
        )
    elif args.model == 'Transformer':
        model = Transformer_E2E_LID(
            input_dim=args.dim,
            feat_dim=64,
            d_k=64,
            d_v=64,
            d_ff=2048,
            n_heads=8,
            dropout=0.1,
            n_lang=args.lang,
            max_seq_len=10000,
            device=device
        )
    elif args.model == 'Conformer':
        model = Conformer(
            input_dim=args.dim,
            feat_dim=64,
            d_k=64,
            d_v=64,
            n_heads=8,
            d_ff=2048,
            max_len=10000,
            dropout=0.1,
            n_lang=args.lang,
            device=device
        )
    
    model.to(device)

    # Data loaders
    train_set = RawFeatures(args.train)
    train_data = DataLoader(
        dataset=train_set,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn_atten,
        num_workers=4,
        pin_memory=True
    )
    test_set = RawFeatures(args.test)
    test_data = DataLoader(
        dataset=test_set,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_fn_atten,
        num_workers=4,
        pin_memory=True
    )

    # Loss functions
    loss_func_CRE = nn.CrossEntropyLoss().to(device)
    KD_loss_func = nn.KLDivLoss(reduction='batchmean').to(device)

    if args.optim == 'warmcosine':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs * len(train_data)
        )
    elif args.optim == 'noam':
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (512 ** (-0.5)) * min((step + 1) ** (-0.5), (step + 1) * args.warmup ** (-1.5))
        )

    for epoch in tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, (utt, labels, seq_len) in enumerate(train_data):
            try:
                utt = utt.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                
                atten_mask = get_atten_mask(seq_len, utt.size(0)).to(device)
                atten_mask_student = get_atten_mask_student(
                    seq_len, utt.size(0),
                    mask_type=args.window,
                    win_len=args.winlen
                ).to(device)

                with autocast():
                    outputs = model(utt, seq_len, atten_mask)
                    outputs_student = model(utt, seq_len, atten_mask_student)
                    loss_teacher = loss_func_CRE(outputs, labels)
                    loss_student = loss_func_CRE(outputs_student, labels)
                    loss_kd = KD_loss_func(
                        F.log_softmax(outputs_student/args.temperature, dim=1),
                        F.softmax(outputs/args.temperature, dim=1)
                    ) * (args.temperature ** 2)
                    loss = args.alpha * loss_teacher + args.beta * loss_student + (1 - args.alpha - args.beta) * loss_kd
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_data):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                total_loss += loss.item() * grad_accum_steps

                if step % 100 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs} | Step {step}/{len(train_data)} | Loss: {total_loss / (step + 1):.4f}")

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if args.batch > min_batch:
                    args.batch = max(args.batch // 2, min_batch)
                    print(f"CUDA OOM. Reducing batch size to {args.batch}.")
                    train_data = DataLoader(
                        dataset=train_set,
                        batch_size=args.batch,
                        shuffle=True,
                        collate_fn=collate_fn_atten,
                        num_workers=4,
                        pin_memory=True
                    )
                else:
                    raise RuntimeError("Minimum batch size reached. Unable to continue training due to OOM.")

        # Evaluate after epoch
        test_loss, test_acc = evaluate(model, test_data, device, loss_func_CRE)
        print(f"[Eval] Epoch {epoch+1}: Test Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}")

        if epoch >= args.epochs - 5:
            torch.save(model.state_dict(), f"{args.model}_dualphoneiisc_{epoch}.pt")

if __name__ == "__main__":
    main()

