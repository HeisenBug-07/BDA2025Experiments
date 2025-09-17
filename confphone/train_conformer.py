import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from data_load import *
import logging
import scoring
import compute_eer
from sklearn.metrics import balanced_accuracy_score
from torch.nn.functional import pairwise_distance
from data_load import PairedDataset, collate_fn_paired
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()


def validation(valid_txt, model, model_name, device, kaldi, num_lang):
    valid_set = RawFeatures(valid_txt)
    valid_data = DataLoader(dataset=valid_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_atten)
    model.eval()
    correct = 0
    total = 0
    scores = 0
    with torch.no_grad():
        for step, (utt, labels, seq_len) in enumerate(valid_data):
            utt = utt.to(device=device, dtype=torch.float)
            labels = labels.to(device)
            atten_mask = get_atten_mask(seq_len, utt.size(0))
            atten_mask = atten_mask.to(device=device)
            # Forward pass
            outputs = model(utt, atten_mask)
            predicted = torch.argmax(outputs, -1)
            total += labels.size(-1)
            correct += (predicted == labels).sum().item()
            if step == 0:
                scores = outputs
            else: 
                scores = torch.cat((scores, outputs), dim=0)
    acc = correct / total
    logging.info('Current Acc.: {:.4f} %'.format(100 * acc))
    # for balanced Acc.
    prediction_all = torch.argmax(scores, -1).squeeze().cpu().numpy()
    with open(valid_txt, 'r') as f:
        lines = f.readlines()
    labels_array = np.array([int(x.split()[-1].strip()) for x in lines])

    scores = scores.squeeze().cpu().numpy()
    logging.info(f"{scores.shape}")
    trial_txt = 'trial_{}.txt'.format(model_name)
    score_txt = 'score_{}.txt'.format(model_name)
    scoring.get_trials(valid_txt, num_lang, trial_txt)
    scoring.get_score(valid_txt, scores, num_lang, score_txt)
    eer_txt = trial_txt.replace('trial', 'eer')
    target_score, non_target_score, score_all = compute_eer.load_file(score_txt, trial_txt)
    p_miss, p_fa = compute_eer.compute_rocch(target_score, non_target_score)
    eer = compute_eer.rocch2eer(p_miss, p_fa)
    cavg = scoring.compute_cavg(trial_txt, score_txt)
    logging.info(f"EER: {eer}")
    logging.info(f"Cavg: {cavg}")
    logging.info(f"Balanced Acc.: {balanced_accuracy_score(labels_array, prediction_all)}")

    return cavg


def main():
    parser = argparse.ArgumentParser(description='paras for training')
    parser.add_argument('--dim', type=int, help='dim of input features',
                        default=392)
    parser.add_argument('--model', type=str, help='model name',
                        default='Transformer')
    parser.add_argument('--train', type=str, help='training data, in .txt')
    parser.add_argument('--test', type=str, help='testing data, in .txt')
    parser.add_argument('--batch', type=int, help='batch size',
                        default=16)
    parser.add_argument('--warmup', type=int, help='num of epochs',
                        default=12000)
    parser.add_argument('--epochs', type=int, help='num of epochs',
                        default=5)
    parser.add_argument('--lang', type=int, help='num of language classes',
                        default=12)  # Updated to 12 languages
    parser.add_argument('--lr', type=float, help='initial learning rate',
                        default=0.0001)
    parser.add_argument('--device', type=int, help='Device name',
                        default=0)
    parser.add_argument('--kaldi', type=str, help='kaldi root', default='/home/hexin/Desktop/kaldi/')
    parser.add_argument('--seed', type=int, help='Device name',
                        default=0)
    parser.add_argument('--accum_steps', type=int, help='gradient accumulation steps', default=1)
    parser.add_argument('--use_amp', action='store_true', help='use mixed precision training', default=False)

    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = Conformer(input_dim=392,
                      feat_dim=32,  # Reduced feature dimension
                      d_k=32,       # Reduced key dimension
                      d_v=32,       # Reduced value dimension
                      n_heads=4,    # Reduced number of attention heads
                      d_ff=1024,    # Reduced feed-forward dimension
                      max_len=100000,
                      dropout=0.1,
                      device=device,
                      n_lang=args.lang)  # Use 12 languages
    model.to(device)

    train_txt = args.train
    train_set = RawFeatures(train_txt)  # Use RawFeatures for single feature-label pairs
    train_data = DataLoader(dataset=train_set,
                            batch_size=args.batch,
                            pin_memory=True,
                            num_workers=4,
                            shuffle=True,
                            collate_fn=collate_fn_atten)  # Use collate_fn_atten
    loss_func_CRE = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_step = len(train_data)
    warm_up_with_cosine_lr = lambda step: step / args.warmup \
        if step <= args.warmup \
        else 0.5 * (math.cos((step - args.warmup) / (args.epochs * total_step - args.warmup) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    scaler = GradScaler(enabled=args.use_amp)  # Initialize GradScaler for mixed precision
    # Train the model

    logger = get_logger('confPhone_iv.log')

    logger.info('start training:')
    
    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()  # Clear gradients at the start of each epoch
        for step, (utt, labels, seq_len) in enumerate(train_data):  # Single input
            utt = utt.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)

            # Forward pass with mixed precision
            with autocast(enabled=args.use_amp):
                outputs = model(utt, get_atten_mask(seq_len, utt.size(0)).to(device))
                loss = loss_func_CRE(outputs, labels)
                loss = loss / args.accum_steps  # Normalize loss for gradient accumulation

            # Backward pass
            scaler.scale(loss).backward()

            # Perform optimizer step after accumulating gradients
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_data):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % 1000 == 0:
                logger.info(f'Epoch {epoch+1}/{args.epochs} step: {step+1}/{total_step} loss: {loss.item()}')

            scheduler.step()

        if epoch >= args.epochs - 5:
            torch.save(model.state_dict(), '{}confPhone_iv{}.ckpt'.format(args.model, epoch))
            validation(args.test, model, args.model, device, kaldi=args.kaldi, num_lang=args.lang)
    logging.info("Training completed")

if __name__ == "__main__":
    main()
