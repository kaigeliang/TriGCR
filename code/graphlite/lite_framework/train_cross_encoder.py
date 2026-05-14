"""
Train Cross-Encoder Verifier for KGQA path verification.

Fixes vs v1:
- Train/val split at question level (no cross-contamination)
- Negative paths matched by hop length to positives
- Step-level loss logging
"""
import sys, random, argparse, os
sys.path.insert(0, 'src')
from src.utils import graph_utils, utils
from src.cross_encoder_verifier import CrossEncoderVerifier
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict


def build_training_data(dataset, n_neg_per_positive: int = 3):
    """Build (question, path, label) triples. Returns samples grouped by question."""
    qid_samples = defaultdict(list)  # question_id -> [(q, path, label), ...]

    for item in tqdm(dataset, desc='Building training data'):
        g = graph_utils.build_graph(item['graph'])
        q_ents = item['q_entity']
        a_ents_raw = item['a_entity']
        a_ents_lower = [a.lower() for a in a_ents_raw]
        question = item['question'].strip().rstrip('?')
        qid = item['id']

        # Positive: GT paths
        truth_paths = graph_utils.get_truth_paths(q_ents, a_ents_raw, g)
        pos_paths = set()
        for tp in truth_paths:
            if tp:
                pos_paths.add(utils.path_to_string(tp))

        if not pos_paths:
            continue

        # Negative: DFS paths NOT ending at answer, matched by hop length
        all_paths = graph_utils.dfs(g, q_ents, max_length=2)
        neg_by_len = defaultdict(list)  # hop_len -> [path_str]
        for p in all_paths:
            if not p:
                continue
            terminal = p[-1][2].lower()
            path_str = utils.path_to_string(p)
            if path_str not in pos_paths and terminal not in a_ents_lower:
                neg_by_len[len(p)].append(path_str)

        # Add positives
        for ps in pos_paths:
            qid_samples[qid].append((question, ps, 1.0))

        # For each positive, sample negatives of the same hop length
        pos_lens = [len(tp) for tp in truth_paths if tp]
        for plen in pos_lens:
            candidates = neg_by_len.get(plen, [])
            random.shuffle(candidates)
            for ns in candidates[:n_neg_per_positive]:
                qid_samples[qid].append((question, ns, 0.0))

    return qid_samples


def main(args):
    cwq_split = f'train[:{args.max_questions}]' if args.max_questions > 0 else 'train'
    wqsp_split = f'train[:{args.max_questions}]' if args.max_questions > 0 else 'train'
    ds_cwq = load_dataset('local_datasets/RoG-cwq', split=cwq_split)
    ds_wqsp = load_dataset('local_datasets/RoG-webqsp', split=wqsp_split)

    print(f'CWQ train: {len(ds_cwq)}, WebQSP train: {len(ds_wqsp)}')

    # Build data per question
    cwq_data = build_training_data(ds_cwq, n_neg_per_positive=args.n_neg)
    wqsp_data = build_training_data(ds_wqsp, n_neg_per_positive=args.n_neg)
    all_qids = list(cwq_data.keys()) + list(wqsp_data.keys())
    all_data = {**cwq_data, **wqsp_data}

    # Flatten for stats
    total_pos = sum(1 for qid in all_qids for s in all_data[qid] if s[2] > 0.5)
    total_neg = sum(1 for qid in all_qids for s in all_data[qid] if s[2] < 0.5)
    total_qids = len(all_qids)
    print(f'Questions with GT paths: {total_qids}')
    print(f'Positives: {total_pos}, Negatives: {total_neg}')

    # Train/val split at QUESTION level
    random.seed(42)
    random.shuffle(all_qids)
    split_n = int(0.9 * len(all_qids))
    train_qids = set(all_qids[:split_n])
    val_qids = set(all_qids[split_n:])

    train_samples = []
    val_samples = []
    for qid in train_qids:
        train_samples.extend(all_data[qid])
    for qid in val_qids:
        val_samples.extend(all_data[qid])

    print(f'Train: {len(train_qids)} questions, {len(train_samples)} samples')
    print(f'  Pos: {sum(1 for s in train_samples if s[2]>0.5)}, '
          f'Neg: {sum(1 for s in train_samples if s[2]<0.5)}')
    print(f'Val:   {len(val_qids)} questions, {len(val_samples)} samples')
    print(f'  Pos: {sum(1 for s in val_samples if s[2]>0.5)}, '
          f'Neg: {sum(1 for s in val_samples if s[2]<0.5)}')

    # Path hop distribution (s[1] is already a path string)
    def hop_count(s):
        return max(1, s.count(' -> ') // 2)  # "h -> r -> t" has 2 arrows per hop
    pos_hops = [hop_count(s[1]) for s in train_samples if s[2] > 0.5]
    neg_hops = [hop_count(s[1]) for s in train_samples if s[2] < 0.5]
    if pos_hops and neg_hops:
        print(f'Hop count - Pos: mean={sum(pos_hops)/len(pos_hops):.1f}, '
              f'Neg: mean={sum(neg_hops)/len(neg_hops):.1f} (should match)')

    # Model
    model = CrossEncoderVerifier(args.model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'Model: {args.model_name}, device: {device}')
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {param_count:,}')

    # DataLoaders
    def collate_fn(batch):
        qs = [b[0] for b in batch]
        ps = [b[1] for b in batch]
        lbs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        return model(qs, ps), lbs

    train_ds = VerifierDataset(train_samples)
    val_ds = VerifierDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Logging
    log_file = args.output_path.replace('.pt', '_log.csv')
    with open(log_file, 'w') as lf:
        lf.write('epoch,train_loss,train_acc,val_acc\n')
    step_log = args.output_path.replace('.pt', '_steps.csv')
    sf = open(step_log, 'w')
    sf.write('global_step,loss\n')

    best_val_acc = 0.0
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for logits, labels in pbar:
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
            global_step += 1

            if global_step % 200 == 0:
                sf.write(f'{global_step},{loss.item():.4f}\n')
                sf.flush()
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'acc': f'{train_correct/train_total:.4f}'})

        train_acc = train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        print(f'  Train loss: {avg_loss:.4f}, acc: {train_acc:.4f}')

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_fp = val_fn = val_tp = val_tn = 0  # for detailed metrics
        with torch.no_grad():
            for logits, labels in val_loader:
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                # Count TP/TN/FP/FN
                for p, l in zip(preds, labels):
                    if p > 0.5 and l > 0.5:
                        val_tp += 1
                    elif p > 0.5 and l < 0.5:
                        val_fp += 1
                    elif p < 0.5 and l > 0.5:
                        val_fn += 1
                    else:
                        val_tn += 1

        val_acc = val_correct / val_total
        val_prec = val_tp / max(1, val_tp + val_fp)
        val_rec = val_tp / max(1, val_tp + val_fn)
        print(f'  Val acc: {val_acc:.4f}, P: {val_prec:.4f}, R: {val_rec:.4f} '
              f'(TP:{val_tp} FP:{val_fp} TN:{val_tn} FN:{val_fn})')
        with open(log_file, 'a') as lf:
            lf.write(f'{epoch+1},{avg_loss:.4f},{train_acc:.4f},{val_acc:.4f}\n')

        if val_acc > best_val_acc + 0.005:
            best_val_acc = val_acc
            epochs_no_improve = 0
            model.save(args.output_path)
            print(f'  Saved best model to {args.output_path}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 2:
                print('  Early stop')
                break

    sf.close()
    print(f'Best val acc: {best_val_acc:.4f}')


class VerifierDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='microsoft/deberta-v3-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--n_neg', type=int, default=3,
                        help='negative samples per positive path')
    parser.add_argument('--max_questions', type=int, default=0,
                        help='max questions to use (0=all, for quick tests)')
    parser.add_argument('--output_path', default='models/cross_encoder_verifier.pt')
    args = parser.parse_args()
    main(args)
