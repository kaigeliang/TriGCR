"""Train Cross-Encoder for entity-level verification. Entity scoring: (Q, path, entity) → correct?"""
import sys, random, argparse, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dvi_gcr')))
from src.utils import graph_utils, utils
from cross_encoder_verifier import CrossEncoderVerifier
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict


def build_entity_data(dataset, n_neg_per_pos: int = 3):
    """Build (question, path_str, label) at entity level. qid → samples."""
    qid_samples = defaultdict(list)
    for item in tqdm(dataset, desc='Building entity data'):
        g = graph_utils.build_graph(item['graph'])
        q_ents = item['q_entity']
        a_ents_raw = item['a_entity']
        a_ents_lower = [a.lower() for a in a_ents_raw]
        question = item['question'].strip().rstrip('?')
        qid = item['id']

        # GT paths → positive samples
        truth_paths = graph_utils.get_truth_paths(q_ents, a_ents_raw, g)
        if not truth_paths:
            continue

        # For each GT path, extract (relation, entity)
        pos_samples = set()
        for tp in truth_paths:
            if not tp:
                continue
            path_str = utils.path_to_string(tp)
            pos_samples.add(path_str)

        if not pos_samples:
            continue

        # Negative: same first-relation, different entity (wrong)
        # Group DFS paths by first-relation NL
        all_paths = graph_utils.dfs(g, q_ents, max_length=2)
        neg_by_rel = defaultdict(list)
        for p in all_paths:
            if not p:
                continue
            ps = utils.path_to_string(p)
            if ps in pos_samples:
                continue
            terminal = p[-1][2].lower()
            if terminal in a_ents_lower:
                continue
            first_rel_nl = p[0][1].rsplit('.', 1)[-1].replace('_s', '').replace('_', ' ')
            neg_by_rel[first_rel_nl].append(ps)

        for ps in pos_samples:
            qid_samples[qid].append((question, ps, 1.0))

        # For each positive, sample negatives from same relation
        for tp in truth_paths:
            if not tp:
                continue
            first_nl = tp[0][1].rsplit('.', 1)[-1].replace('_s', '').replace('_', ' ')
            neg_candidates = neg_by_rel.get(first_nl, [])
            random.shuffle(neg_candidates)
            for ns in neg_candidates[:n_neg_per_pos]:
                qid_samples[qid].append((question, ns, 0.0))

    return qid_samples


def main(args):
    cwq_split = f'train[:{args.max_questions}]' if args.max_questions > 0 else 'train'
    wqsp_split = f'train[:{args.max_questions}]' if args.max_questions > 0 else 'train'
    ds_cwq = load_dataset(os.path.join(args.data_path, 'RoG-cwq'), split=cwq_split)
    ds_wqsp = load_dataset(os.path.join(args.data_path, 'RoG-webqsp'), split=wqsp_split)
    print(f'CWQ: {len(ds_cwq)}, WebQSP: {len(ds_wqsp)}')

    cwq_data = build_entity_data(ds_cwq, n_neg_per_pos=args.n_neg)
    wqsp_data = build_entity_data(ds_wqsp, n_neg_per_pos=args.n_neg)
    all_qids = list(cwq_data.keys()) + list(wqsp_data.keys())
    all_data = {**cwq_data, **wqsp_data}

    total_pos = sum(1 for q in all_qids for s in all_data[q] if s[2] > 0.5)
    total_neg = sum(1 for q in all_qids for s in all_data[q] if s[2] < 0.5)
    print(f'Questions: {len(all_qids)}, Pos: {total_pos}, Neg: {total_neg}')

    # Question-level split
    random.seed(42)
    random.shuffle(all_qids)
    n_train = int(0.9 * len(all_qids))
    train_qids = set(all_qids[:n_train])
    val_qids = set(all_qids[n_train:])
    train_samples = [s for q in train_qids for s in all_data[q]]
    val_samples = [s for q in val_qids for s in all_data[q]]
    print(f'Train: {len(train_qids)} q, {len(train_samples)} samples')
    print(f'Val:   {len(val_qids)} q, {len(val_samples)} samples')

    # Model
    model = CrossEncoderVerifier(args.model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'Model: {args.model_name}, device: {device}')

    def collate_fn(batch):
        qs = [b[0] for b in batch]
        ps = [b[1] for b in batch]
        lbs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        return model(qs, ps), lbs

    train_ds = VerifierDataset(train_samples)
    val_ds = VerifierDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    log_file = args.output_path.replace('.pt', '_log.csv')
    with open(log_file, 'w') as lf:
        lf.write('epoch,train_loss,train_acc,val_acc\n')

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0; train_correct = 0; train_total = 0
        for logits, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        train_acc = train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        print(f'  Train loss: {avg_loss:.4f}, acc: {train_acc:.4f}')

        model.eval()
        val_correct = 0; val_total = 0
        val_tp = val_fp = val_tn = val_fn = 0
        with torch.no_grad():
            for logits, labels in val_loader:
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                for p, l in zip(preds, labels):
                    if p > 0.5 and l > 0.5: val_tp += 1
                    elif p > 0.5 and l < 0.5: val_fp += 1
                    elif p < 0.5 and l > 0.5: val_fn += 1
                    else: val_tn += 1

        val_acc = val_correct / val_total
        val_prec = val_tp / max(1, val_tp + val_fp)
        val_rec = val_tp / max(1, val_tp + val_fn)
        print(f'  Val acc: {val_acc:.4f}, P: {val_prec:.4f}, R: {val_rec:.4f} (TP:{val_tp} FP:{val_fp} TN:{val_tn} FN:{val_fn})')
        with open(log_file, 'a') as lf:
            lf.write(f'{epoch+1},{avg_loss:.4f},{train_acc:.4f},{val_acc:.4f}\n')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(args.output_path)
            print(f'  Saved to {args.output_path}')

    print(f'Best val acc: {best_val_acc:.4f}')


class VerifierDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='rmanluo')
    p.add_argument('--model_name', default='microsoft/deberta-v3-base')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--n_neg', type=int, default=3)
    p.add_argument('--max_questions', type=int, default=0)
    p.add_argument('--output_path', default='models/entity_verifier.pt')
    args = p.parse_args()
    main(args)
