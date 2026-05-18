"""Entity-Level Decoding QA: MC selection of relations/entities, no token generation."""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dvi_gcr')))
from src.utils import graph_utils, utils
from src.llms import get_registed_model
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from entity_level_decoder import EntityLevelDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test[:50]')
parser.add_argument('--model_name', default='llama318B')
parser.add_argument('--data_path', default='rmanluo')
parser.add_argument('--d', '-d', default='RoG-webqsp')
parser.add_argument('--shuffle_seed', type=int, default=None)
parser.add_argument('--max_depth', type=int, default=2)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--prefix', default='')
parser.add_argument('--verifier_path', default=None,
                   help='path to trained cross-encoder verifier checkpoint')
parser.add_argument('--entity_verifier_path', default=None,
                   help='path to trained entity-level verifier checkpoint')
parser.add_argument('--verifier_model', default='microsoft/deberta-v3-base',
                   help='base model for verifier')
args1, _ = parser.parse_known_args()
LLM = get_registed_model(args1.model_name)
LLM.add_args(parser)
args = parser.parse_args()

tokenizer_source = args.model_path or args.verifier_model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
dataset = load_dataset(os.path.join(args.data_path, args.d), split=args.split)
if args.shuffle_seed is not None:
    dataset = dataset.shuffle(seed=args.shuffle_seed)

if args.entity_verifier_path and args.verifier_path:
    # Lite mode: both verifiers loaded, skip Llama loading
    decoder = EntityLevelDecoder(None, tokenizer)
    decoder.load_verifier(args.verifier_path, args.verifier_model)
    decoder.load_entity_verifier(args.entity_verifier_path, args.verifier_model)
    print(f"Lite mode: verifiers only, skipping Llama load")
else:
    model = LLM(args)
    model.prepare_for_inference()
    decoder = EntityLevelDecoder(model.model, tokenizer)
    if args.verifier_path:
        decoder.load_verifier(args.verifier_path, args.verifier_model)
        print(f"Loaded verifier from {args.verifier_path}")
    if args.entity_verifier_path:
        decoder.load_entity_verifier(args.entity_verifier_path, args.verifier_model)
        print(f"Loaded entity verifier from {args.entity_verifier_path}")
    print(f"Loaded entity verifier from {args.entity_verifier_path}")

post_fix = f"{args.prefix}_entdec_k{args.top_k}_d{args.max_depth}"
output_dir = os.path.join('results/EntityDecode', f"{args.d}_{args.model_name}_{args.split}", post_fix)
os.makedirs(output_dir, exist_ok=True)
fout = open(os.path.join(output_dir, 'predictions.jsonl'), 'w')
print(f"Save results to: {output_dir}")

total = correct_sum = hit_sum = 0
for sample in tqdm(dataset):
    g = graph_utils.build_graph(sample['graph'])
    q_ents = sample['q_entity']
    a_ents = [a.lower() for a in sample['a_entity']]

    try:
        candidates = decoder.decode(g, q_ents, sample['question'],
                                    max_depth=args.max_depth, top_k=args.top_k)
    except Exception as e:
        print(f"ERROR on {sample['id']}: {e}")
        import traceback; traceback.print_exc()
        continue

    total += 1
    predicted_answers = [c[0] for c in candidates]
    predicted_paths = []
    for entity, path, score in candidates:
        path_str = utils.path_to_string(path)
        predicted_paths.append(f"# Reasoning Path:\n{path_str}\n# Answer:\n{entity}")

    gt_lower = set(a_ents)
    pred_lower = set(a.lower() for a in predicted_answers)
    acc = len(gt_lower & pred_lower) / len(gt_lower) if gt_lower else 0.0
    correct_sum += acc
    hit_sum += 1.0 if (gt_lower & pred_lower) else 0.0

    truth_paths = graph_utils.get_truth_paths(q_ents, sample['a_entity'], g)
    fout.write(json.dumps({
        "id": sample['id'], "question": sample['question'],
        "prediction": predicted_paths, "ground_truth": sample['answer'],
        "ground_truth_paths": [utils.path_to_string(tp) for tp in truth_paths if tp], "input": "",
    }) + "\n")

fout.close()
print(f"Results ({total}): Acc={correct_sum/total*100:.2f}% Hit={hit_sum/total*100:.1f}%")
