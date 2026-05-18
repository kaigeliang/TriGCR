import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from tqdm import tqdm
from src.llms import get_registed_model
import os
from datasets import load_dataset, Dataset
from src.utils.qa_utils import eval_path_result_w_ans, extract_topk_prediction
from src import utils
import json
from multiprocessing import Pool
from functools import partial
from src.qa_prompt_builder import PathGenerationWithAnswerPromptBuilder

def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results

def prediction(data, processed_list, input_builder, model, output_topk=-1):
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    if id in processed_list:
        return None

    input_query, ground_paths, trie = input_builder.process_input(data)
    if trie is None:
        return None
    start_token_ids = model.tokenizer.convert_tokens_to_ids(
        input_builder.PATH_START_TOKEN
    )
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    input = model.prepare_model_prompt(input_query)
    prediction = model.generate_sentence(
        input,
        trie,
        start_token_ids=start_token_ids,
        end_token_ids=end_token_ids,
        enable_constrained_by_default=False,
    )
    if prediction is None:
        return None
    if output_topk is not None and output_topk > 0:
        prediction = extract_topk_prediction(prediction, output_topk)
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "ground_truth_paths": ground_paths,
        "input": input,
    }
    return result


def main(args, LLM):
    input_file = os.path.join(args.data_path, args.d)
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    post_fix = f"{args.prefix}{args.prompt_mode}-{args.generation_mode}-k{args.k}-index_len{args.index_path_length}"
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        post_fix += "_" + rule_postfix
    data_name = args.d + "_undirected" if args.undirected else args.d
    output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
    print("Save results to: ", output_dir)

    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = LLM(args)
    
    print("Prepare pipline for inference...")
    model.prepare_for_inference()
    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer,
        args.prompt_mode,
        index_path_length=args.index_path_length,
        undirected=args.undirected,
        add_rule=args.add_rule,
        constraint_aware=args.constraint_aware,
        max_index_paths=args.max_index_paths,
        require_shared_tail=args.require_shared_tail,
        embedding_guided=args.embedding_guided,
        embedding_model_path=args.embedding_model_path,
        embedding_candidate_paths=args.embedding_candidate_paths,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        embedding_max_length=args.embedding_max_length,
        hybrid_guided=args.hybrid_guided,
        hybrid_embedding_weight=args.hybrid_embedding_weight,
        hybrid_relation_weight=args.hybrid_relation_weight,
        hybrid_entity_weight=args.hybrid_entity_weight,
        hybrid_coverage_weight=args.hybrid_coverage_weight,
        hybrid_specificity_weight=args.hybrid_specificity_weight,
        hybrid_length_penalty=args.hybrid_length_penalty,
        filter_generic_sources=args.filter_generic_sources,
        generic_source_threshold=args.generic_source_threshold,
    )
    
    # Save args file
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    fout, processed_list =  get_output_file(os.path.join(output_dir, 'predictions.jsonl'), force=args.force)
    
    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                        output_topk=args.output_topk,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction(data, processed_list, input_builder, model, output_topk=args.output_topk)
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
            else:
                print("None result for: ", data["id"])
    fout.close()
            
    eval_path_result_w_ans(os.path.join(output_dir, 'predictions.jsonl'))
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='rmanluo')
    argparser.add_argument('--d', '-d', type=str, default='RoG-webqsp')
    argparser.add_argument('--split', type=str, default='test[:100]')
    argparser.add_argument('--index_path_length', type=int, default=2)
    argparser.add_argument('--predict_path', type=str, default='results/GenPaths')
    argparser.add_argument('--model_name', type=str, help="model_name for save results", default='gcr-Llama-2-7b-chat-hf')
    argparser.add_argument('--force', action='store_true', help="force to overwrite the results")
    argparser.add_argument("--n", type=int, default=1, help="number of processes")
    argparser.add_argument("--undirected", type=lambda x: (str(x).lower() == 'true'), default=False)
    argparser.add_argument("--debug", action="store_true", help="print debug information")
    argparser.add_argument("--prompt_mode", type=str, default="zero-shot", choices=["zero-shot", "mcq-zero-shot", "few-shot"])
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument(
        "--constraint_aware",
        action="store_true",
        help="rank and optionally filter KG-Trie paths by shared terminal-entity coverage",
    )
    argparser.add_argument(
        "--max_index_paths",
        type=int,
        default=None,
        help="maximum number of ranked paths to keep in the KG-Trie index",
    )
    argparser.add_argument(
        "--require_shared_tail",
        action="store_true",
        help="keep only paths whose terminal entity is reachable from multiple topic entities when possible",
    )
    argparser.add_argument(
        "--embedding_guided",
        action="store_true",
        help="rank candidate KG paths by question-path embedding similarity before building the KG-Trie",
    )
    argparser.add_argument(
        "--embedding_model_path",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="embedding model used for question-path path ranking",
    )
    argparser.add_argument(
        "--embedding_candidate_paths",
        type=int,
        default=10000,
        help="number of DFS candidate paths collected before embedding ranking",
    )
    argparser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64,
        help="batch size for embedding path ranking",
    )
    argparser.add_argument(
        "--embedding_device",
        type=str,
        default="cpu",
        help="device for the embedding model, e.g. cpu or cuda:0",
    )
    argparser.add_argument(
        "--embedding_max_length",
        type=int,
        default=128,
        help="maximum token length for embedding model inputs",
    )
    argparser.add_argument(
        "--hybrid_guided",
        action="store_true",
        help="rank candidate KG paths with embedding, relation lexical overlap, entity overlap, and structural coverage",
    )
    argparser.add_argument(
        "--hybrid_embedding_weight",
        type=float,
        default=1.0,
        help="weight for question-path embedding similarity in hybrid path ranking",
    )
    argparser.add_argument(
        "--hybrid_relation_weight",
        type=float,
        default=0.25,
        help="weight for relation-name keyword overlap in hybrid path ranking",
    )
    argparser.add_argument(
        "--hybrid_entity_weight",
        type=float,
        default=0.10,
        help="weight for entity-name keyword overlap in hybrid path ranking",
    )
    argparser.add_argument(
        "--hybrid_coverage_weight",
        type=float,
        default=0.20,
        help="weight for shared-tail structural coverage across topic entities",
    )
    argparser.add_argument(
        "--hybrid_specificity_weight",
        type=float,
        default=0.10,
        help="weight for preferring paths that start from specific rather than generic topic entities",
    )
    argparser.add_argument(
        "--hybrid_length_penalty",
        type=float,
        default=0.03,
        help="small penalty for longer paths in hybrid path ranking",
    )
    argparser.add_argument(
        "--filter_generic_sources",
        action="store_true",
        help="for multi-entity questions, drop paths from generic type-like topic entities when specific sources are available",
    )
    argparser.add_argument(
        "--generic_source_threshold",
        type=float,
        default=0.05,
        help="minimum source specificity needed to avoid generic-source filtering",
    )
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp_undirected/Llama-2-7b-chat-hf_align-spectoken-joint/test/predictions_3_False.jsonl",
    )
    argparser.add_argument(
        "--output_topk",
        type=int,
        default=-1,
        help="keep only the top-k generated candidates before saving and evaluating; -1 keeps all candidates",
    )
    argparser.add_argument("--prefix", type=str, default="")

    args, _  = argparser.parse_known_args()
    
    LLM = get_registed_model(args.model_name)
    LLM.add_args(argparser)
    
    args = argparser.parse_args()
    
    main(args, LLM)
