import src.utils as utils
import random
import math
import re
from typing import Callable
from .trie import MarisaTrie

ROG_ROMPT_TEMPLATE = """{instruction}

{input}"""

PROMPT_TEMPLATE = """{input}

{instruction}"""

class GraphConstrainedPromptBuilder(object):
    _embedding_model_cache = {}

    def __init__(
        self,
        tokenizer,
        prompt="zero-shot",
        undirected=False,
        index_path_length=2,
        add_rule=False,
        constraint_aware=False,
        max_index_paths=None,
        require_shared_tail=False,
        embedding_guided=False,
        embedding_model_path="sentence-transformers/all-MiniLM-L6-v2",
        embedding_candidate_paths=10000,
        embedding_batch_size=64,
        embedding_device="cpu",
        embedding_max_length=128,
        hybrid_guided=False,
        hybrid_embedding_weight=1.0,
        hybrid_relation_weight=0.25,
        hybrid_entity_weight=0.10,
        hybrid_coverage_weight=0.20,
        hybrid_specificity_weight=0.10,
        hybrid_length_penalty=0.03,
        filter_generic_sources=False,
        generic_source_threshold=0.05,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.undirected = undirected
        self.index_path_length = index_path_length
        self.add_rule = add_rule
        self.constraint_aware = constraint_aware
        self.max_index_paths = max_index_paths
        self.require_shared_tail = require_shared_tail
        self.embedding_guided = embedding_guided
        self.embedding_model_path = embedding_model_path
        self.embedding_candidate_paths = embedding_candidate_paths
        self.embedding_batch_size = embedding_batch_size
        self.embedding_device = embedding_device
        self.embedding_max_length = embedding_max_length
        self.hybrid_guided = hybrid_guided
        self.hybrid_embedding_weight = hybrid_embedding_weight
        self.hybrid_relation_weight = hybrid_relation_weight
        self.hybrid_entity_weight = hybrid_entity_weight
        self.hybrid_coverage_weight = hybrid_coverage_weight
        self.hybrid_specificity_weight = hybrid_specificity_weight
        self.hybrid_length_penalty = hybrid_length_penalty
        self.filter_generic_sources = filter_generic_sources
        self.generic_source_threshold = generic_source_threshold
        self.prompt_template = self.get_prompt_template(self.prompt)

    def get_path_collection_limit(self):
        if self.embedding_guided or self.hybrid_guided:
            return self.embedding_candidate_paths
        return self.max_index_paths

    def get_embedding_model(self):
        cache_key = (self.embedding_model_path, self.embedding_device)
        if cache_key not in self._embedding_model_cache:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path)
            model = AutoModel.from_pretrained(self.embedding_model_path)
            model.eval()
            model.to(self.embedding_device)
            self._embedding_model_cache[cache_key] = (tokenizer, model)
        return self._embedding_model_cache[cache_key]

    @staticmethod
    def path_to_embedding_text(path):
        pieces = []
        for index, (head, relation, tail) in enumerate(path):
            relation = relation.replace(".", " ").replace("_", " ")
            if index == 0:
                pieces.extend([head, relation, tail])
            else:
                pieces.extend([relation, tail])
        return " ".join(pieces)

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "before", "after", "between",
        "both", "by", "did", "do", "does", "for", "from", "has", "have", "had",
        "he", "her", "his", "in", "into", "is", "it", "its", "of", "on", "or",
        "that", "the", "their", "there", "these", "this", "to", "was", "were",
        "what", "when", "where", "which", "who", "whom", "whose", "with", "won",
    }
    GENERIC_ENTITY_TERMS = {
        "actor", "actors", "album", "award", "bachelor", "book", "city", "college",
        "company", "country", "degree", "director", "entity", "film", "films",
        "language", "man", "movie", "nation", "person", "place", "prime",
        "school", "state", "title", "university", "woman", "world", "year",
    }

    @classmethod
    def normalize_text_tokens(cls, text):
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        return {token for token in tokens if token not in cls.STOPWORDS and len(token) > 1}

    @classmethod
    def relation_tokens(cls, relation):
        relation = str(relation).replace(".", " ").replace("_", " ")
        return cls.normalize_text_tokens(relation)

    @staticmethod
    def normalized_overlap(left_terms, right_terms):
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms & right_terms) / math.sqrt(len(left_terms) * len(right_terms))

    @classmethod
    def source_specificity(cls, source):
        source_terms = cls.normalize_text_tokens(source)
        if not source_terms:
            return 0.0
        specific_terms = source_terms - cls.GENERIC_ENTITY_TERMS
        return len(specific_terms) / len(source_terms)

    def encode_texts(self, texts):
        import torch
        import torch.nn.functional as F

        tokenizer, model = self.get_embedding_model()
        all_embeddings = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start : start + self.embedding_batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.embedding_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.embedding_device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(F.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(all_embeddings, dim=0)

    def select_embedding_guided_paths(self, question, paths_list):
        if not paths_list or self.max_index_paths is None or self.max_index_paths <= 0:
            return paths_list

        query_embedding = self.encode_texts([question])[0]
        path_texts = [self.path_to_embedding_text(path) for path in paths_list]
        path_embeddings = self.encode_texts(path_texts)
        scores = path_embeddings @ query_embedding
        ranked_indices = scores.argsort(descending=True).tolist()
        selected_indices = ranked_indices[: self.max_index_paths]
        return [paths_list[index] for index in selected_indices]

    def maybe_filter_generic_source_paths(self, paths_list, source_entities):
        if not self.filter_generic_sources or len(source_entities) <= 1:
            return paths_list

        specific_sources = {
            source
            for source in source_entities
            if self.source_specificity(source) > self.generic_source_threshold
        }
        if not specific_sources:
            return paths_list

        filtered_paths = [path for path in paths_list if path and path[0][0] in specific_sources]
        return filtered_paths or paths_list

    def select_hybrid_guided_paths(self, question, paths_list, source_entities):
        if not paths_list or self.max_index_paths is None or self.max_index_paths <= 0:
            return paths_list

        paths_list = self.maybe_filter_generic_source_paths(paths_list, source_entities)

        query_embedding = self.encode_texts([question])[0]
        path_texts = [self.path_to_embedding_text(path) for path in paths_list]
        path_embeddings = self.encode_texts(path_texts)
        embedding_scores = (path_embeddings @ query_embedding).tolist()
        question_terms = self.normalize_text_tokens(question)

        tail_to_sources = {}
        for path in paths_list:
            if not path:
                continue
            tail_to_sources.setdefault(path[-1][-1], set()).add(path[0][0])

        source_count = max(1, len(source_entities))
        scored_paths = []
        for index, path in enumerate(paths_list):
            relation_terms = set()
            entity_terms = set()
            for head, relation, tail in path:
                relation_terms.update(self.relation_tokens(relation))
                entity_terms.update(self.normalize_text_tokens(head))
                entity_terms.update(self.normalize_text_tokens(tail))

            tail = path[-1][-1] if path else ""
            shared_tail_count = len(tail_to_sources.get(tail, set()))
            coverage_score = (shared_tail_count - 1) / max(1, source_count - 1)
            relation_score = self.normalized_overlap(question_terms, relation_terms)
            entity_score = self.normalized_overlap(question_terms, entity_terms)
            specificity_score = self.source_specificity(path[0][0]) if path else 0.0
            length_score = len(path) / max(1, self.index_path_length)

            score = (
                self.hybrid_embedding_weight * embedding_scores[index]
                + self.hybrid_relation_weight * relation_score
                + self.hybrid_entity_weight * entity_score
                + self.hybrid_coverage_weight * coverage_score
                + self.hybrid_specificity_weight * specificity_score
                - self.hybrid_length_penalty * length_score
            )
            scored_paths.append((score, len(path), path_texts[index], index))

        ranked_paths = sorted(scored_paths, key=lambda item: (-item[0], item[1], item[2]))
        selected_indices = [index for _, _, _, index in ranked_paths[: self.max_index_paths]]
        return [paths_list[index] for index in selected_indices]

    def select_constraint_aware_paths(self, paths_list, source_entities):
        if not paths_list:
            return paths_list

        tail_to_sources = {}
        for path in paths_list:
            if not path:
                continue
            source = path[0][0]
            tail = path[-1][-1]
            tail_to_sources.setdefault(tail, set()).add(source)

        def score(path):
            tail = path[-1][-1] if path else ""
            source_coverage = len(tail_to_sources.get(tail, set()))
            starts_from_topic = 1 if path and path[0][0] in source_entities else 0
            return (-source_coverage, -starts_from_topic, len(path), utils.path_to_string(path))

        ranked_paths = sorted(paths_list, key=score)
        if self.require_shared_tail and len(source_entities) > 1:
            integrated_paths = [
                path
                for path in ranked_paths
                if len(tail_to_sources.get(path[-1][-1], set())) > 1
            ]
            if integrated_paths:
                ranked_paths = integrated_paths

        if self.max_index_paths is not None and self.max_index_paths > 0:
            ranked_paths = ranked_paths[: self.max_index_paths]
        return ranked_paths

    def get_prompt_template(self, template_name):
        try:
            template_name = template_name.upper().replace("-", "_") + "_PROMPT"
            return self.__getattribute__(template_name)
        except:
            raise ValueError(f"The template name: {template_name} is not valid.")

    def format_input_with_template(self, question, start_entities, choices = []):
        if len(choices) > 0:
            return self.prompt_template.format(
                question=question, entities=",".join(start_entities), choices="\n".join(choices)
            )
        else:
            return self.prompt_template.format(
                question=question, entities=",".join(start_entities)
            )
    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results
    
    def get_graph_index(self, question_dict):
        # Try to load the pre-build index
        if "paths" in question_dict:
            paths_list = question_dict["paths"]
        else:
            g = utils.build_graph(question_dict["graph"], self.undirected)
            if self.add_rule:
                rules = question_dict['predicted_paths']
                if len(rules) > 0:
                    paths_list = self.apply_rules(g, rules, question_dict["q_entity"])
                else:
                    paths_list = utils.dfs_limited(g, question_dict["q_entity"], self.index_path_length, self.get_path_collection_limit())
            else:
                paths_list = utils.dfs_limited(g, question_dict["q_entity"], self.index_path_length, self.get_path_collection_limit())

        if self.hybrid_guided:
            paths_list = self.select_hybrid_guided_paths(
                question_dict["question"], paths_list, question_dict["q_entity"]
            )
        elif self.embedding_guided:
            paths_list = self.select_embedding_guided_paths(question_dict["question"], paths_list)

        if self.constraint_aware:
            paths_list = self.select_constraint_aware_paths(
                paths_list, question_dict["q_entity"]
            )

        paths_list_str = [utils.path_to_string(p) for p in paths_list]
        if len(paths_list_str) == 0:
            return None
        tokenized_paths = self.tokenizer(
            paths_list_str, padding=False, add_special_tokens=False
        ).input_ids
        tokenized_path_list = [
            ids + [self.tokenizer.eos_token_id] for ids in tokenized_paths
        ]
        return MarisaTrie(tokenized_path_list, max_token_id=len(self.tokenizer) + 1)

    def process_input(self, question_dict, return_tire = True):
        question = question_dict["question"]
        start_node = question_dict["q_entity"]
        anser_node = question_dict["a_entity"]
        choices = question_dict.get("choices", [])
        trie = None
        if return_tire:
            trie = self.get_graph_index(question_dict)

        g = utils.build_graph(question_dict["graph"], self.undirected)
        truth_paths = utils.get_truth_paths(start_node, anser_node, g)
        ground_paths = [utils.path_to_string(path) for path in truth_paths]

        if not question.endswith("?"):
            question += "?"

        input = self.format_input_with_template(question, start_node, choices=choices)
        return input, ground_paths, trie


class PathGenerationPromptBuilder(GraphConstrainedPromptBuilder):
    ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question: 
{question}
# Topic entities: 
{entities}
Reasoning path:
"""
    FEW_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question:
what is the name of justin bieber brother
# Topic entities: 
Justin Bieber
# Reasoning path:
Justin Bieber -> people.person.parents -> Jeremy Bieber -> people.person.children -> Jaxon Bieber

# Question:
where to fly into bali?
# Topic entities: 
Bali
# Reasoning path:
Bali -> location.location.contains -> Ngurah Rai International Airport

# Question:
what country is the grand bahama island in?
# Topic entities: 
Grand Bahama
# Reasoning path:
Grand Bahama -> location.location.containedby -> Bahamas

# Question:
who is the prime minister of ethiopia?
# Topic entities: 
Ethiopia
# Reasoning path:
Ethiopia -> government.governmental_jurisdiction.governing_officials -> m.0l0j4x3 -> government.government_position_held.office_holder -> Hailemariam Desalegn

# Question: 
{question}
# Topic entities: 
{entities}
# Reasoning path:
"""


class JointReasoningPromptBuilder(GraphConstrainedPromptBuilder):
    PATH_START_TOKEN = "<PATH>"
    PATH_END_TOKEN = "</PATH>"

    ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. It should start with <PATH> and end with </PATH>. When given a question, please generate some reasoning paths in the KG starting from the topic entities that you believe can aid in answering it. Then, use these reasoning paths to derive the answer to the question.

# Question: 
{question}
# Topic entities: 
{entities}
"""
    ZERO_SHOT_NO_MORE_THAN_10_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. It should start with <PATH> and end with </PATH>. When given a question, please generate some reasoning paths in the KG starting from the topic entities that you believe can aid in answering it. Then, use these reasoning paths to derive the answer to the question. Do not generate more than 10 reasoning paths.

# Question: 
{question}
# Topic entities: 
{entities}
"""
    MULTIPATH_GEN_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given the question, please generate some reasoning paths in the KG starting from the topic entities that you believe can aid in answering it.

# Question: 
{question}
# Topic entities: 
{entities}
# Reasoning paths:
"""
    def get_graph_index(self, question_dict):
        if "paths" in question_dict:
            paths_list = question_dict["paths"]
        else:
            g = utils.build_graph(question_dict["graph"], self.undirected)
            paths_list = utils.dfs_limited(g, question_dict["q_entity"], self.index_path_length, self.get_path_collection_limit())

        if self.hybrid_guided:
            paths_list = self.select_hybrid_guided_paths(
                question_dict["question"], paths_list, question_dict["q_entity"]
            )
        elif self.embedding_guided:
            paths_list = self.select_embedding_guided_paths(question_dict["question"], paths_list)

        if self.constraint_aware:
            paths_list = self.select_constraint_aware_paths(
                paths_list, question_dict["q_entity"]
            )

        paths_list_str = [f"{self.PATH_START_TOKEN}{utils.path_to_string(path)}{self.PATH_END_TOKEN}" for path in paths_list]
        if len(paths_list_str) == 0:
            return None
        tokenized_paths = self.tokenizer(
            paths_list_str, padding=False, add_special_tokens=False
        ).input_ids

        return MarisaTrie(tokenized_paths, max_token_id=len(self.tokenizer) + 1)

class PathGenerationWithAnswerPromptBuilder(JointReasoningPromptBuilder):
    ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question: 
{question}
# Topic entities: 
{entities}
"""
    MCQ_ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question: 
{question}
# Topic entities: 
{entities}
# Answer Choices:
{choices}
"""

class RetrievalPromptBuilder(GraphConstrainedPromptBuilder):
    entity_template = '''Question: {question}
Please generate entities that are relevant to the question.''' 
    relation_template = '''Question: {question}
Please generate relations that are relevant to the question.''' 
    triple_template = '''Question: {question}
Please generate triples that are relevant to the question.''' 

    def __init__(self, tokenizer, prompt="zero-shot", undirected = False,index_path_length=2, add_rule=False) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.undirected = undirected
        self.index_path_length = index_path_length
        self.add_rule = add_rule

    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results

    def get_graph_index(self, question_dict):
        cache_paths_list = None
        if "paths" in question_dict:
            cache_paths_list = question_dict["paths"]
        g = utils.build_graph(question_dict["graph"], self.undirected)
        if self.add_rule:
            rules = question_dict['predicted_paths']
            if len(rules) > 0:
                paths_list = self.apply_rules(g, rules, question_dict["q_entity"])
            else:
                if cache_paths_list is not None:
                    paths_list = cache_paths_list
                else:
                    paths_list = utils.dfs(g, question_dict["q_entity"], self.index_path_length)
        else:
            if cache_paths_list is not None:
                paths_list = cache_paths_list
            else:
                paths_list = utils.dfs(g, question_dict["q_entity"], self.index_path_length)

        paths_list_str = [utils.path_to_string(p) for p in paths_list]
        if len(paths_list_str) == 0:
            return None, None, None, None, None
        relation_list = set()
        entity_list = set()
        triple_list = set()
        for path in paths_list:
            for h, rel, t in path:
                relation_list.add(rel)
                entity_list.add(h)
                entity_list.add(t)
                triple_list.add((h, rel, t))
        # Build Entities Prefix Trie 
        tokenized_entity_list = self.tokenizer(list(entity_list), padding=False, add_special_tokens=False).input_ids
        tokenized_entity_list = [
                    ids + [self.tokenizer.eos_token_id] for ids in tokenized_entity_list
                ]
        entity_trie = MarisaTrie(tokenized_entity_list)
        
        # Build Relations Prefix Trie
        tokenized_relation_list = self.tokenizer(list(relation_list), padding=False, add_special_tokens=False).input_ids
        tokenized_relation_list = [
                    ids + [self.tokenizer.eos_token_id] for ids in tokenized_relation_list
                ]
        relation_trie = MarisaTrie(tokenized_relation_list)
        
        # Build Triples Prefix Trie
        triple_list_str = [f"[{h}, {rel}, {t}]" for h, rel, t in triple_list]
        tokenized_triple_list = self.tokenizer(triple_list_str, padding=False, add_special_tokens=False).input_ids
        tokenized_tokenized_triple_listpath_list = [
                    ids + [self.tokenizer.eos_token_id] for ids in tokenized_triple_list
                ]
        triple_trie = MarisaTrie(tokenized_tokenized_triple_listpath_list)
        
        return [entity_trie, relation_trie, triple_trie], entity_list, relation_list, triple_list, paths_list
    
    def process_input(self, question_dict, return_tire = True):
        question = question_dict["question"]
        start_node = question_dict["q_entity"]
        anser_node = question_dict["a_entity"]

        trie = None
        if return_tire:
            trie, entity_list, relation_list, triple_list, paths_list = self.get_graph_index(question_dict)

        g = utils.build_graph(question_dict["graph"], self.undirected)
        truth_paths = utils.get_truth_paths(start_node, anser_node, g)
        ground_paths = [utils.path_to_string(path) for path in truth_paths]

        if not question.endswith("?"):
            question += "?"

        entity_query = self.entity_template.format(question=question)
        relation_query = self.relation_template.format(question=question)
        triple_query = self.triple_template.format(question=question)
        return [entity_query, relation_query, triple_query], ground_paths, trie, entity_list, relation_list, triple_list, paths_list
    
class PromptBuilder(object):
    ROG_SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the given question. Please keep the answer as simple as possible and only return answers. Please return each answer at a new line."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    ROG_SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    ROG_SAQ_GRAPH_INSTRUCTION = """Based on the knowledge graph, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and only return answers."""
    SAQ_GRAPH_INSTRUCTION = """Based on the knowledge graph, please answer the given question. Please keep the answer as simple as possible."""
    MCQ_EXPLAIN_INSTRUCTION = """Based on the reasoning paths, please select the answers from the given choices and explain why."""
    SAQ_EXPLAIN_INSTRUCTION = """Based on the reasoning paths, please answer the given question and explain why."""
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    SIMPLE_GRAPH_CONTEXT = """Subgraph:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""
    GCR_PROMPT = "\nAnswer: "
    def __init__(
        self,
        add_rule=False,
        add_path=False,
        use_true=False,
        cot=False,
        explain=False,
        use_random=False,
        each_line=False,
        simple_graph=False,
        edge_graph=False,
        maximun_token=4096,
        use_rog_prompt = False,
        use_gcr = False,
        tokenize: Callable = lambda x: len(x),
    ):
        self.add_path = add_path
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.simple_graph = simple_graph
        self.edge_graph = edge_graph
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line
        self.use_rog_prompt = use_rog_prompt
        self.use_gcr = use_gcr
        
        if use_rog_prompt:
            self.SAQ_INSTRUCTION = self.ROG_SAQ_INSTRUCTION
            self.SAQ_RULE_INSTRUCTION = self.ROG_SAQ_RULE_INSTRUCTION
            self.SAQ_GRAPH_INSTRUCTION = self.ROG_SAQ_GRAPH_INSTRUCTION
            global PROMPT_TEMPLATE
            PROMPT_TEMPLATE = ROG_ROMPT_TEMPLATE

    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results

    def direct_answer(self, question_dict):
        graph = utils.build_graph(question_dict["graph"])
        entities = question_dict["q_entity"]
        rules = question_dict["predicted_paths"]
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction

    def process_input(self, question_dict):
        """
        Take question as input and return the input with prompt
        """
        question = question_dict["question"]

        if not question.endswith("?"):
            question += "?"

        if self.add_path:
            if self.use_true:
                lists_of_paths = question_dict["ground_paths"]
            else:
                lists_of_paths = question_dict['predicted_paths']
            # lists_of_paths = [utils.path_to_string(p) for p in retrieved_paths]
        else:
            if self.add_rule:
                graph = utils.build_graph(question_dict["graph"])
                entities = question_dict["q_entity"]
                if self.use_true:
                    rules = question_dict["ground_paths"]
                elif self.use_random:
                    _, rules = utils.get_random_paths(entities, graph)
                else:
                    rules = question_dict["predicted_paths"]
                if len(rules) > 0:
                    reasoning_paths = self.apply_rules(graph, rules, entities)
                    lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]
                    # context = "\n".join([utils.path_to_string(p) for p in reasoning_paths])
                else:
                    lists_of_paths = []
            # input += self.GRAPH_CONTEXT.format(context = context)
        if self.simple_graph:
            graph = utils.build_graph(question_dict["graph"])
            rules = question_dict["predicted_paths"]
            entities = question_dict["q_entity"]
            subgraph = set()
            if len(rules) > 0:
                reasoning_paths = self.apply_rules(graph, rules, entities)
                for p in reasoning_paths:
                    for triple in p:
                        subgraph.add(triple)
            subgraph = list(subgraph)
            list_of_graph = []
            for head, relation, tail in subgraph:
                list_of_graph.append(f"[ {head} | {relation} | {tail} ]")
        if self.edge_graph:
            graph = utils.build_graph(question_dict["graph"])
            rules = question_dict["predicted_paths"]
            entities = question_dict["q_entity"]
            subgraph = set()
            if len(rules) > 0:
                reasoning_paths = self.apply_rules(graph, rules, entities)
                for p in reasoning_paths:
                    for triple in p:
                        subgraph.add(triple)
            subgraph = list(subgraph)

        input = self.QUESTION.format(question=question)
        # MCQ
        if len(question_dict['choices']) > 0:
            choices = '\n'.join(question_dict['choices'])
            input += self.CHOICES.format(choices = choices)
            if self.add_rule or self.add_path:
                if self.explain:
                    instruction = self.MCQ_EXPLAIN_INSTRUCTION
                else:
                    instruction = self.MCQ_RULE_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        # SAQ
        else:
            if self.add_rule or self.add_path:
                if self.explain:
                    instruction = self.SAQ_EXPLAIN_INSTRUCTION
                else:
                    instruction = self.SAQ_RULE_INSTRUCTION
            elif self.simple_graph:
                instruction = self.SAQ_GRAPH_INSTRUCTION
            else:
                instruction = self.SAQ_INSTRUCTION

        if self.cot:
            instruction += self.COT

        if self.each_line:
            instruction += self.EACH_LINE

        if self.use_gcr:
            instruction += self.GCR_PROMPT
        
        if self.add_rule or self.add_path:
            other_prompt = PROMPT_TEMPLATE.format(
                instruction=instruction,
                input=self.GRAPH_CONTEXT.format(context="") + input,
            )
            context = self.check_prompt_length(
                other_prompt, lists_of_paths, self.maximun_token
            )

            input = self.GRAPH_CONTEXT.format(context=context) + input

        if self.simple_graph:
            other_prompt = PROMPT_TEMPLATE.format(
                instruction=instruction,
                input=self.SIMPLE_GRAPH_CONTEXT.format(context="") + input,
            )
            context = self.check_prompt_length(
                other_prompt, list_of_graph, self.maximun_token
            )

            input = self.SIMPLE_GRAPH_CONTEXT.format(context=context) + input

        if self.edge_graph:
            node2id = {}
            list_of_graph = []
            for h, r, t in subgraph:
                if h not in node2id:
                    node2id[h] = len(node2id)
                if t not in node2id:
                    node2id[t] = len(node2id)
                list_of_graph.append((node2id[h], r, node2id[t]))
            node_list = "\n".join([f"{v}: {k}" for k, v in node2id.items()])
            other_prompt = PROMPT_TEMPLATE.format(
                instruction=instruction,
                input=self.SIMPLE_GRAPH_CONTEXT.format(context=node_list) + input,
            )
            context = self.check_prompt_length(
                other_prompt, list_of_graph, self.maximun_token
            )

            input = (
                self.SIMPLE_GRAPH_CONTEXT.format(
                    context="node_id,node_attr\n"
                    + node_list
                    + "src,edge_attr,dst\n"
                    + context
                )
                + input
            )
   
        input = PROMPT_TEMPLATE.format(instruction=instruction, input=input)

        return input

    def check_prompt_length(self, prompt, list_of_paths, maximun_token):
        """Check whether the input prompt is too long. If it is too long, remove the first path and check again."""
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths
        else:
            # Shuffle the paths
            random.shuffle(list_of_paths)
            new_list_of_paths = []
            # check the length of the prompt
            for p in list_of_paths:
                tmp_all_paths = "\n".join(new_list_of_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    return "\n".join(new_list_of_paths)
                new_list_of_paths.append(p)
