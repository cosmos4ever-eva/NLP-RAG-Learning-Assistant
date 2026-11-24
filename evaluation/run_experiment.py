"""Reproducible dev retrieval audit and controlled model generation.

Gold fields are used only by the scorer, never by retrieval or generation.
Raw artifacts contain private course excerpts; do not publish them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from parsing.parser import parse_all_course_files
from vector_search.retriever import VectorRetriever
from graph_rag.pipeline import build_prompt, get_kg_triples, ECNUGPT_BASE_URL, ECNUGPT_MODEL
import requests


PROVIDERS = {
    'ecnu': {
        'base_url': ECNUGPT_BASE_URL,
        'default_model': ECNUGPT_MODEL,
        'key_env': 'ECNUGPT_API_KEY',
        'system_prompt': '你是华东师范大学大模型 ChatECNU，也是一个专业、耐心、清晰的 NLP 课程教学助手。',
    },
    'deepseek': {
        'base_url': 'https://api.deepseek.com',
        'default_model': 'deepseek-flash',
        'key_env': 'DEEPSEEK_API_KEY',
        'system_prompt': '你是一个专业、耐心、清晰的 NLP 课程教学助手。',
    },
}


def resolve_provider(name, model=None):
    if name not in PROVIDERS:
        raise ValueError('Unknown provider: ' + name)
    config = dict(PROVIDERS[name])
    config['name'] = name
    config['model'] = model or config['default_model']
    config['temperature'] = None if config['model'] == 'deepseek-flash' else 0.3
    config['thinking_disabled'] = name == 'deepseek'
    return config


def build_model_payload(prompt, provider):
    payload = {
        'model': provider['model'],
        'max_tokens': 1200,
        'stream': False,
        'messages': [
            {'role': 'system', 'content': provider['system_prompt']},
            {'role': 'user', 'content': prompt},
        ],
    }
    if provider['temperature'] is not None:
        payload['temperature'] = provider['temperature']
    if provider['thinking_disabled']:
        payload['thinking'] = {'type': 'disabled'}
    return payload


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(split):
    cases = [
        json.loads(line)
        for line in (ROOT / 'evaluation/questions.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    return [row for row in cases if row['split'] == split]


def verify_model_access(session, key, provider):
    try:
        response = session.get(
            provider['base_url'] + '/models',
            headers={'Authorization': 'Bearer ' + key},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SystemExit(provider['name'] + ' auth preflight could not connect: ' + str(exc)) from exc
    if response.status_code != 200:
        raise SystemExit(
            provider['name'] + ' auth preflight failed, HTTP ' + str(response.status_code)
            + '; check ' + provider['key_env'] + '. No model calls made.'
        )


def post_model(session, key, payload, provider):
    last_error = None
    for attempt in range(3):
        try:
            response = session.post(
                provider['base_url'] + '/chat/completions',
                headers={'Authorization': 'Bearer ' + key},
                json=payload,
                timeout=120,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit('Model request failed after retries: ' + str(exc)) from exc
        if response.status_code == 200:
            return response
        if response.status_code not in {429, 500, 502, 503, 504}:
            raise SystemExit(
                'Model request failed, HTTP ' + str(response.status_code)
                + '; partial artifacts preserved.'
            )
        last_error = 'HTTP ' + str(response.status_code)
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise SystemExit('Model request failed after retries: ' + str(last_error))


def generate_answer(case, system, results, triples, key, direct, provider):
    prompt = build_prompt(
        case['question'],
        results if system != 'llm_only' else [],
        triples if system == 'graph_rag' else [],
        allow_model_knowledge=(system == 'llm_only'),
    )
    payload = build_model_payload(prompt, provider)
    worker_session = requests.Session()
    worker_session.trust_env = not direct
    start = time.perf_counter()
    response = post_model(worker_session, key, payload, provider)
    data = response.json()
    return {
        'id': case['id'],
        'system': system,
        'prompt': prompt,
        'answer': data['choices'][0]['message']['content'],
        'latency_seconds': time.perf_counter() - start,
        'provider': provider['name'],
        'model': data.get('model', provider['model']),
        'usage': data.get('usage'),
        'finish_reason': data['choices'][0].get('finish_reason'),
    }


def summarize_retrieval(rows):
    valid = [row for row in rows if row['anchor_hit_at_3'] is not None]
    summary = {
        'cases': len(rows),
        'evidence_cases': len(valid),
        'anchor_hits': sum(row['anchor_hit_at_3'] for row in valid),
        'anchor_hit_at_3': (
            sum(row['anchor_hit_at_3'] for row in valid) / len(valid)
            if valid else None
        ),
        'context_hits': sum(row['context_hit'] for row in valid),
        'context_hit_rate': (
            sum(row['context_hit'] for row in valid) / len(valid)
            if valid else None
        ),
        'by_category': {},
        'answer_quality_scored': False,
        'human_annotations_completed': False,
    }
    for category in sorted({row['category'] for row in valid}):
        subset = [row for row in valid if row['category'] == category]
        summary['by_category'][category] = {
            'n': len(subset),
            'anchor_hits': sum(row['anchor_hit_at_3'] for row in subset),
            'context_hits': sum(row['context_hit'] for row in subset),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--split', choices=('dev', 'test'), default='dev')
    ap.add_argument('--generate', action='store_true')
    ap.add_argument('--direct', action='store_true', help='Connect without environment proxies; TLS verification remains enabled.')
    ap.add_argument('--start-index', type=int, default=0, help='Zero-based inclusive index within the selected split.')
    ap.add_argument('--end-index', type=int, help='Zero-based exclusive index within the selected split.')
    ap.add_argument('--systems', default='llm_only,tfidf_rag,graph_rag')
    ap.add_argument('--provider', choices=sorted(PROVIDERS), default='ecnu')
    ap.add_argument('--model', help='Override the provider default model.')
    args = ap.parse_args()
    provider = resolve_provider(args.provider, args.model)
    systems = tuple(part.strip() for part in args.systems.split(',') if part.strip())
    allowed_systems = {'llm_only','tfidf_rag','graph_rag'}
    if not systems or not set(systems) <= allowed_systems:
        raise SystemExit('Invalid --systems value.')
    session = requests.Session()
    session.trust_env = not args.direct
    key = os.getenv(provider['key_env'], '').strip()
    if args.generate and not key:
        raise SystemExit(provider['key_env'] + ' is missing; no model calls made.')
    if args.generate:
        verify_model_access(session, key, provider)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cases = load_cases(args.split)
    selected_range = [args.start_index, args.end_index if args.end_index is not None else len(cases)]
    cases = cases[selected_range[0]:selected_range[1]]
    materials = parse_all_course_files()
    chunks = [c for doc in materials.values() for c in doc['chunks']]
    retriever = VectorRetriever()
    # Keep experimental indexing in memory; do not overwrite the application's index.
    retriever._save = lambda _: None
    retriever.build_index(chunks)
    manifest = {'split':args.split,'split_index_range':selected_range,'test_evaluated':args.split == 'test',
                'top_k_anchors':3,'neighbor_window':2,
                'max_context_chunks':12,'comparison_source_diversity':True,
                'provider':provider['name'],'base_url':provider['base_url'],'model':provider['model'],
                'max_neighbor_chunks_per_page':1,
                'temperature':provider['temperature'],'thinking_disabled':provider['thinking_disabled'],
                'max_tokens':1200,'python':platform.python_version(),
                'generation_requested':args.generate,'generation_parallelism':len(systems),
                'systems':systems,
                'chunk_count':len(chunks),
                'materials':{n:d['chunk_count'] for n,d in materials.items()},
                'hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'evaluation/questions.jsonl', ROOT/'backend/parsing/parser.py', ROOT/'backend/vector_search/retriever.py', ROOT/'backend/graph_rag/pipeline.py', ROOT/'backend/knowledge_graph/triples.json', Path(__file__)]},
                'course_hashes':{p.name:sha(p) for p in (ROOT/'course_materials').iterdir() if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    rows = []
    for case in cases:
        anchors = retriever.search(case['question'],top_k=3)
        results = retriever.search_context(case['question'],top_k=3,neighbor_window=2)
        triples = get_kg_triples(
            case['question'],
            context_texts=[result.chunk.text for result in results],
        )
        gold = {(Path(e['source']).stem.replace(' ','_')[:50], page) for e in case['source_evidence'] for page in e['locators']}
        anchor_pages = {(r.chunk.source,r.chunk.page) for r in anchors}
        context_pages = {(r.chunk.source,r.chunk.page) for r in results}
        needed = {tuple(t[k] for k in ('head','relation','tail')) for t in case['required_relations']}
        found = {tuple(t[k] for k in ('head','relation','tail')) for t in triples}
        row = {'id':case['id'],'question':case['question'],'category':case['category'],
               'expected_behavior':case['expected_behavior'],
               'anchor_hit_at_3':bool(gold & anchor_pages) if gold else None,
               'context_hit':bool(gold & context_pages) if gold else None,
               'gold_evidence_recall':len(gold & context_pages)/len(gold) if gold else None,
               'required_relation_recall':len(needed & found)/len(needed) if needed else None,
               'anchors':[r.model_dump() for r in anchors],
               'sources':[r.model_dump() for r in results],'triples':triples}
        rows.append(row)
        with (out/'retrieval.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        if args.generate:
            with ThreadPoolExecutor(max_workers=len(systems)) as pool:
                futures = [
                    pool.submit(generate_answer,case,system,results,triples,key,args.direct,provider)
                    for system in systems
                ]
                for system, future in zip(systems, futures):
                    record = future.result()
                    with (out/'answers.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    print(case['id'],system,'completed',flush=True)
    summary = summarize_retrieval(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
