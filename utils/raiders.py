"""Read Raiders threads and preserve text, explicit quote links and provenance."""
from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import ftfy
import pandas as pd


class PostHTMLParser(HTMLParser):
    """Extract readable text and quote targets before removing HTML markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.quote_targets = set()
        self.skip_quote_label = False
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {'script', 'style'}:
            self.hidden_depth += 1
        if tag == 'a':
            self.skip_quote_label = False
            if 'quotelink' in attrs.get('class', '').split():
                url = urlsplit(attrs.get('href', ''))
                match = re.fullmatch(r'p(\d+)', url.fragment)
                # External domains and cross-board references must not be
                # mistaken for local /pol/ posts that share the same number.
                host_ok = url.hostname in {None, 'boards.4chan.org', 'boards.4channel.org'}
                if match and host_ok:
                    parts = [part for part in url.path.split('/') if part]
                    board = 'pol' if not parts or parts[0] in {'thread', 'res'} else parts[0]
                    self.quote_targets.add((board, int(match.group(1))))
                    self.skip_quote_label = True
                    self.parts.append(' ')
        if tag in {'br', 'p', 'div', 'li'}:
            self.parts.append('\n')
        # A <wbr> splits the source HTML, not the underlying word/URL.

    def handle_endtag(self, tag):
        if tag == 'a':
            self.skip_quote_label = False
        if tag in {'script', 'style'}:
            self.hidden_depth = max(0, self.hidden_depth - 1)
        if tag in {'p', 'div', 'li'}:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.skip_quote_label and not self.hidden_depth:
            self.parts.append(data)


def parse_post_html(raw_html):
    parser = PostHTMLParser()
    parser.feed(raw_html or '')
    parser.close()
    text = ftfy.fix_encoding(''.join(parser.parts))
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text, sorted(parser.quote_targets)


def text_fingerprint(text):
    """Identify equal text after case/whitespace normalization, not near copies."""
    normalized = ' '.join(text.casefold().split())
    return sha256(normalized.encode('utf-8')).hexdigest() if normalized else None


def compile_terms(terms):
    return {
        term: re.compile(r'(?<!\w)' + r'\s+'.join(re.escape(word) for word in term.split()) + r'(?!\w)', re.I)
        for term in dict.fromkeys(terms)
    }


def matched_terms(text, patterns):
    return [term for term, pattern in patterns.items() if pattern.search(text)]


def iter_threads(path, max_threads=None):
    if max_threads is not None and max_threads < 1:
        raise ValueError('max_threads must be positive or None')
    with Path(path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if max_threads is not None and line_number > max_threads:
                break
            thread = json.loads(line)
            if not isinstance(thread, dict) or not isinstance(thread.get('posts'), list):
                raise ValueError(f'Expected a posts list on source line {line_number}')
            yield line_number, thread


def prepare_raiders(path, seed_terms, anchor_terms, max_threads=None, progress_every=1000):
    """Scan source threads; retain every post of each seed-matching thread."""
    seed_patterns = compile_terms(seed_terms)
    anchor_patterns = {name: compile_terms(variants) for name, variants in anchor_terms.items()}
    rows, edges = [], []
    totals = Counter()
    seed_counts = Counter()
    anchor_counts = Counter()
    for line_number, thread in iter_threads(path, max_threads):
        source_posts = thread['posts']
        totals['threads_scanned'] += 1
        if not source_posts:
            totals['empty_threads'] += 1
            continue
        openers = [post for post in source_posts if post.get('resto') == 0]
        if len(openers) != 1:
            raise ValueError(f'Expected one opening post on source line {line_number}')
        thread_id = int(openers[0]['no'])
        thread_rows, thread_edges = [], []
        for post in source_posts:
            post_id = int(post['no'])
            if post.get('resto') not in {0, thread_id}:
                raise ValueError(f'Inconsistent thread ID at post {post_id}')
            raw_html = post.get('com') or ''
            text, quotes = parse_post_html(raw_html)
            seeds = matched_terms(text, seed_patterns)
            anchors = [name for name, variants in anchor_patterns.items() if matched_terms(text, variants)]
            totals['posts_scanned'] += 1
            totals['posts_with_text_scanned'] += bool(text)
            seed_counts.update(seeds)
            anchor_counts.update(anchors)
            thread_rows.append({
                'post_id': post_id, 'thread_id': thread_id,
                'source_line': line_number, 'timestamp': post.get('time'),
                'is_op': post.get('resto') == 0,
                'subject_html': post.get('sub', ''), 'raw_html': raw_html,
                'text': text, 'text_hash': text_fingerprint(text),
                'seed_matches': seeds, 'anchor_matches': anchors,
                'quote_targets': [{'board': board, 'post_id': target} for board, target in quotes],
            })
            thread_edges.extend({'source_post_id': post_id, 'source_thread_id': thread_id,
                                 'target_board': board, 'target_post_id': target}
                                for board, target in quotes)
        if any(row['seed_matches'] for row in thread_rows):
            totals['threads_retained'] += 1
            rows.extend(thread_rows)
            edges.extend(thread_edges)
        if progress_every and totals['threads_scanned'] % progress_every == 0:
            print(f"Scanned {totals['threads_scanned']:,} threads / {totals['posts_scanned']:,} posts; retained {totals['threads_retained']:,} threads", flush=True)
    columns = ['post_id','thread_id','source_line','timestamp','is_op','subject_html',
               'raw_html','text','text_hash','seed_matches','anchor_matches','quote_targets']
    posts = pd.DataFrame(rows, columns=columns)
    if posts['post_id'].duplicated().any():
        raise ValueError('Duplicate post IDs found; inspect repeated source threads before splitting')
    edge_table = pd.DataFrame(edges, columns=['source_post_id','source_thread_id','target_board','target_post_id'])
    lookup = posts.set_index('post_id')['thread_id'].to_dict()
    edge_table['target_thread_id'] = [lookup.get(target) if board == 'pol' else None
                                    for board, target in zip(edge_table['target_board'], edge_table['target_post_id'])]
    edge_table['target_present'] = edge_table['target_thread_id'].notna()
    edge_table['same_thread'] = edge_table['target_present'] & edge_table['source_thread_id'].eq(edge_table['target_thread_id'])
    totals['posts_retained'] = len(posts)
    totals['explicit_quote_links'] = len(edge_table)
    return posts, edge_table, dict(totals), seed_counts, anchor_counts


def thread_role(thread_id, salt='raiders-v1', reference_fraction=0.2, development_fraction=0.2):
    """Stable assignment of complete threads, independent of input order."""
    if not (0 < reference_fraction < 1 and 0 < development_fraction < 1
            and reference_fraction + development_fraction < 1):
        raise ValueError('Each role must have a positive fraction')
    score = int(sha256(f'{salt}:{thread_id}'.encode()).hexdigest()[:16], 16) / 2**64
    if score < reference_fraction:
        return 'reference'
    if score < reference_fraction + development_fraction:
        return 'development'
    return 'evaluation'


def select_model_inputs(posts, discovery_scope='seed_posts'):
    """Keep contexts intact; exclude duplicate model inputs across roles."""
    if discovery_scope not in {'seed_posts', 'all_retrieved_posts'}:
        raise ValueError('Unsupported discovery_scope')
    usable = posts.loc[posts['text'].str.len().gt(0)].copy()
    all_discovery_hashes = set(usable.loc[usable['split'].ne('reference'), 'text_hash'])
    reference = usable.loc[usable['split'].eq('reference') & usable['anchor_matches'].map(bool)].copy()
    rejected_reference = reference.loc[reference['text_hash'].isin(all_discovery_hashes)].copy()
    reference = reference.loc[~reference['text_hash'].isin(all_discovery_hashes)].copy()
    reference = reference.sort_values('post_id').drop_duplicates('text_hash')
    candidates = usable.loc[usable['split'].ne('reference')].copy()
    if discovery_scope == 'seed_posts':
        candidates = candidates.loc[candidates['seed_matches'].map(bool)]
    # Evaluation targets take precedence. Development targets identical to any
    # evaluation context are excluded, including short copied replies.
    evaluation_hashes = set(usable.loc[usable['split'].eq('evaluation'), 'text_hash'])
    development = candidates.loc[candidates['split'].eq('development') & ~candidates['text_hash'].isin(evaluation_hashes)]
    evaluation = candidates.loc[candidates['split'].eq('evaluation')]
    development = development.sort_values('post_id').drop_duplicates('text_hash').copy()
    evaluation = evaluation.sort_values('post_id').drop_duplicates('text_hash').copy()
    for frame in (reference, development, evaluation):
        frame['Post Text'] = frame['text']
    return reference, development, evaluation, rejected_reference
