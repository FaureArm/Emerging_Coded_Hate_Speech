import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


EMBEDDING_PROCEDURE = "target-subword-mean-v1"


def load_embedding_model(configuration, requested_device="auto"):
    """Load once using the frozen bank's exact model/tokenizer revision and settings."""
    from transformers import AutoModel, AutoTokenizer

    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda.")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available()
                          else "cpu" if requested_device == "auto" else requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use the Colab T4 runtime or --device cpu.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(configuration["tokenizer_checkpoint"],
                                                  revision=configuration["tokenizer_revision"], use_fast=True)
        model = AutoModel.from_pretrained(configuration["checkpoint"],
                                         revision=configuration["resolved_revision"]).to(device)
    except OSError as error:
        raise RuntimeError("Cannot load the bank's pinned BERT revision. Provide network access or its local Hugging Face cache.") from error
    model.eval()
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required.")
    if (model.config.hidden_size != configuration["hidden_size"]
            or model.config.max_position_embeddings != configuration["max_position_embeddings"]
            or str(next(model.parameters()).dtype) != configuration["model_dtype"]):
        raise ValueError("Loaded BERT architecture/dtype differs from the reference bank.")
    return model, tokenizer, device


def embed_anchor_occurrence(text, start, end, model, tokenizer, device, context_tokens):
    """Return one contextual vector for the character span text[start:end]."""
    # Locate the anchor in BERT's subword tokens, keeping original text offsets.
    tokens = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )
    positions = [
        i
        for i, (left, right) in enumerate(tokens["offset_mapping"])
        if right > left and right > start and left < end
    ]
    if not positions:
        raise ValueError(f"No BERT tokens overlap the anchor span {(start, end)}.")

    # Select surrounding context without losing anchors near the end of a post.
    first = max(0, positions[0] - context_tokens)
    last = min(len(tokens["input_ids"]) - 1, positions[-1] + context_tokens)
    window_start = tokens["offset_mapping"][first][0]
    window_end = tokens["offset_mapping"][last][1]
    window_text = text[window_start:window_end]

    inputs = tokenizer(
        window_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = inputs.pop("offset_mapping")[0]
    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        raise ValueError("Context exceeds BERT's limit. Reduce context_tokens.")

    # Translate the anchor's character positions into positions in this window.
    relative_start = start - window_start
    relative_end = end - window_start
    anchor_mask = (
        (offsets[:, 1] > offsets[:, 0])  # Exclude special tokens such as [CLS].
        & (offsets[:, 1] > relative_start)
        & (offsets[:, 0] < relative_end)
    )
    if not anchor_mask.any():
        raise ValueError("The anchor is missing from its context window.")

    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        token_vectors = model(**inputs).last_hidden_state[0]
        # Average only the anchor's subwords, not every token in the window.
        vector = token_vectors[anchor_mask.to(device)].mean(dim=0)
    return vector.cpu().numpy()


def embed_anchors(anchor_occurrences, embedding_posts, model, tokenizer, device, context_tokens):
    # Each matrix row corresponds to the same row in this metadata table.
    anchor_occurrence_metadata = (
        anchor_occurrences.drop_duplicates(["post_id", "anchor", "start", "end"])
        .reset_index(drop=True)
        .copy()
    )
    occurrence_vectors = []
    for occurrence in anchor_occurrence_metadata.itertuples(index=False):
        text = embedding_posts.loc[occurrence.post_id, "Text for BERT"]
        if text[occurrence.start : occurrence.end] != occurrence.matched_text:
            raise ValueError(
                "Text changed after matching. Rerun anchor matching first."
            )
        occurrence_vectors.append(
            embed_anchor_occurrence(text, occurrence.start, occurrence.end,
                                    model, tokenizer, device, context_tokens)
        )
    anchor_occurrence_embeddings = np.stack(occurrence_vectors)

    # HMApp averages repeated occurrences of an anchor within the same post.
    # Keep different posts separate: one reference vector per (anchor, post_id).
    reference_vectors = []
    reference_records = []
    for (anchor, post_id), group in anchor_occurrence_metadata.groupby(
        ["anchor", "post_id"], sort=False
    ):
        reference_vectors.append(anchor_occurrence_embeddings[group.index].mean(axis=0))
        reference_records.append(
            {
                "anchor": anchor,
                "post_id": post_id,
                "n_occurrences": len(group),
            }
        )

    anchor_embeddings = np.stack(reference_vectors)
    anchor_metadata = pd.DataFrame(reference_records)
    # Historical pilot note; the new caller supplies the independent reference corpus.
    # These references currently come from the same corpus as the candidates.
    # Reserve reference posts separately from discovery posts before evaluation.
    anchor_reference_post_ids = set(anchor_metadata["post_id"])

    return (
        anchor_occurrence_metadata,
        occurrence_vectors,
        anchor_occurrence_embeddings,
        reference_vectors,
        reference_records,
        anchor_embeddings,
        anchor_metadata,
        anchor_reference_post_ids,
    )


def prepare_context_windows(post_text, post_occurrences, tokenizer, context_tokens):
    """Find the context window surrounding each candidate occurrence."""
    encoded_post = tokenizer(
        post_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )

    offsets = np.asarray(encoded_post["offset_mapping"])
    windows = {}

    for occurrence_id, occurrence in post_occurrences.iterrows():
        start = occurrence["start"]
        end = occurrence["end"]

        if post_text[start:end] != occurrence["matched_text"]:
            raise ValueError("Text changed. Rerun candidate occurrence matching.")

        # Find BERT tokens overlapping the candidate's original passage.
        overlaps_candidate = (offsets[:, 1] > start) & (offsets[:, 0] < end)
        real_tokens = offsets[:, 1] > offsets[:, 0]

        token_indices = np.flatnonzero(overlaps_candidate & real_tokens)

        if len(token_indices) == 0:
            raise ValueError("No BERT tokens found for a candidate occurrence.")

        # Add the same amount of surrounding context used for anchors.
        first_token = max(0, token_indices[0] - context_tokens)
        last_token = min(
            len(offsets) - 1,
            token_indices[-1] + context_tokens,
        )

        window_start = int(offsets[first_token, 0])
        window_end = int(offsets[last_token, 1])
        window_position = (window_start, window_end)

        # Candidates sharing a window can share one BERT calculation.
        if window_position not in windows:
            windows[window_position] = {
                "start": window_start,
                "end": window_end,
                "occurrences": [],
            }

        windows[window_position]["occurrences"].append(
            (occurrence_id, occurrence["word_spans"])
        )

    return list(windows.values())


def average_candidate_tokens(
    token_vectors,
    token_offsets,
    word_spans,
    window_start,
):
    """Average the candidate's subword vectors within one context window."""
    selected_tokens = torch.zeros(len(token_offsets), dtype=torch.bool)

    for word_start, word_end in word_spans:
        # Convert original post positions into positions within this window.
        relative_start = word_start - window_start
        relative_end = word_end - window_start

        overlaps_word = (token_offsets[:, 1] > relative_start) & (
            token_offsets[:, 0] < relative_end
        )

        selected_tokens = selected_tokens | overlaps_word

    # Exclude special tokens and padding.
    real_tokens = token_offsets[:, 1] > token_offsets[:, 0]
    selected_tokens = selected_tokens & real_tokens

    if not selected_tokens.any():
        raise ValueError("Candidate words are missing from their context window.")

    selected_tokens = selected_tokens.to(token_vectors.device)
    vector = token_vectors[selected_tokens].mean(dim=0)

    return vector.cpu().numpy()


def embed_window_batch(post_text, window_batch, model, tokenizer, device):
    """Run BERT on several windows and extract their candidate vectors."""
    passages = []

    for window in window_batch:
        passages.append(post_text[window["start"] : window["end"]])

    inputs = tokenizer(
        passages,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offsets = inputs.pop("offset_mapping")

    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        raise ValueError("Context is too long. Reduce context_tokens.")

    for name in inputs:
        inputs[name] = inputs[name].to(device)

    occurrence_vectors = {}

    with torch.inference_mode():
        outputs = model(**inputs)
        batch_token_vectors = outputs.last_hidden_state

        for window_index, window in enumerate(window_batch):
            for occurrence_id, word_spans in window["occurrences"]:
                occurrence_vectors[occurrence_id] = average_candidate_tokens(
                    batch_token_vectors[window_index],
                    offsets[window_index],
                    word_spans,
                    window["start"],
                )

    return occurrence_vectors


def embed_candidates(candidate_occurrences, embedding_posts, candidate_batch_size,
                     model, tokenizer, device, context_tokens):
    if type(candidate_batch_size) is not int or candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be a positive integer.")
    # 1. Prepare one metadata row and one vector slot per occurrence.
    candidate_occurrence_metadata = candidate_occurrences.drop_duplicates(
        ["post_id", "candidate", "word_spans"]
    ).reset_index(drop=True)

    if candidate_occurrence_metadata.empty:
        raise ValueError("Run candidate occurrence matching first.")

    candidate_occurrence_embeddings = np.empty(
        (len(candidate_occurrence_metadata), model.config.hidden_size),
        dtype=np.float32,
    )

    # 2. Calculate contextual vectors, processing each post's windows in batches.
    posts = candidate_occurrence_metadata.groupby("post_id", sort=False)

    for post_id, post_occurrences in tqdm(posts, desc="Embedding candidate posts"):
        post_text = embedding_posts.loc[post_id, "Text for BERT"]
        windows = prepare_context_windows(post_text, post_occurrences, tokenizer, context_tokens)

        for batch_start in range(0, len(windows), candidate_batch_size):
            window_batch = windows[batch_start : batch_start + candidate_batch_size]
            batch_vectors = embed_window_batch(post_text, window_batch, model, tokenizer, device)

            for occurrence_id, vector in batch_vectors.items():
                candidate_occurrence_embeddings[occurrence_id] = vector

    # 3. Average repeated occurrences of the same candidate within each post.
    candidate_post_vectors = []
    candidate_post_records = []

    groups = candidate_occurrence_metadata.groupby(
        ["candidate", "post_id"],
        sort=False,
    )

    for (candidate, post_id), occurrences_in_post in groups:
        vectors = candidate_occurrence_embeddings[occurrences_in_post.index]

        candidate_post_vectors.append(vectors.mean(axis=0))
        candidate_post_records.append(
            {
                "candidate": candidate,
                "post_id": post_id,
                "n_occurrences": len(occurrences_in_post),
            }
        )

    candidate_embeddings = np.stack(candidate_post_vectors)
    candidate_metadata = pd.DataFrame(candidate_post_records)

    return (
        candidate_occurrence_metadata,
        candidate_occurrence_embeddings,
        candidate_post_vectors,
        candidate_post_records,
        candidate_embeddings,
        candidate_metadata,
    )
