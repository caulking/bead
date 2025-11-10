#!/usr/bin/env python3
"""Generate 2AFC (two-alternative forced choice) pairs from cross-product items.

This script:
1. Loads cross-product items (verb × template combinations)
2. Fills templates using MixedFillingStrategy
3. Scores filled items with language model (REFACTORED: uses bead/items/scoring.py)
4. Creates forced-choice items (REFACTORED: uses bead/items/forced_choice.py)
5. Assigns quantiles (REFACTORED: uses bead/lists/stratification.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from bead.items.forced_choice import create_forced_choice_items_from_groups  # NEW
from bead.items.item import Item
from bead.items.scoring import LanguageModelScorer  # NEW
from bead.lists.stratification import assign_quantiles_by_uuid  # NEW
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template


def load_cross_product_items(path: str, limit: int | None = None) -> list[Item]:
    """Load cross-product items from JSONL."""
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line)
            items.append(Item(**data))
    return items


def load_templates(path: str) -> dict[str, Template]:
    """Load templates from JSONL and return dict keyed by ID."""
    templates = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            template = Template(**data)
            templates[str(template.id)] = template
    return templates


def load_lexicons(lexicon_dir: Path) -> dict[str, Lexicon]:
    """Load all lexicons from directory."""
    lexicons = {}

    # Map slot names to lexicon files
    slot_to_lexicon = {
        "subj": ("bleached_nouns.jsonl", "bleached_nouns"),
        "obj": ("bleached_nouns.jsonl", "bleached_nouns"),
        "noun": ("bleached_nouns.jsonl", "bleached_nouns"),
        "verb": ("verbnet_verbs.jsonl", "verbnet_verbs"),
        "adj": ("bleached_adjectives.jsonl", "bleached_adjectives"),
        "adjective": ("bleached_adjectives.jsonl", "bleached_adjectives"),
        "det": ("determiners.jsonl", "determiners"),
        "prep": ("prepositions.jsonl", "prepositions"),
    }

    # Load unique lexicon files
    loaded_files = {}
    for slot_name, (filename, lex_name) in slot_to_lexicon.items():
        if filename not in loaded_files:
            path = lexicon_dir / filename
            if path.exists():
                loaded_files[filename] = Lexicon.from_jsonl(str(path), name=lex_name)

        # Map slot name to loaded lexicon
        if filename in loaded_files:
            lexicons[slot_name] = loaded_files[filename]

    return lexicons


def fill_templates_with_mixed_strategy(
    items: list[Item],
    templates: dict[str, Template],
    lexicons: dict[str, Lexicon],
) -> dict[str, str]:
    """Fill templates and return mapping of item_id -> filled_text.

    For now, uses simple exhaustive strategy with first available entry.
    """
    filled_texts = {}

    for item in items:
        template_id = str(item.item_template_id)
        if template_id not in templates:
            print(f"Warning: Template {template_id} not found, skipping item {item.id}")
            continue

        template = templates[template_id]

        # Get verb from item metadata
        verb_lemma = item.item_metadata.get("verb_lemma")
        if not verb_lemma:
            print(f"Warning: No verb_lemma in item {item.id}, skipping")
            continue

        # Build slot values
        slot_values = {"verb": verb_lemma}

        # Fill other slots with first available lexicon entry
        for slot_name in template.slots.keys():
            if slot_name == "verb":
                continue  # Already set

            # Get lexicon for this slot
            if slot_name in lexicons:
                lexicon = lexicons[slot_name]
                if len(lexicon) > 0:
                    # Use first entry (items is a dict)
                    first_item = next(iter(lexicon.items.values()))
                    # Use form if available, otherwise use lemma
                    slot_values[slot_name] = first_item.form or first_item.lemma
            else:
                # Try to infer lexicon from slot name
                if slot_name in ["subj", "obj", "noun"]:
                    if "noun" in lexicons and len(lexicons["noun"]) > 0:
                        first_item = next(iter(lexicons["noun"].items.values()))
                        slot_values[slot_name] = first_item.form or first_item.lemma
                elif slot_name in ["adj", "adjective"]:
                    if "adjective" in lexicons and len(lexicons["adjective"]) > 0:
                        first_item = next(iter(lexicons["adjective"].items.values()))
                        slot_values[slot_name] = first_item.form or first_item.lemma

        # Fill template
        try:
            filled_text = template.template_string
            for slot_name, value in slot_values.items():
                filled_text = filled_text.replace(f"{{{slot_name}}}", value)

            # Check if any slots remain unfilled
            if "{" in filled_text and "}" in filled_text:
                print(
                    f"Warning: Unfilled slots remain in item {item.id}: {filled_text}"
                )
                # Skip items with unfilled slots
                continue

            filled_texts[str(item.id)] = filled_text
        except Exception as e:
            print(f"Warning: Error filling item {item.id}: {e}")
            continue

    return filled_texts


def score_filled_items_with_lm(
    items: list[Item],
    cache_dir: Path,
    model_name: str = "gpt2",
) -> dict[str, float]:
    """Score filled items with language model using bead/items/scoring.py.

    REFACTORED: Now uses LanguageModelScorer from bead.
    """
    print(f"  Loading model {model_name}...")

    # Use bead's LanguageModelScorer
    scorer = LanguageModelScorer(
        model_name=model_name,
        cache_dir=cache_dir,
        device="cpu",
        text_key="text",  # Will extract from rendered_elements["text"]
    )

    print(f"  Scoring {len(items)} items...")

    # Create temporary items with filled text in rendered_elements
    temp_items = []
    item_id_map = {}
    for i, item in enumerate(items):
        if i % 10 == 0:
            print(f"    Progress: {i}/{len(items)}")

        temp_item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": item.rendered_elements.get("text", "")},
        )
        temp_items.append(temp_item)
        item_id_map[temp_item.id] = str(item.id)

    # Score batch
    scores_list = scorer.score_batch(temp_items)

    # Map back to original item IDs
    scores = {}
    for temp_item, score in zip(temp_items, scores_list):
        original_id = item_id_map[temp_item.id]
        scores[original_id] = score

    print("  Scoring complete.")
    return scores


def create_forced_choice_pairs(
    items: list[Item],
    lm_scores: dict[str, float],
) -> list[Item]:
    """Create 2AFC items using bead/items/forced_choice.py.

    REFACTORED: Now uses create_forced_choice_items_from_groups from bead.
    Creates two types of forced-choice items:
    1. Same-verb pairs (same verb, different frames)
    2. Different-verb pairs (different verbs, same frame)
    """
    # Add scores to item metadata for use in metadata_fn
    for item in items:
        item.item_metadata["lm_score"] = lm_scores.get(str(item.id), float("-inf"))

    # Helper to extract text from items
    def extract_text(item: Item) -> str:
        return item.rendered_elements.get("text", "")

    # 1. Create same-verb pairs (group by verb_lemma)
    print("  Creating same-verb pairs...")

    same_verb_items = create_forced_choice_items_from_groups(
        items=items,
        group_by=lambda item: item.item_metadata.get("verb_lemma", "unknown"),
        n_alternatives=2,
        extract_text=extract_text,
        include_group_metadata=True,
    )

    # Add pair_type and additional metadata
    for fc_item in same_verb_items:
        # Extract source item IDs
        item1_id = fc_item.item_metadata.get("source_item_0_id")
        item2_id = fc_item.item_metadata.get("source_item_1_id")

        # Find original items to get additional metadata
        source_items = [i for i in items if str(i.id) in [item1_id, item2_id]]
        if len(source_items) == 2:
            fc_item.item_metadata.update({
                "pair_type": "same_verb",
                "verb": source_items[0].item_metadata.get("verb_lemma"),
                "template1": source_items[0].item_metadata.get("template_structure"),
                "template2": source_items[1].item_metadata.get("template_structure"),
                "lm_score_a": lm_scores.get(str(source_items[0].id), float("-inf")),
                "lm_score_b": lm_scores.get(str(source_items[1].id), float("-inf")),
                "lm_score_diff": abs(
                    lm_scores.get(str(source_items[0].id), 0) -
                    lm_scores.get(str(source_items[1].id), 0)
                ),
            })

    print(f"  ✓ Created {len(same_verb_items)} same-verb pairs")

    # 2. Create different-verb pairs (group by template_id)
    print("  Creating different-verb pairs...")

    different_verb_items = create_forced_choice_items_from_groups(
        items=items,
        group_by=lambda item: str(item.item_template_id),
        n_alternatives=2,
        extract_text=extract_text,
        include_group_metadata=True,
    )

    # Add pair_type and additional metadata
    for fc_item in different_verb_items:
        item1_id = fc_item.item_metadata.get("source_item_0_id")
        item2_id = fc_item.item_metadata.get("source_item_1_id")

        source_items = [i for i in items if str(i.id) in [item1_id, item2_id]]
        if len(source_items) == 2:
            fc_item.item_metadata.update({
                "pair_type": "different_verb",
                "template_id": str(source_items[0].item_template_id),
                "template_structure": source_items[0].item_metadata.get("template_structure"),
                "verb1": source_items[0].item_metadata.get("verb_lemma"),
                "verb2": source_items[1].item_metadata.get("verb_lemma"),
                "lm_score_a": lm_scores.get(str(source_items[0].id), float("-inf")),
                "lm_score_b": lm_scores.get(str(source_items[1].id), float("-inf")),
                "lm_score_diff": abs(
                    lm_scores.get(str(source_items[0].id), 0) -
                    lm_scores.get(str(source_items[1].id), 0)
                ),
            })

    print(f"  ✓ Created {len(different_verb_items)} different-verb pairs")

    return same_verb_items + different_verb_items


def assign_quantiles_to_pairs(
    pair_items: list[Item],
    n_quantiles: int = 10,
) -> list[Item]:
    """Assign quantile bins using bead/lists/stratification.py.

    REFACTORED: Now uses assign_quantiles_by_uuid from bead.
    Stratifies by pair_type so same-verb and different-verb pairs
    get separate quantile distributions.
    """
    print(f"  Assigning quantiles (stratified by pair_type)...")

    # Build metadata dict for quantile assignment
    item_metadata = {
        item.id: item.item_metadata
        for item in pair_items
    }

    # Get item IDs
    item_ids = [item.id for item in pair_items]

    # Assign quantiles stratified by pair_type
    quantile_assignments = assign_quantiles_by_uuid(
        item_ids=item_ids,
        item_metadata=item_metadata,
        property_key="lm_score_diff",
        n_quantiles=n_quantiles,
        stratify_by_key="pair_type",  # Separate quantiles for same_verb vs different_verb
    )

    # Add quantile to each item's metadata
    for item in pair_items:
        item.item_metadata["quantile"] = quantile_assignments[item.id]

    print(f"  ✓ Assigned quantiles to {len(pair_items)} pairs")
    return pair_items


def main(item_limit: int | None = None) -> None:
    """Generate 2AFC pairs from cross-product.

    Parameters
    ----------
    item_limit : int | None
        Limit number of cross-product items to process.
        If None, process all items.
    """
    print("=" * 60)
    print("2AFC PAIR GENERATION")
    print("=" * 60)

    # Paths
    base_dir = Path(__file__).parent
    items_path = base_dir / "items" / "cross_product_items.jsonl"
    templates_path = base_dir / "templates" / "generic_frames.jsonl"
    lexicons_dir = base_dir / "lexicons"
    output_path = base_dir / "items" / "2afc_pairs.jsonl"

    print("\n[1/6] Loading cross-product items...")
    items = load_cross_product_items(str(items_path), limit=item_limit)
    print(f"✓ Loaded {len(items)} cross-product items")

    print("\n[2/6] Loading templates...")
    templates = load_templates(str(templates_path))
    print(f"✓ Loaded {len(templates)} templates")

    print("\n[3/6] Loading lexicons...")
    lexicons = load_lexicons(lexicons_dir)
    print(f"✓ Loaded {len(lexicons)} lexicons")
    for name, lexicon in lexicons.items():
        print(f"  - {name}: {len(lexicon)} entries")

    print("\n[4/6] Filling templates...")
    filled_texts = fill_templates_with_mixed_strategy(items, templates, lexicons)
    print(f"✓ Filled {len(filled_texts)} items")

    if not filled_texts:
        print("Error: No items were successfully filled. Exiting.")
        return

    # Create Items with filled text in rendered_elements
    filled_items = []
    for item in items:
        if str(item.id) in filled_texts:
            item.rendered_elements["text"] = filled_texts[str(item.id)]
            filled_items.append(item)

    # Show examples
    print("\nExample filled texts:")
    for i, item in enumerate(filled_items[:3]):
        print(f"  {i + 1}. {item.rendered_elements['text']}")

    print(f"\n[5/6] Scoring with language model (REFACTORED: uses bead/items/scoring.py)...")
    cache_dir = base_dir / ".cache"
    lm_scores = score_filled_items_with_lm(
        filled_items, cache_dir=cache_dir, model_name="gpt2"
    )
    print(f"✓ Scored {len(lm_scores)} items")

    print(f"\n[6/6] Creating forced-choice pairs (REFACTORED: uses bead/items/forced_choice.py)...")
    pair_items = create_forced_choice_pairs(filled_items, lm_scores)

    if not pair_items:
        print("Error: No pairs were created. Exiting.")
        return

    print(f"\n[7/7] Assigning quantiles (REFACTORED: uses bead/lists/stratification.py)...")
    pair_items = assign_quantiles_to_pairs(pair_items, n_quantiles=10)

    # Save
    print(f"\n[8/8] Saving to {output_path}...")
    with open(output_path, "w") as f:
        for item in pair_items:
            f.write(item.model_dump_json() + "\n")

    print(f"✓ Saved {len(pair_items)} 2AFC pairs")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    same_verb_count = sum(
        1 for item in pair_items if item.item_metadata.get("pair_type") == "same_verb"
    )
    different_verb_count = sum(
        1 for item in pair_items if item.item_metadata.get("pair_type") == "different_verb"
    )
    print(f"  - Same-verb pairs: {same_verb_count}")
    print(f"  - Different-verb pairs: {different_verb_count}")
    print(f"  - Total pairs: {len(pair_items)}")
    print(f"  - Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 2AFC pairs from cross-product items"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cross-product items to process (default: all)",
    )
    args = parser.parse_args()

    main(item_limit=args.limit)
