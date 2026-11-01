"""Config merge utility — FR4: deep + named-merge + ordering.

Public API:
    merge_configs(parent, child) -> dict
"""
from __future__ import annotations

import copy
from typing import Any


def merge_configs(parent: dict, child: dict) -> dict:
    """Merge child config on top of parent following SR2 config merge rules.

    Rules (applied uniformly and recursively):
    1. Scalar values: child wins.
    2. Dict/map values: deep merge recursively.
    3. Lists of objects where ALL elements have a 'name:' field: merged by
       name key. Matched entries are deep-merged in-place at parent's
       position. Parent entries with no child match are kept. Child-only
       entries are appended after all parent entries in child declaration order.
    4. All other lists: child replaces entirely.

    Inputs are not mutated. Returns a new dict.
    """
    return _merge(copy.deepcopy(parent), copy.deepcopy(child))


def _merge(parent: Any, child: Any) -> Any:
    """Recursive merge — operates on deep copies so inputs are never mutated."""
    if isinstance(parent, dict) and isinstance(child, dict):
        return _merge_dicts(parent, child)
    if isinstance(parent, list) and isinstance(child, list):
        return _merge_lists(parent, child)
    # Scalar (or type mismatch): child wins
    return child


def _merge_dicts(parent: dict, child: dict) -> dict:
    result = dict(parent)
    for key, child_val in child.items():
        if key in result:
            result[key] = _merge(result[key], child_val)
        else:
            result[key] = child_val
    return result


def _is_named_list(lst: list) -> bool:
    """Return True iff lst is non-empty and every element is a dict with 'name'."""
    if not lst:
        return False
    return all(isinstance(item, dict) and "name" in item for item in lst)


def _merge_lists(parent: list, child: list) -> list:
    """Merge two lists.

    - If parent is a named list (all elements are dicts with 'name:'): apply
      named-merge rules — deep-merge matched entries in-place, keep parent
      unmatched entries, append child-only entries in child order.
    - If parent is empty but child is non-empty and child is a named list,
      treat as named-merge (child entries become result).
    - Otherwise: child replaces entirely.
    """
    # Determine named-merge eligibility from the non-empty list
    # (parent takes precedence; fall back to child when parent is empty)
    if _is_named_list(parent):
        return _named_merge(parent, child)
    if not parent and _is_named_list(child):
        # Parent is empty; child named list — all child entries are "new"
        return _named_merge(parent, child)
    # Plain list: child replaces
    return child


def _named_merge(parent: list, child: list) -> list:
    """Named-merge: merge by 'name' key.

    Ordering:
    - Parent entries retain their position (matched or unmatched).
    - Child-only (new name) entries append after all parent entries, in
      child declaration order.
    """
    # Build index of child entries by name
    child_by_name: dict[str, dict] = {}
    child_new: list[dict] = []  # child entries with names not in parent
    parent_names: set[str] = {item["name"] for item in parent}

    for item in child:
        name = item["name"]
        if name in parent_names:
            child_by_name[name] = item
        else:
            child_new.append(item)

    # Merge parent entries in-place
    result: list[dict] = []
    for parent_item in parent:
        name = parent_item["name"]
        if name in child_by_name:
            merged = _merge(parent_item, child_by_name[name])
            result.append(merged)
        else:
            result.append(parent_item)

    # Append child-only entries in child declaration order
    result.extend(child_new)
    return result
