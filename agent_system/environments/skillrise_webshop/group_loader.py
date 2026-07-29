import json
import random
from typing import List, Dict


class GroupLoader:
    """Loads cross-task meta-RL groups and serves them in fixed-size batches.

    Each line of the jsonl is one group of K related tasks (same task_type/category).
    A training step consumes `batch_groups` groups; every group is then replayed by N
    parallel trials (handled by the env manager), so the env worker count must equal
    `batch_groups * N`.

    WebShop task identity is `session_idx` (loaded via env.reset(session=idx)); this is
    the analogue of ALFWorld's gamefile path. The group file was generated with seed=0,
    so training must use env.seed=0 for the session_idx identities to line up.
    """

    def __init__(self, group_file: str, batch_groups: int, seed: int = 0):
        self.group_file = group_file
        self.batch_groups = batch_groups
        self.groups: List[Dict] = self._load(group_file)
        assert len(self.groups) > 0, f"no groups found in {group_file}"
        # One-time shuffle with a fixed seed, then serve in cursor order. The group file
        # is laid out in contiguous task_type blocks; shuffling makes every batch a
        # random mix of categories at their natural proportions (matches the baseline).
        random.Random(seed).shuffle(self.groups)
        self.K = self.groups[0]["K"]
        self._cursor = 0

    def _load(self, path: str) -> List[Dict]:
        groups = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                g["session_idxs"] = [int(t["session_idx"]) for t in g["tasks"]]
                groups.append(g)
        return groups

    def next_batch(self) -> List[Dict]:
        """Return the next `batch_groups` groups, cycling through the dataset."""
        batch = []
        for _ in range(self.batch_groups):
            batch.append(self.groups[self._cursor % len(self.groups)])
            self._cursor += 1
        return batch
