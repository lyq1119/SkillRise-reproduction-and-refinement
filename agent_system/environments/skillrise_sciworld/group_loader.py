import json
import random
from typing import List, Dict


class GroupLoader:
    """Loads cross-task meta-RL groups and serves them in fixed-size batches.

    Each line of the jsonl is one group of K related tasks (same task_type/family).
    A training step consumes `batch_groups` groups; every group is then replayed by
    N parallel trials (handled by the env manager), so the env worker count must
    equal `batch_groups * N`.

    ScienceWorld task identity is a `[task_id, variation]` integer pair (loaded via
    env.load(taskNames[task_id], variation) then reset). This is the analogue of
    ALFWorld's gamefile path; here we expose it per group as `pairs`.
    """

    def __init__(self, group_file: str, batch_groups: int, seed: int = 0):
        self.group_file = group_file
        self.batch_groups = batch_groups
        self.groups: List[Dict] = self._load(group_file)
        assert len(self.groups) > 0, f"no groups found in {group_file}"
        # One-time shuffle with a fixed seed, then serve in cursor order. The group
        # file is laid out in contiguous task_type blocks; shuffling makes every
        # batch a random mix of families at their natural proportions.
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
                g["pairs"] = [[int(t["task_id"]), int(t["variation"])] for t in g["tasks"]]
                groups.append(g)
        return groups

    def next_batch(self) -> List[Dict]:
        """Return the next `batch_groups` groups, cycling through the dataset."""
        batch = []
        for _ in range(self.batch_groups):
            batch.append(self.groups[self._cursor % len(self.groups)])
            self._cursor += 1
        return batch
