import os
import json
import random
from typing import List, Dict


def _split_root(split: str = "train") -> str:
    data = os.environ.get("ALFWORLD_DATA", "")
    return os.path.join(os.path.expandvars(data), "json_2.1.1", split)


def _train_root() -> str:
    return _split_root("train")


def resolve_gamefile(task_dir: str, trial: str, split: str = "train") -> str:
    """Map a group entry (task_dir + trial) to its TextWorld game file path.

    `split` selects the json_2.1.1 subdir (train / valid_seen / valid_unseen);
    val eval groups live under valid_seen.
    """
    path = os.path.join(_split_root(split), task_dir, trial, "game.tw-pddl")
    return path


class GroupLoader:
    """Loads cross-task meta-RL groups and serves them in fixed-size batches.

    Each line of the jsonl is one group of K related tasks (same task_type).
    A training step consumes `batch_groups` groups; every group is then replayed
    by N parallel trials (handled by the env manager), so the env worker count
    must equal `batch_groups * N`.
    """

    def __init__(self, group_file: str, batch_groups: int, seed: int = 0,
                 split: str = "train"):
        self.group_file = group_file
        self.batch_groups = batch_groups
        self.split = split
        self.groups: List[Dict] = self._load(group_file)
        assert len(self.groups) > 0, f"no groups found in {group_file}"
        # Shuffle the whole pool once with a fixed seed, then serve in cursor order.
        # The group file is laid out in contiguous task_type blocks, so an unshuffled
        # cursor would make every batch single-type. A one-time shuffle matches the
        # LaMer/GiGPO baseline, where each env worker draws from a mixed pool of all
        # task types (textworld's rng.shuffle + shuffled_cycle), so every batch is a
        # random mix of all types at their natural proportions.
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
                g["gamefiles"] = [
                    resolve_gamefile(t["task_dir"], t["trial"], split=self.split)
                    for t in g["tasks"]
                ]
                groups.append(g)
        return groups

    def next_batch(self) -> List[Dict]:
        """Return the next `batch_groups` groups, cycling through the dataset."""
        batch = []
        for _ in range(self.batch_groups):
            batch.append(self.groups[self._cursor % len(self.groups)])
            self._cursor += 1
        return batch
