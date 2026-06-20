"""The bridge channel — epoch-versioned store of validated exports.

Within an epoch the schema is frozen: every submitted artifact is validated
against the current `BridgeSchema` and either accepted (stored for the parent to
read) or rejected (logged). Between epochs the schema may change only through the
governance path (architecture C). The parent's view is artifacts only — never
logs, never ground truth.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum

from .schema import BridgeArtifact, BridgeSchema, validate


class Architecture(str, Enum):
    TYPED_STATIC = "A"     # frozen schema, no governance
    FREE_FORM = "B"        # length-capped untyped text
    TYPED_GOVERNED = "C"   # evolving schema via governance


class Bridge:
    def __init__(self, schema: BridgeSchema, architecture: Architecture):
        self.schema = schema
        self.architecture = architecture
        self.epoch = 0
        self.artifacts_by_epoch: dict[int, list[BridgeArtifact]] = defaultdict(list)
        self.rejected: list[dict] = []
        # version history: (epoch, schema_version, schema)
        self.schema_history: list[tuple] = [(0, schema.version, schema)]

    # -- submission (within the frozen epoch) ------------------------------
    def submit(self, artifact: BridgeArtifact) -> tuple[bool, list]:
        ok, errors = validate(artifact, self.schema)
        if ok:
            self.artifacts_by_epoch[self.epoch].append(artifact)
        else:
            self.rejected.append({
                "epoch": self.epoch,
                "artifact_id": artifact.artifact_id,
                "errors": errors,
            })
        return ok, errors

    # -- the parent's view (artifacts only) --------------------------------
    def parent_view(self, epoch: int | None = None) -> list[dict]:
        e = self.epoch if epoch is None else epoch
        return [a.parent_view(self.schema) for a in self.artifacts_by_epoch.get(e, [])]

    def artifacts(self, epoch: int | None = None) -> list[BridgeArtifact]:
        e = self.epoch if epoch is None else epoch
        return list(self.artifacts_by_epoch.get(e, []))

    # -- epoch / schema lifecycle ------------------------------------------
    def advance_epoch(self) -> None:
        self.epoch += 1

    def commit_schema(self, new_schema: BridgeSchema) -> None:
        """Land a governed schema change (called by the governance layer only)."""
        self.schema = new_schema
        self.schema_history.append((self.epoch, new_schema.version, new_schema))

    def schema_version_at(self, epoch: int) -> str:
        version = self.schema_history[0][1]
        for e, v, _ in self.schema_history:
            if e <= epoch:
                version = v
        return version
