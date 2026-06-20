"""Bridge artifact schema (Appendix B) and the versioned, governed schema object.

`BridgeArtifact` is the typed export an inner agent emits. `BridgeSchema` is the
contract it is validated against — frozen within an epoch, and (architecture C)
the mutable object the governance path revises between epochs. Everything here is
deterministic: artifact ids are content-derived (uuid5), never random, so a run
is reproducible byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field, replace

# A fixed namespace so artifact ids are a pure function of content.
_NS = uuid.UUID("6f1c0e2a-4b7d-5e8f-9a0b-1c2d3e4f5061")

ARTIFACT_TYPES = (
    "anomaly_report",
    "regularity_claim",
    "bottleneck_summary",
    "strategy_postmortem",
    "environment_change_request",
    "bridge_modification_proposal",
)
CONFIDENCE_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class BridgeSchema:
    """The typed contract for bridge exports. Versioned with semver.

    Architecture A pins this and never changes it; architecture C evolves it
    through governance; architecture B uses the degenerate free-form schema below.
    """

    version: str = "0.1.0"
    allowed_types: tuple = ARTIFACT_TYPES
    claim_max_len: int = 280
    allowed_confidence: tuple = CONFIDENCE_LEVELS
    require_evidence: bool = True          # evidence_traces must be non-empty
    allow_self_confidence: bool = False    # whether a self-assessed confidence is surfaced to parent
    free_form: bool = False                # architecture B: untyped capped text

    @staticmethod
    def free_form_schema(max_len: int = 600) -> "BridgeSchema":
        return BridgeSchema(
            version="0.1.0-freeform",
            allowed_types=("free_form",),
            claim_max_len=max_len,
            require_evidence=False,
            free_form=True,
        )

    def bumped(self, level: str = "minor") -> str:
        major, minor, patch = (self.version.split("-")[0].split(".") + ["0", "0"])[:3]
        major, minor, patch = int(major), int(minor), int(patch)
        if level == "major":
            major, minor, patch = major + 1, 0, 0
        elif level == "patch":
            patch += 1
        else:
            minor += 1
        return f"{major}.{minor}.{patch}"


@dataclass
class BridgeArtifact:
    type: str
    claim: str
    emitting_agents: list
    step_range: tuple
    confidence: str = "medium"
    evidence_traces: list = field(default_factory=list)
    prior_artifact_refs: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)      # structured claim data (e.g. {"key","value"})
    schema_version: str = "0.1.0"
    artifact_id: str = ""
    content_hash: str = ""

    def _canonical(self) -> str:
        body = {
            "type": self.type,
            "claim": self.claim,
            "evidence_traces": list(self.evidence_traces),
            "confidence": self.confidence,
            "emitting_agents": list(self.emitting_agents),
            "step_range": list(self.step_range),
            "payload": self.payload,
            "schema_version": self.schema_version,
            "prior_artifact_refs": list(self.prior_artifact_refs),
        }
        return json.dumps(body, sort_keys=True)

    def finalize(self) -> "BridgeArtifact":
        """Stamp the deterministic content hash and id."""
        canonical = self._canonical()
        self.content_hash = hashlib.sha256(canonical.encode()).hexdigest()
        self.artifact_id = str(uuid.uuid5(_NS, self.content_hash))
        return self

    def parent_view(self, schema: BridgeSchema) -> dict:
        """What the parent is shown. A self-assessed confidence is only surfaced
        when the schema permits it (governance can add this — a known capture
        vector, since parents over-weight it)."""
        view = {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "claim": self.claim,
            "evidence_traces": list(self.evidence_traces),
            "emitting_agents": list(self.emitting_agents),
            "step_range": list(self.step_range),
            "payload": self.payload,
            "schema_version": self.schema_version,
            "prior_artifact_refs": list(self.prior_artifact_refs),
        }
        if schema.allow_self_confidence:
            view["confidence"] = self.confidence
        return view

    def as_dict(self) -> dict:
        return asdict(self)


def make_artifact(schema: BridgeSchema, **kwargs) -> BridgeArtifact:
    """Construct and finalize an artifact stamped with the current schema version."""
    kwargs.setdefault("schema_version", schema.version)
    art = BridgeArtifact(**kwargs)
    return art.finalize()


def validate(artifact: BridgeArtifact, schema: BridgeSchema) -> tuple[bool, list]:
    """Validate an artifact against a (frozen) schema. Returns (ok, errors)."""
    errors: list = []
    if artifact.type not in schema.allowed_types:
        errors.append(f"type {artifact.type!r} not in allowed {schema.allowed_types}")
    if len(artifact.claim) > schema.claim_max_len:
        errors.append(f"claim length {len(artifact.claim)} > cap {schema.claim_max_len}")
    if not schema.free_form:
        if artifact.confidence not in schema.allowed_confidence:
            errors.append(f"confidence {artifact.confidence!r} not allowed")
        if schema.require_evidence and not artifact.evidence_traces:
            errors.append("evidence_traces required but empty")
    if artifact.schema_version != schema.version:
        errors.append(f"schema_version {artifact.schema_version} != channel {schema.version}")
    # Recompute the hash to guarantee integrity.
    expected = hashlib.sha256(artifact._canonical().encode()).hexdigest()
    if artifact.content_hash != expected:
        errors.append("content_hash mismatch")
    return (not errors, errors)
