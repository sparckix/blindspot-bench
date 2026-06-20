"""The parent overseer — a channel-limited LLM that perceives the inner world
only through the bridge.

It answers the probe battery, rates the inner world's exports, and reviews
channel-modification proposals. It never sees the logs, the battery scores, the
oversight gap, or lambda. The decisive design property lives here: the parent's
*answers* track artifact truthfulness (so a false claim raises the gap) while the
parent's *rating* tracks only observable quality signals (so a well-dressed false
claim can still raise the rating). That decoupling is what makes a rating-up /
gap-up gaming event possible.
"""

from .parent import ParentOverseer

__all__ = ["ParentOverseer"]
