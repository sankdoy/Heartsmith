from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Dict, Iterable, Literal


GroupName = Literal["global", "passing", "play"]


@dataclass
class Parameter:
    name: str
    group: GroupName
    default: float
    value: float
    min_value: float
    max_value: float
    locked: bool
    doc: Dict[str, str] = field(default_factory=dict)

    def clamp(self) -> None:
        self.value = max(self.min_value, min(self.max_value, self.value))


class ParameterSet:
    def __init__(self, params: Iterable[Parameter] | None = None) -> None:
        self._params: Dict[str, Parameter] = {}
        if params is None:
            params = _default_params()
        for param in params:
            self._params[param.name] = param

    def __getitem__(self, name: str) -> float:
        return self._params[name].value

    def get_param(self, name: str) -> Parameter:
        return self._params[name]

    def all(self) -> list[Parameter]:
        return list(self._params.values())

    def update(self, name: str, value: float) -> None:
        if name not in self._params:
            return
        param = self._params[name]
        param.value = float(value)
        param.clamp()

    def set_locked(self, name: str, locked: bool) -> None:
        if name not in self._params:
            return
        self._params[name].locked = locked

    def clamp(self) -> None:
        for param in self._params.values():
            param.clamp()

    def copy(self) -> ParameterSet:
        return ParameterSet(params=[Parameter(**{
            "name": p.name,
            "group": p.group,
            "default": p.default,
            "value": p.value,
            "min_value": p.min_value,
            "max_value": p.max_value,
            "locked": p.locked,
            "doc": dict(p.doc),
        }) for p in self._params.values()])

    def apply(self, other: ParameterSet) -> None:
        for name, param in other._params.items():
            if name not in self._params:
                continue
            self._params[name].value = param.value
            self._params[name].locked = param.locked

    def to_json(self) -> str:
        data = {
            name: {
                "group": param.group,
                "default": param.default,
                "value": param.value,
                "min": param.min_value,
                "max": param.max_value,
                "locked": param.locked,
                "doc": param.doc,
            }
            for name, param in self._params.items()
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> ParameterSet:
        raw = json.loads(payload)
        params = []
        for name, data in raw.items():
            params.append(
                Parameter(
                    name=name,
                    group=data["group"],
                    default=data["default"],
                    value=data["value"],
                    min_value=data["min"],
                    max_value=data["max"],
                    locked=data.get("locked", False),
                    doc=data.get("doc", {}),
                )
            )
        return cls(params)

    def to_table_rows(self) -> list[dict]:
        rows = []
        for param in self._params.values():
            rows.append(
                {
                    "name": param.name,
                    "group": param.group,
                    "value": param.value,
                    "min": param.min_value,
                    "max": param.max_value,
                    "locked": param.locked,
                }
            )
        return rows


def _doc(what: str, when: str, increased: str, edge: str) -> Dict[str, str]:
    return {
        "what": what,
        "when": when,
        "increased": increased,
        "edge": edge,
    }


def _default_params() -> list[Parameter]:
    params = [
        Parameter(
            name="exploration_epsilon",
            group="global",
            default=0.02,
            value=0.02,
            min_value=0.0,
            max_value=0.20,
            locked=False,
            doc=_doc(
                "Random exploration probability.",
                "Any decision with multiple close moves.",
                "More randomness in play.",
                "Keep low for deterministic debugging.",
            ),
        ),
        Parameter(
            name="softmax_temperature",
            group="global",
            default=0.15,
            value=0.15,
            min_value=0.01,
            max_value=1.0,
            locked=False,
            doc=_doc(
                "Sampling temperature for softmax.",
                "Used when epsilon is zero.",
                "Higher temp makes choices more uniform.",
                "Too low becomes greedy.",
            ),
        ),
        Parameter(
            name="late_game_weight",
            group="global",
            default=0.35,
            value=0.35,
            min_value=0.0,
            max_value=1.0,
            locked=False,
            doc=_doc(
                "Weight for late-trick adjustments.",
                "Late hand decisions.",
                "Greater focus on end-game control.",
                "Overweighting can ignore early safety.",
            ),
        ),
        Parameter(
            name="risk_aversion",
            group="global",
            default=0.60,
            value=0.60,
            min_value=0.0,
            max_value=1.5,
            locked=False,
            doc=_doc(
                "Global scaling for risky moves.",
                "Situations with point exposure.",
                "Avoids risky points more aggressively.",
                "Too high can cause passive play.",
            ),
        ),
        Parameter(
            name="debug_explain_threshold",
            group="global",
            default=0.15,
            value=0.15,
            min_value=0.0,
            max_value=1.0,
            locked=False,
            doc=_doc(
                "Score gap threshold for explain logs.",
                "Close decisions.",
                "More explain logging.",
                "Set to 0 to disable explain logs.",
            ),
        ),
        Parameter(
            name="pass_points_bias",
            group="passing",
            default=1.5,
            value=1.5,
            min_value=-2.0,
            max_value=5.0,
            locked=False,
            doc=_doc(
                "Bias to pass point cards.",
                "Passing phase.",
                "Passes more points away.",
                "Overuse can break suit balance.",
            ),
        ),
        Parameter(
            name="pass_heart_bias",
            group="passing",
            default=0.9,
            value=0.9,
            min_value=-2.0,
            max_value=5.0,
            locked=False,
            doc=_doc(
                "Bias to pass hearts.",
                "Passing phase.",
                "Passes more hearts.",
                "Too high can leave no hearts to break.",
            ),
        ),
        Parameter(
            name="pass_high_heart_bias",
            group="passing",
            default=1.2,
            value=1.2,
            min_value=-2.0,
            max_value=5.0,
            locked=False,
            doc=_doc(
                "Extra bias for high hearts.",
                "Passing phase.",
                "Passes high hearts more.",
                "Overuse can keep low hearts only.",
            ),
        ),
        Parameter(
            name="pass_qs_bias",
            group="passing",
            default=4.0,
            value=4.0,
            min_value=0.0,
            max_value=8.0,
            locked=False,
            doc=_doc(
                "Bias to pass the queen of spades.",
                "Passing phase.",
                "Strongly prefers passing Q spades.",
                "Can be moot if you cannot pass it.",
            ),
        ),
        Parameter(
            name="pass_as_bias",
            group="passing",
            default=2.0,
            value=2.0,
            min_value=-2.0,
            max_value=6.0,
            locked=False,
            doc=_doc(
                "Bias to pass ace of spades.",
                "Passing phase.",
                "Passes A spades more.",
                "May lose control if spades are safe.",
            ),
        ),
        Parameter(
            name="pass_ks_bias",
            group="passing",
            default=1.5,
            value=1.5,
            min_value=-2.0,
            max_value=6.0,
            locked=False,
            doc=_doc(
                "Bias to pass king of spades.",
                "Passing phase.",
                "Passes K spades more.",
                "May reduce control of spades.",
            ),
        ),
        Parameter(
            name="pass_high_spade_bias",
            group="passing",
            default=0.8,
            value=0.8,
            min_value=-2.0,
            max_value=5.0,
            locked=False,
            doc=_doc(
                "Bias to pass high spades.",
                "Passing phase.",
                "Passes higher spades.",
                "Too high can keep low spades only.",
            ),
        ),
        Parameter(
            name="pass_short_suit_preserve",
            group="passing",
            default=1.2,
            value=1.2,
            min_value=-3.0,
            max_value=5.0,
            locked=False,
            doc=_doc(
                "Penalty for passing from short suits.",
                "Passing phase.",
                "Preserves short suits for voids.",
                "Too high can ignore point cards.",
            ),
        ),
        Parameter(
            name="pass_long_suit_reduce",
            group="passing",
            default=0.7,
            value=0.7,
            min_value=-3.0,
            max_value=5.0,
            locked=False,
            doc=_doc(
                "Bias to reduce long suits.",
                "Passing phase.",
                "Passes from long suits more.",
                "Can reduce control in a suit.",
            ),
        ),
        Parameter(
            name="pass_void_bonus",
            group="passing",
            default=2.0,
            value=2.0,
            min_value=0.0,
            max_value=6.0,
            locked=False,
            doc=_doc(
                "Bonus for creating voids.",
                "Passing phase.",
                "Encourages void creation.",
                "Void chasing can hurt hand strength.",
            ),
        ),
        Parameter(
            name="pass_doubleton_bonus",
            group="passing",
            default=0.8,
            value=0.8,
            min_value=-2.0,
            max_value=4.0,
            locked=False,
            doc=_doc(
                "Bonus for breaking doubletons.",
                "Passing phase.",
                "Favors turning doubletons into voids.",
                "May over-prioritize suit distribution.",
            ),
        ),
        Parameter(
            name="pass_keep_control_cards",
            group="passing",
            default=-0.8,
            value=-0.8,
            min_value=-5.0,
            max_value=2.0,
            locked=False,
            doc=_doc(
                "Bias to keep high control cards.",
                "Passing phase.",
                "Keeps A/K in non-spades.",
                "Too low can keep dangerous points.",
            ),
        ),
        Parameter(
            name="pass_balance_penalty",
            group="passing",
            default=0.6,
            value=0.6,
            min_value=0.0,
            max_value=3.0,
            locked=False,
            doc=_doc(
                "Penalty for unbalanced suits.",
                "Passing phase.",
                "Maintains suit balance.",
                "Overuse can block void creation.",
            ),
        ),
        Parameter(
            name="pass_moon_block_bias",
            group="passing",
            default=0.0,
            value=0.0,
            min_value=-3.0,
            max_value=3.0,
            locked=False,
            doc=_doc(
                "Bias for anti-moon passing.",
                "Passing phase.",
                "Reserved for moon blocking.",
                "Unused in V1 logic.",
            ),
        ),
        Parameter(
            name="play_point_avoid_bias",
            group="play",
            default=2.0,
            value=2.0,
            min_value=-2.0,
            max_value=6.0,
            locked=False,
            doc=_doc(
                "Penalty for playing point cards.",
                "All play decisions.",
                "Avoids point cards more.",
                "Too high can force worse losses.",
            ),
        ),
        Parameter(
            name="play_qs_avoid_bias",
            group="play",
            default=4.0,
            value=4.0,
            min_value=0.0,
            max_value=10.0,
            locked=False,
            doc=_doc(
                "Penalty for playing Q spades.",
                "All play decisions.",
                "Strongly avoids Q spades.",
                "May keep Q too long.",
            ),
        ),
        Parameter(
            name="play_heart_avoid_bias",
            group="play",
            default=1.2,
            value=1.2,
            min_value=-2.0,
            max_value=6.0,
            locked=False,
            doc=_doc(
                "Penalty for playing hearts.",
                "All play decisions.",
                "Avoids hearts more.",
                "Can delay necessary heart plays.",
            ),
        ),
        Parameter(
            name="lead_heart_penalty_unbroken",
            group="play",
            default=5.0,
            value=5.0,
            min_value=0.0,
            max_value=12.0,
            locked=False,
            doc=_doc(
                "Penalty for leading hearts before break.",
                "When leading without hearts broken.",
                "More reluctant to lead hearts.",
                "Can cause awkward leads if only hearts.",
            ),
        ),
        Parameter(
            name="break_hearts_bias",
            group="play",
            default=-0.3,
            value=-0.3,
            min_value=-3.0,
            max_value=3.0,
            locked=False,
            doc=_doc(
                "Bias toward breaking hearts by dumping.",
                "When void and dumping off-suit.",
                "Encourages breaking hearts sooner.",
                "Too low can throw points early.",
            ),
        ),
        Parameter(
            name="win_trick_bias",
            group="play",
            default=-0.2,
            value=-0.2,
            min_value=-3.0,
            max_value=3.0,
            locked=False,
            doc=_doc(
                "Bias for winning a trick.",
                "When a move can win.",
                "More willingness to win tricks.",
                "Winning can be costly with points.",
            ),
        ),
        Parameter(
            name="win_trick_points_multiplier",
            group="play",
            default=3.0,
            value=3.0,
            min_value=0.0,
            max_value=8.0,
            locked=False,
            doc=_doc(
                "Penalty multiplier for winning with points on table.",
                "When likely to win a trick.",
                "Avoids winning point-heavy tricks.",
                "Overuse can throw control away.",
            ),
        ),
        Parameter(
            name="win_trick_late_game_bonus",
            group="play",
            default=-1.0,
            value=-1.0,
            min_value=-6.0,
            max_value=2.0,
            locked=False,
            doc=_doc(
                "Bias for winning late-game tricks.",
                "Late in the hand.",
                "Wins more late tricks.",
                "Can over-prioritize late control.",
            ),
        ),
        Parameter(
            name="lead_short_suit_bonus",
            group="play",
            default=-0.8,
            value=-0.8,
            min_value=-4.0,
            max_value=2.0,
            locked=False,
            doc=_doc(
                "Bias to lead short suits.",
                "When leading.",
                "Leads shorter suits more.",
                "Too low can expose voids early.",
            ),
        ),
        Parameter(
            name="lead_long_suit_penalty",
            group="play",
            default=0.4,
            value=0.4,
            min_value=-2.0,
            max_value=4.0,
            locked=False,
            doc=_doc(
                "Penalty for leading long suits.",
                "When leading.",
                "Avoids leading long suits.",
                "May miss safe lead options.",
            ),
        ),
        Parameter(
            name="follow_low_card_bias",
            group="play",
            default=-0.6,
            value=-0.6,
            min_value=-3.0,
            max_value=3.0,
            locked=False,
            doc=_doc(
                "Bias to follow low cards.",
                "When following suit.",
                "Plays lower cards more.",
                "Too low can waste low cards early.",
            ),
        ),
        Parameter(
            name="dump_points_when_void_bonus",
            group="play",
            default=-1.2,
            value=-1.2,
            min_value=-6.0,
            max_value=2.0,
            locked=False,
            doc=_doc(
                "Bias to dump points when void.",
                "When off-suit.",
                "Encourages dumping points.",
                "Can throw points too early.",
            ),
        ),
        Parameter(
            name="dump_qs_when_void_bonus",
            group="play",
            default=-3.0,
            value=-3.0,
            min_value=-12.0,
            max_value=0.0,
            locked=False,
            doc=_doc(
                "Bias to dump Q spades when void.",
                "When off-suit.",
                "Encourages dumping Q spades.",
                "May be illegal on first trick.",
            ),
        ),
        Parameter(
            name="avoid_dumping_points_first_trick_penalty",
            group="play",
            default=20.0,
            value=20.0,
            min_value=0.0,
            max_value=50.0,
            locked=False,
            doc=_doc(
                "Penalty for dumping points on first trick.",
                "First trick and void in clubs.",
                "Avoids point dumping.",
                "Too high can overconstrain choices.",
            ),
        ),
        Parameter(
            name="protect_from_qs_trap_bias",
            group="play",
            default=1.0,
            value=1.0,
            min_value=-3.0,
            max_value=6.0,
            locked=False,
            doc=_doc(
                "Penalty for low spades when Q unseen.",
                "Spade leads before Q spades seen.",
                "Avoids low spade traps.",
                "May avoid safe low spades.",
            ),
        ),
        Parameter(
            name="spade_high_cover_bias",
            group="play",
            default=-0.3,
            value=-0.3,
            min_value=-3.0,
            max_value=3.0,
            locked=False,
            doc=_doc(
                "Bias to cover spades with A/K.",
                "Spade tricks when Q unseen.",
                "Covers spades more often.",
                "Can waste high spades.",
            ),
        ),
        Parameter(
            name="safe_discard_nonpoints_bias",
            group="play",
            default=-0.7,
            value=-0.7,
            min_value=-4.0,
            max_value=2.0,
            locked=False,
            doc=_doc(
                "Bias to discard safe nonpoints.",
                "When void and discarding.",
                "Encourages safe nonpoint discards.",
                "May hold points too long.",
            ),
        ),
        Parameter(
            name="avoid_leading_spades_with_q_unseen",
            group="play",
            default=1.2,
            value=1.2,
            min_value=-2.0,
            max_value=6.0,
            locked=False,
            doc=_doc(
                "Penalty for leading spades before Q seen.",
                "When leading.",
                "Avoids spade leads.",
                "May ignore safe spade leads.",
            ),
        ),
        Parameter(
            name="keep_ace_king_for_control_bias",
            group="play",
            default=-0.2,
            value=-0.2,
            min_value=-3.0,
            max_value=3.0,
            locked=False,
            doc=_doc(
                "Bias to save A/K early.",
                "Early tricks.",
                "Keeps control cards longer.",
                "Can delay necessary plays.",
            ),
        ),
        Parameter(
            name="moon_attempt_bias",
            group="play",
            default=0.0,
            value=0.0,
            min_value=-4.0,
            max_value=4.0,
            locked=False,
            doc=_doc(
                "Bias for shooting the moon.",
                "Reserved for future logic.",
                "Increases moon attempts.",
                "Unused in V1.",
            ),
        ),
        Parameter(
            name="moon_likelihood_threshold",
            group="play",
            default=0.85,
            value=0.85,
            min_value=0.5,
            max_value=0.98,
            locked=False,
            doc=_doc(
                "Threshold for moon logic.",
                "Reserved for future logic.",
                "Higher threshold means fewer attempts.",
                "Unused in V1.",
            ),
        ),
        Parameter(
            name="anti_moon_block_bias",
            group="play",
            default=0.0,
            value=0.0,
            min_value=-4.0,
            max_value=4.0,
            locked=False,
            doc=_doc(
                "Bias for blocking moon attempts.",
                "Reserved for future logic.",
                "Stronger moon blocking.",
                "Unused in V1.",
            ),
        ),
    ]
    return params
