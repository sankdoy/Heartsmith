from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from hearts_ai.bots.base import Bot
from hearts_ai.core.cards import Card, Rank, Suit
from hearts_ai.core.scoring import card_points
from hearts_ai.core.state import TrickState
from hearts_ai.training.params import ParameterSet
from hearts_ai.util.rng import RNG, create_rng

logger = logging.getLogger(__name__)


def rank_norm(rank: Rank) -> float:
    return (int(rank) - 2) / 12.0


def suit_len_norm(count: int) -> float:
    return count / 13.0


def trick_idx_norm(trick_index: int) -> float:
    return trick_index / 12.0


def points_on_table_norm(trick: TrickState) -> float:
    hearts = sum(1 for _, card in trick.cards if card.suit == Suit.HEARTS)
    qs = any(card.suit == Suit.SPADES and card.rank == Rank.QUEEN for _, card in trick.cards)
    return (hearts + (13 if qs else 0)) / 26.0


@dataclass
class MoveScore:
    card: Card
    score: float
    terms: list[tuple[str, float]]


class HeuristicBot(Bot):
    def __init__(
        self,
        params: ParameterSet | None = None,
        rng: RNG | None = None,
        explain_enabled: bool = True,
    ) -> None:
        self._params = params or ParameterSet()
        self._rng = rng or create_rng()
        self._qs_played = False
        self._last_hand_size = 0
        self._hands_seen = 0
        self._explain_logs = 0
        self._explain_enabled = explain_enabled

    def update_params(self, params: ParameterSet) -> None:
        self._params = params

    def score_legal_moves(
        self,
        hand: list[Card],
        legal_moves: list[Card],
        trick: TrickState,
        hearts_broken: bool,
        is_first_trick: bool,
    ) -> list[MoveScore]:
        return self._score_moves(hand, legal_moves, trick, hearts_broken, is_first_trick)

    def choose_with_trace(
        self,
        hand: list[Card],
        legal_moves: list[Card],
        trick: TrickState,
        hearts_broken: bool,
        is_first_trick: bool,
        top_n: int = 5,
    ) -> dict:
        scores = self._score_moves(hand, legal_moves, trick, hearts_broken, is_first_trick)
        scores_sorted = sorted(scores, key=lambda m: m.score)
        chosen = scores_sorted[0] if scores_sorted else None
        top = scores_sorted[:top_n]
        return {
            "chosen": chosen.card.short() if chosen else None,
            "chosen_score": chosen.score if chosen else None,
            "candidates": [
                {
                    "card": entry.card.short(),
                    "score": entry.score,
                    "terms": sorted(entry.terms, key=lambda t: abs(t[1]), reverse=True)[:5],
                }
                for entry in top
            ],
        }

    def choose_pass(self, hand: list[Card]) -> list[Card]:
        scores = self._score_pass_cards(hand)
        scores.sort(key=lambda s: s[1], reverse=True)
        return [card for card, _ in scores[:3]]

    def choose_card(
        self,
        hand: list[Card],
        legal_moves: list[Card],
        trick: TrickState,
        hearts_broken: bool,
        is_first_trick: bool,
    ) -> Card:
        self._maybe_update_hand_state(hand)
        if not legal_moves:
            raise ValueError("No legal moves")

        epsilon = self._params["exploration_epsilon"]
        if epsilon > 0 and self._rng.random() < epsilon:
            return self._rng.choice(legal_moves)

        move_scores = self._score_moves(hand, legal_moves, trick, hearts_broken, is_first_trick)
        if not move_scores:
            return legal_moves[0]

        self._maybe_log_explain(move_scores, trick, hearts_broken)

        temperature = self._params["softmax_temperature"]
        if temperature > 0 and epsilon == 0:
            chosen = self._sample_softmax(move_scores, temperature)
        else:
            chosen = min(move_scores, key=lambda m: m.score).card

        chosen_score = next((m for m in move_scores if m.card == chosen), move_scores[0])
        if card_points(chosen_score.card) > 0:
            self._log_point_card(chosen_score)
        self._update_qs_seen(trick, chosen_score.card)
        return chosen_score.card

    def _score_pass_cards(self, hand: list[Card]) -> list[tuple[Card, float]]:
        suit_counts = _suit_counts(hand)
        scores: list[tuple[Card, float]] = []
        for card in hand:
            points_norm = card_points(card) / 13.0
            is_heart = 1.0 if card.suit == Suit.HEARTS else 0.0
            is_qs = 1.0 if card.suit == Suit.SPADES and card.rank == Rank.QUEEN else 0.0
            is_as = 1.0 if card.suit == Suit.SPADES and card.rank == Rank.ACE else 0.0
            is_ks = 1.0 if card.suit == Suit.SPADES and card.rank == Rank.KING else 0.0
            is_spade = 1.0 if card.suit == Suit.SPADES else 0.0
            rank_value = rank_norm(card.rank)

            suit_count = suit_counts[card.suit]
            suit_norm = suit_len_norm(suit_count)
            would_void = 1.0 if suit_count == 1 else 0.0
            would_doubleton = 1.0 if suit_count == 2 else 0.0

            control_non_spade = 1.0 if card.suit != Suit.SPADES and card.rank in (Rank.ACE, Rank.KING) else 0.0

            counts_after = suit_counts.copy()
            counts_after[card.suit] -= 1
            imbalance = (max(counts_after.values()) - min(counts_after.values())) / 13.0

            score = 0.0
            score += self._params["pass_points_bias"] * points_norm
            score += self._params["pass_heart_bias"] * is_heart
            score += self._params["pass_high_heart_bias"] * is_heart * rank_value
            score += self._params["pass_qs_bias"] * is_qs
            score += self._params["pass_as_bias"] * is_as
            score += self._params["pass_ks_bias"] * is_ks
            score += self._params["pass_high_spade_bias"] * is_spade * rank_value
            score += self._params["pass_short_suit_preserve"] * (-(1 - suit_norm))
            score += self._params["pass_long_suit_reduce"] * suit_norm
            score += self._params["pass_keep_control_cards"] * control_non_spade
            score += self._params["pass_balance_penalty"] * imbalance
            score += self._params["pass_void_bonus"] * would_void
            score += self._params["pass_doubleton_bonus"] * would_doubleton
            score += self._params["pass_moon_block_bias"] * 0.0

            scores.append((card, score))
        return scores

    def _score_moves(
        self,
        hand: list[Card],
        legal_moves: list[Card],
        trick: TrickState,
        hearts_broken: bool,
        is_first_trick: bool,
    ) -> list[MoveScore]:
        lead_suit = trick.lead_suit
        suit_counts = _suit_counts(hand)
        only_hearts = all(card.suit == Suit.HEARTS for card in hand)
        trick_index = max(0, 13 - len(hand))
        trick_norm = min(1.0, trick_idx_norm(trick_index) * (1.0 + self._params["late_game_weight"]))
        table_points = points_on_table_norm(trick)

        must_follow = lead_suit is not None and any(card.suit == lead_suit for card in hand)
        best_scores: list[MoveScore] = []

        for move in legal_moves:
            is_leading = lead_suit is None
            move_is_point = 1.0 if card_points(move) > 0 else 0.0
            move_is_qs = 1.0 if move.suit == Suit.SPADES and move.rank == Rank.QUEEN else 0.0
            move_is_heart = 1.0 if move.suit == Suit.HEARTS else 0.0
            dumping_off_suit = 1.0 if lead_suit is not None and move.suit != lead_suit else 0.0

            currently_winning = 0.0
            if lead_suit is not None and move.suit == lead_suit:
                highest = max((card.rank for _, card in trick.cards if card.suit == lead_suit), default=Rank.TWO)
                if move.rank > highest:
                    currently_winning = 1.0

            first_trick_point_dump = 0.0
            if is_first_trick and lead_suit == Suit.CLUBS and dumping_off_suit:
                has_non_point = any(card_points(card) == 0 for card in hand)
                if has_non_point and move_is_point:
                    first_trick_point_dump = 1.0

            suit_norm = suit_len_norm(suit_counts[move.suit])
            rank_value = rank_norm(move.rank)

            feature_qs_trap_risk = 1.0 if move.suit == Suit.SPADES and not self._qs_played and rank_value <= 0.35 else 0.0
            feature_spade_cover = 1.0 if move.suit == Suit.SPADES and move.rank in (Rank.ACE, Rank.KING) and not self._qs_played and table_points == 0 else 0.0
            safe_win = currently_winning == 1.0 and table_points == 0 and move_is_point == 0
            feature_spending_control = 1.0 if move.rank in (Rank.ACE, Rank.KING) and trick_norm < 0.35 and not safe_win else 0.0

            score = 0.0
            terms: list[tuple[str, float]] = []

            def add_term(name: str, value: float) -> None:
                if value == 0:
                    return
                terms.append((name, value))

            add = self._params.__getitem__

            term = add("play_point_avoid_bias") * move_is_point
            score += term
            add_term("play_point_avoid_bias", term)

            term = add("play_qs_avoid_bias") * move_is_qs
            score += term
            add_term("play_qs_avoid_bias", term)

            term = add("play_heart_avoid_bias") * move_is_heart
            score += term
            add_term("play_heart_avoid_bias", term)

            term = add("lead_heart_penalty_unbroken") * (
                1.0 if is_leading and move_is_heart and not hearts_broken and not only_hearts else 0.0
            )
            score += term
            add_term("lead_heart_penalty_unbroken", term)

            term = add("break_hearts_bias") * (
                1.0 if move_is_heart and not hearts_broken and dumping_off_suit else 0.0
            )
            score += term
            add_term("break_hearts_bias", term)

            term = add("win_trick_bias") * currently_winning
            score += term
            add_term("win_trick_bias", term)

            term = add("win_trick_points_multiplier") * (currently_winning * table_points)
            score += term
            add_term("win_trick_points_multiplier", term)

            term = add("win_trick_late_game_bonus") * (currently_winning * trick_norm)
            score += term
            add_term("win_trick_late_game_bonus", term)

            term = add("lead_short_suit_bonus") * (1.0 if is_leading else 0.0) * (1 - suit_norm)
            score += term
            add_term("lead_short_suit_bonus", term)

            term = add("lead_long_suit_penalty") * (1.0 if is_leading else 0.0) * suit_norm
            score += term
            add_term("lead_long_suit_penalty", term)

            term = add("follow_low_card_bias") * (1.0 if (not is_leading and must_follow) else 0.0) * rank_value
            score += term
            add_term("follow_low_card_bias", term)

            term = add("dump_points_when_void_bonus") * (dumping_off_suit * move_is_point)
            score += term
            add_term("dump_points_when_void_bonus", term)

            term = add("dump_qs_when_void_bonus") * (dumping_off_suit * move_is_qs)
            score += term
            add_term("dump_qs_when_void_bonus", term)

            term = add("avoid_dumping_points_first_trick_penalty") * first_trick_point_dump
            score += term
            add_term("avoid_dumping_points_first_trick_penalty", term)

            term = add("protect_from_qs_trap_bias") * feature_qs_trap_risk
            score += term
            add_term("protect_from_qs_trap_bias", term)

            term = add("spade_high_cover_bias") * feature_spade_cover
            score += term
            add_term("spade_high_cover_bias", term)

            term = add("safe_discard_nonpoints_bias") * (
                dumping_off_suit * (1.0 - move_is_point) * rank_value
            )
            score += term
            add_term("safe_discard_nonpoints_bias", term)

            term = add("avoid_leading_spades_with_q_unseen") * (
                1.0 if is_leading and move.suit == Suit.SPADES and not self._qs_played else 0.0
            )
            score += term
            add_term("avoid_leading_spades_with_q_unseen", term)

            term = add("keep_ace_king_for_control_bias") * feature_spending_control
            score += term
            add_term("keep_ace_king_for_control_bias", term)

            score *= (1.0 + self._params["risk_aversion"] * move_is_point)

            best_scores.append(MoveScore(card=move, score=score, terms=terms))

        return best_scores

    def _sample_softmax(self, move_scores: list[MoveScore], temperature: float) -> Card:
        scores = [m.score for m in move_scores]
        min_score = min(scores)
        exp_scores = [math.exp(-(score - min_score) / max(temperature, 1e-6)) for score in scores]
        total = sum(exp_scores)
        if total <= 0:
            return min(move_scores, key=lambda m: m.score).card
        r = self._rng.random() * total
        running = 0.0
        for move, weight in zip(move_scores, exp_scores):
            running += weight
            if r <= running:
                return move.card
        return move_scores[-1].card

    def _maybe_update_hand_state(self, hand: list[Card]) -> None:
        if len(hand) == 13 and self._last_hand_size != 13:
            self._hands_seen += 1
            self._qs_played = False
        self._last_hand_size = len(hand)

    def _maybe_log_explain(self, move_scores: list[MoveScore], trick: TrickState, hearts_broken: bool) -> None:
        if len(move_scores) < 2:
            return
        if not self._should_log_explain():
            return
        sorted_scores = sorted(move_scores, key=lambda m: m.score)
        best = sorted_scores[0]
        second = sorted_scores[1]
        threshold = self._params["debug_explain_threshold"]
        if abs(best.score - second.score) > threshold:
            return

        top_terms = sorted(best.terms, key=lambda t: abs(t[1]), reverse=True)[:5]
        lead = trick.lead_suit.value if trick.lead_suit else "-"
        table_points = points_on_table_norm(trick)
        terms_text = ",".join([f"{name}:{value:.2f}" for name, value in top_terms])
        if terms_text:
            logger.debug(
                "Close call t=%d lead=%s hearts_broken=%s table_points=%.2f top=%s %.3f vs %s %.3f terms=%s",
                max(0, 13 - self._last_hand_size),
                lead,
                hearts_broken,
                table_points,
                best.card.short(),
                best.score,
                second.card.short(),
                second.score,
                terms_text,
            )
        else:
            logger.debug(
                "Close call t=%d lead=%s hearts_broken=%s table_points=%.2f top=%s %.3f vs %s %.3f",
                max(0, 13 - self._last_hand_size),
                lead,
                hearts_broken,
                table_points,
                best.card.short(),
                best.score,
                second.card.short(),
                second.score,
            )
        self._explain_logs += 1

    def _log_point_card(self, best: MoveScore) -> None:
        if not self._should_log_explain():
            return
        top_terms = sorted(best.terms, key=lambda t: abs(t[1]), reverse=True)[:3]
        terms_text = ",".join([f"{name}:{value:.2f}" for name, value in top_terms])
        if terms_text:
            logger.debug(
                "Point card %s score=%.3f terms=%s",
                best.card.short(),
                best.score,
                terms_text,
            )
        else:
            logger.debug("Point card %s score=%.3f", best.card.short(), best.score)
        self._explain_logs += 1

    def _should_log_explain(self) -> bool:
        if not self._explain_enabled:
            return False
        if self._hands_seen <= 0:
            return False
        allowed = 2 * (self._hands_seen // 100 + 1)
        return self._explain_logs < allowed

    def _update_qs_seen(self, trick: TrickState, played_card: Card) -> None:
        if played_card.suit == Suit.SPADES and played_card.rank == Rank.QUEEN:
            self._qs_played = True
        if any(card.suit == Suit.SPADES and card.rank == Rank.QUEEN for _, card in trick.cards):
            self._qs_played = True


def _suit_counts(hand: list[Card]) -> dict[Suit, int]:
    counts = {Suit.CLUBS: 0, Suit.DIAMONDS: 0, Suit.SPADES: 0, Suit.HEARTS: 0}
    for card in hand:
        counts[card.suit] += 1
    return counts
