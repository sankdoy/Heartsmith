from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit


RULES_TEXT = (
    "Game Rules\n\n"
    "Setup:\n"
    "- 4 players, 13 cards each.\n"
    "- Scores are penalty points (lower is better).\n\n"
    "Passing:\n"
    "- Left -> Right -> Across -> Keep -> repeat.\n"
    "- Each player passes exactly 3 cards (except keep).\n\n"
    "Trick Play:\n"
    "- 13 tricks per hand.\n"
    "- First trick must lead 2 of clubs.\n"
    "- Must follow suit if possible.\n"
    "- Hearts cannot be led until broken unless only hearts remain.\n"
    "- First trick: if void in clubs, you may not dump points if you have any non-point card.\n\n"
    "Scoring:\n"
    "- Each heart = 1 point.\n"
    "- Queen of spades = 13 points.\n"
    "- Trick winner takes all points in the trick.\n\n"
    "Shooting the Moon:\n"
    "- If a player takes all 26 points: they score 0, others score 26.\n\n"
    "Game End:\n"
    "- Play until someone reaches 100+ points.\n"
    "- End only when there is a single lowest-score player.\n\n"
    "Training Settings Glossary\n\n"
    "- Training (simple): presets + Auto mode manage settings for you.\n"
    "- Training (advanced): full control of hands/tick, seeds, opponents, logging.\n"
    "- hands_per_tick: hands simulated per training update tick.\n"
    "- seed: deterministic RNG seed for repeatability.\n"
    "- train_seed_mode: fixed/cycle/random seed strategy.\n"
    "- updates_per_sec: UI update cadence for training metrics.\n"
    "- hands_per_second_target: adaptive speed target (0 disables).\n"
    "- fast_mode: reduces logging and update frequency.\n"
    "- log_every_n_hands: summary log frequency.\n"
    "- log_verbosity: Quiet/Normal/Verbose controls debug output.\n\n"
    "Evaluation Settings Glossary\n\n"
    "- eval_hands_per_seed: hands simulated per eval seed.\n"
    "- eval_seed_preset: quick/standard/thorough seed lists.\n"
    "- Run evaluation now: evaluate current params immediately.\n\n"
    "Plot Controls Glossary\n\n"
    "- window: number of recent ticks shown.\n"
    "- smoothing: EMA or rolling mean to reduce noise.\n\n"
    "Opponent Pools\n\n"
    "- Training pool: opponents used during optimization.\n"
    "- Evaluation pool: opponents used for reporting; can be stricter.\n\n"
    "Stats Glossary\n\n"
    "- Train mean points: average points/hand over training tick.\n"
    "- Eval mean points ±SE: average points/hand and standard error across seeds.\n"
    "- Holdout mean points: evaluation on holdout seeds.\n"
    "- Best eval mean: lowest eval mean points seen so far.\n"
    "- Win rate: fraction of hands with lowest points (ties count as win).\n"
    "- Q♠ taken rate: fraction of hands where Q♠ is taken.\n"
    "- Δ vs SafeBot: difference in points/hand (lower is better).\n\n"
    "Auto Mode Notes\n\n"
    "- Auto mode may switch seed strategy, opponents, and eval cadence.\n"
    "- Plateau means recent evals are no longer improving.\n"
    "- Overfit means eval improves but holdout worsens.\n\n"
    "Decision Inspector\n\n"
    "- Use Play mode to inspect a single hand’s decisions.\n"
    "- Capture next hand now: stores traces in runs/<run_id>/samples/.\n\n"
    "Baselines\n\n"
    "- Average points/hand baseline is 6.5 (26 points / 4 players).\n"
    "- Holdout uses a separate seed set to check for overfitting.\n"
    "- Training uses deterministic seeds unless seed mode changes.\n\n"
    "How to Interpret Results\n\n"
    "- Lower points/hand is better.\n"
    "- Compare eval and holdout trends to detect overfitting.\n"
    "- A rising win rate with falling points is a good sign.\n"
)


class AboutPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("About / Rules"))
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(RULES_TEXT)
        layout.addWidget(text)
        self.setLayout(layout)
