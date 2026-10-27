"""debate.py

State and rules of a panel debate: who speaks in each round, what they may say, and when the debate ends.

Rounds (one pass around the flow's ring of participants each):
    1 opinions    every participant; tool agents call their tools, experts search their article or abstain
    2 challenges  every participant that did not abstain reviews other participants' claims (at most 3 reviews)
    3 replies     only participants whose claims were challenged; defend, concede or clarify, nothing new
    4 challenges  extra cycle, only when the tool agents still disagree and a defended challenge involves them
    5 replies     replies to the round-4 challenges

The loop closes by construction: every claim gets one review window, every challenge gets one reply window (or is
marked unanswered), reply rounds cannot open new challenges, and there is at most one extra cycle.

Agreement is decided here from the tool agents' structured positions, never from what a model says about it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

ROUND_KINDS = {1: "opinions", 2: "challenges", 3: "replies", 4: "challenges", 5: "replies"}
MAX_REVIEWS_PER_ROUND = 3
MAX_CLAIMS_PER_TURN = 4
AGE_TOLERANCE_RINGS = 1.0
CHUNK_ID = re.compile(r"\b[a-z0-9_]+:\d{3}\b")


@dataclass
class Participant:
    node_id: str
    code: str
    kind: str  # "tool_agent" or "expert"
    label: str
    card: str  # what the others see about this participant (tools or article)


@dataclass
class Claim:
    claim_id: str
    by: str
    round: int
    text: str
    evidence: list[str]
    reviews: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Challenge:
    challenge_id: str
    by: str
    claim_id: str
    target: str
    round: int
    reason: str
    evidence: list[str]
    status: str = "open"  # open | defended | conceded | clarified | unanswered
    response: str = ""
    response_evidence: list[str] = field(default_factory=list)


def normalize_id(value: object) -> str:
    """Claim and challenge ids as models write them ('[R1-SEX_A-1]', ' c2 ') -> 'R1-SEX_A-1', 'C2'."""
    return str(value or "").strip().strip("[]()`'\"").strip().upper()


def participant_code(node_id: str) -> str:
    """abalone_sex_agent_a -> SEX_A; article_nash_1994_agent -> NASH_1994."""
    core = re.sub(r"^(abalone|article)_", "", node_id)
    core = re.sub(r"_agent(_|$)", r"\1", core).strip("_")
    return core.upper()


class Debate:
    def __init__(self, task: str, participants: list[Participant], max_cycles: int = 2):
        if task not in ("sex", "age"):
            raise ValueError(f"task must be 'sex' or 'age', got {task!r}")
        self.task = task
        self.participants = {p.node_id: p for p in participants}
        self.order = [p.node_id for p in participants]
        self.max_cycles = max_cycles
        self.round = 1
        self.cycles_done = 0
        self.claims: dict[str, Claim] = {}
        self.challenges: dict[str, Challenge] = {}
        self.positions: dict[str, list[dict[str, Any]]] = {}  # tool agent node -> positions per turn
        self.expectations: dict[str, str] = {}
        self.abstained: set[str] = set()
        self.spoke: dict[int, list[str]] = {}
        self.events: list[str] = []
        self.next_round_needed = True
        self._pass_closed_for_round = 0
        self.disputed_claims: set[str] = set()

    # ------------------------------------------------------------------ who speaks

    @property
    def kind(self) -> str:
        return ROUND_KINDS[self.round]

    def tool_agents(self) -> list[str]:
        return [n for n in self.order if self.participants[n].kind == "tool_agent"]

    def open_challenges_for(self, node_id: str) -> list[Challenge]:
        return [c for c in self.challenges.values() if c.target == node_id and c.status == "open" and c.round == self.round - 1]

    def needs_turn(self, node_id: str) -> bool:
        if node_id not in self.participants:
            return True
        if self.kind == "opinions":
            return True
        if node_id in self.abstained:
            return False
        if self.kind == "challenges":
            if self.round == 2:
                return any(c.by != node_id for c in self.claims.values())
            involved = {self.claims[c].by for c in self.disputed_claims} | {
                ch.by for ch in self.challenges.values() if ch.claim_id in self.disputed_claims
            }
            return node_id in involved or self.participants[node_id].kind == "tool_agent"
        return bool(self.open_challenges_for(node_id))

    # ------------------------------------------------------------------ recording a turn

    def record_turn(self, node_id: str, output: dict[str, Any]) -> list[str]:
        """Apply one participant's validated output to the debate. Returns warnings about discarded items."""
        participant = self.participants[node_id]
        result = output.get("result") or {}
        warnings: list[str] = []
        self.spoke.setdefault(self.round, []).append(node_id)

        if output.get("status") == "abstained":
            if self.kind == "opinions" and participant.kind == "expert":
                self.abstained.add(node_id)
                self.events.append(f"Round {self.round}: {participant.code} abstained (its article does not cover the specimen).")
                return warnings
            warnings.append(f"{participant.code} tried to abstain in a {self.kind} round; ignored.")

        position = result.get("position")
        if participant.kind == "tool_agent" and isinstance(position, dict):
            self.positions.setdefault(node_id, []).append({"round": self.round, **position})
        if participant.kind == "expert" and result.get("expectation"):
            self.expectations[node_id] = str(result["expectation"])

        if self.kind == "opinions":
            for n, claim in enumerate((result.get("claims") or [])[:MAX_CLAIMS_PER_TURN], start=1):
                evidence = [str(e) for e in claim.get("evidence") or [] if str(e).strip()]
                if not evidence:
                    warnings.append(f"{participant.code}: a claim without evidence was discarded.")
                    continue
                claim_id = f"R{self.round}-{participant.code}-{n}"
                self.claims[claim_id] = Claim(claim_id, node_id, self.round, str(claim.get("text", "")).strip(), evidence)
        elif result.get("claims"):
            warnings.append(f"{participant.code}: new claims are only accepted in round 1; {len(result['claims'])} ignored.")

        if self.kind == "challenges":
            warnings += self._record_reviews(node_id, result.get("reviews") or [])
        elif result.get("reviews"):
            warnings.append(f"{participant.code}: reviews are only accepted in challenge rounds; ignored.")

        if self.kind == "replies":
            warnings += self._record_responses(node_id, result.get("responses") or [])
        elif result.get("responses"):
            warnings.append(f"{participant.code}: responses are only accepted in reply rounds; ignored.")
        return warnings

    def _record_reviews(self, node_id: str, reviews: list[dict[str, Any]]) -> list[str]:
        code = self.participants[node_id].code
        warnings = []
        accepted = 0
        for review in reviews:
            claim_id = normalize_id(review.get("claim_id"))
            claim = self.claims.get(claim_id)
            if claim is None:
                warnings.append(f"{code}: review of unknown claim {claim_id!r} discarded.")
                continue
            if claim.by == node_id:
                warnings.append(f"{code}: cannot review its own claim {claim_id}.")
                continue
            if self.round == 4 and claim_id not in self.disputed_claims:
                warnings.append(f"{code}: round 4 only reviews disputed claims; {claim_id} discarded.")
                continue
            if accepted >= MAX_REVIEWS_PER_ROUND:
                warnings.append(f"{code}: more than {MAX_REVIEWS_PER_ROUND} reviews; the rest were discarded.")
                break
            verdict = review.get("verdict")
            evidence = [str(e) for e in review.get("evidence") or [] if str(e).strip()]
            reason = str(review.get("reason", "")).strip()
            if verdict == "challenge" and not evidence:
                warnings.append(f"{code}: challenge of {claim_id} without evidence discarded.")
                continue
            accepted += 1
            claim.reviews.append({"by": node_id, "verdict": verdict, "reason": reason, "evidence": evidence, "round": self.round})
            if verdict == "challenge":
                challenge_id = f"C{len(self.challenges) + 1}"
                self.challenges[challenge_id] = Challenge(challenge_id, node_id, claim_id, claim.by, self.round, reason, evidence)
        return warnings

    def _record_responses(self, node_id: str, responses: list[dict[str, Any]]) -> list[str]:
        code = self.participants[node_id].code
        warnings = []
        mine = {c.challenge_id: c for c in self.open_challenges_for(node_id)}
        for response in responses:
            challenge = mine.get(normalize_id(response.get("challenge_id")))
            if challenge is None:
                warnings.append(f"{code}: response to {response.get('challenge_id')!r}, which is not an open challenge to it, discarded.")
                continue
            action = response.get("action")
            challenge.status = {"defend": "defended", "concede": "conceded", "clarify": "clarified"}.get(action, "unanswered")
            challenge.response = str(response.get("text", "")).strip()
            challenge.response_evidence = [str(e) for e in response.get("evidence") or [] if str(e).strip()]
        return warnings

    # ------------------------------------------------------------------ end of a pass

    def finish_pass(self) -> bool:
        """Close the current round (once) and decide whether another pass is needed; advances the round if so."""
        if self._pass_closed_for_round == self.round:
            return self.next_round_needed
        self._pass_closed_for_round = self.round
        if self.kind == "replies":
            for challenge in self.challenges.values():
                if challenge.status == "open" and challenge.round == self.round - 1:
                    challenge.status = "unanswered"
        need = False
        if self.round == 1:
            need = len(self.order) > 1
        elif self.round in (2, 4):
            need = any(c.round == self.round for c in self.challenges.values())
        elif self.round == 3:
            self.cycles_done = 1
            disputed = {
                c.claim_id
                for c in self.challenges.values()
                if c.status == "defended" and (self.participants[c.target].kind == "tool_agent" or self.participants[c.by].kind == "tool_agent")
            }
            need = not self.converged() and bool(disputed) and self.max_cycles > 1
            self.disputed_claims = disputed if need else set()
        elif self.round == 5:
            self.cycles_done = 2
        if need:
            self.events.append(f"Round {self.round} closed; round {self.round + 1} ({ROUND_KINDS[self.round + 1]}) follows.")
            self.round += 1
        else:
            self.events.append(f"Round {self.round} closed; the debate ends.")
        self.next_round_needed = need
        return need

    # ------------------------------------------------------------------ agreement

    def final_positions(self) -> dict[str, dict[str, Any]]:
        return {node: positions[-1] for node, positions in self.positions.items() if positions}

    def converged(self) -> bool:
        positions = [self.final_positions().get(node) for node in self.tool_agents()]
        if len(positions) < 2 or any(p is None for p in positions):
            return False
        a, b = positions[0], positions[1]
        if self.task == "sex":
            return a.get("maturity") == b.get("maturity") and a.get("sex_call") == b.get("sex_call")
        try:
            close = abs(float(a["rings"]) - float(b["rings"])) <= AGE_TOLERANCE_RINGS
            overlap = float(a["rings_low"]) <= float(b["rings_high"]) and float(b["rings_low"]) <= float(a["rings_high"])
        except (KeyError, TypeError, ValueError):
            return False
        return close and overlap

    # ------------------------------------------------------------------ rendering

    def roster(self) -> str:
        return "\n".join(f"- {p.code} ({p.label}): {p.card}" for p in self.participants.values())

    def render_state(self, node_id: str) -> str:
        """The debate_state section as one participant must see it before its turn."""
        me = self.participants[node_id]
        lines = [
            f"Task: {self.task}. Round {self.round}: {self.kind.upper()}. You are {me.code} ({me.label}).",
            "",
            "What to do in this round:",
            *[f"- {line}" for line in self.instructions(node_id)],
            "",
            "Participants:",
            self.roster(),
        ]
        positions = self.final_positions()
        if positions:
            lines += ["", "Latest positions of the tool agents:"]
            for node, position in positions.items():
                lines.append(f"- {self.participants[node].code} (round {position['round']}): {self.position_text(position)}")
        if self.expectations:
            lines += ["", "Expectations of the article experts:"]
            lines += [f"- {self.participants[n].code}: {text}" for n, text in self.expectations.items()]
        if self.claims:
            lines += ["", "Claims (id, author, text, evidence, reviews so far):"]
            for claim in self.claims.values():
                author = self.participants[claim.by].code
                lines.append(f"- [{claim.claim_id}] {author}: {claim.text} Evidence: {', '.join(claim.evidence)}")
                for review in claim.reviews:
                    lines.append(f"    - {review['verdict']} by {self.participants[review['by']].code}: {review['reason']}")
        if self.challenges:
            lines += ["", "Challenges:"]
            for c in self.challenges.values():
                target_note = " (ADDRESSED TO YOU)" if c.target == node_id and c.status == "open" else ""
                reply = f" Reply ({c.status}): {c.response}" if c.response else f" Status: {c.status}."
                lines.append(
                    f"- [{c.challenge_id}]{target_note} {self.participants[c.by].code} challenges [{c.claim_id}]: {c.reason} "
                    f"Evidence: {', '.join(c.evidence)}.{reply}"
                )
        if self.abstained:
            lines += ["", "Abstained (their articles do not cover the specimen): " + ", ".join(self.participants[n].code for n in self.abstained)]
        return "\n".join(lines)

    def instructions(self, node_id: str) -> list[str]:
        me = self.participants[node_id]
        if self.kind == "opinions":
            if me.kind == "tool_agent":
                return [
                    "Call each of your two tools once on the specimen, with the dataset-unit values from problem_spec.",
                    "Fill 'position' and give at most 4 claims; each claim's evidence names the tool and the value it returned.",
                    "If other participants already spoke this round, say in a claim whether you agree with them and why.",
                    "Leave 'reviews' and 'responses' empty.",
                ]
            return [
                "Search your article for what it says about a specimen like this one and about the assumptions of the tools listed below.",
                "Use the original-unit values from problem_spec when you compare with your article.",
                "At least one claim must compare a figure from your article with this specimen (for example its length in mm "
                "against a size at maturity, or its ring count against an ageing result) and say what that implies for it.",
                "Give at most 4 claims; the evidence of each claim is the list of chunk_ids you cite. Fill 'expectation' with one cited sentence about this specimen.",
                "If your article does not cover anything useful here, set status to 'abstained' and say 'The article does not cover this topic.'",
                "Leave 'reviews' and 'responses' empty.",
            ]
        if self.kind == "challenges":
            scope = "only the disputed claims " + ", ".join(sorted(self.disputed_claims)) if self.round == 4 else "claims made by other participants"
            evidence = "a tool result" if me.kind == "tool_agent" else "chunk_ids from your article"
            steps = [
                f"Review {scope}: at most {MAX_REVIEWS_PER_ROUND} reviews, each 'confirm' or 'challenge' with a reason and evidence ({evidence}).",
                "Challenge only what your evidence contradicts; confirm what it supports; skip what it does not touch.",
                "Leave 'claims' and 'responses' empty.",
            ]
            if me.kind == "tool_agent":
                steps.append("Repeat your current 'position'; change it only if a claim cites evidence your tools cannot refute.")
            return steps
        steps = [
            "Answer every challenge marked ADDRESSED TO YOU: 'defend' (with evidence), 'concede' (and correct your position) or 'clarify'.",
            "Do not open new challenges and do not add new claims.",
        ]
        if me.kind == "tool_agent":
            steps.append("Fill 'position' with your final position after these replies.")
        return steps

    def position_text(self, position: dict[str, Any]) -> str:
        if self.task == "sex":
            return f"maturity {position.get('maturity')}, sex_call {position.get('sex_call')}, reliability {position.get('reliability')}. {position.get('rationale', '')}".strip()
        return (
            f"rings {position.get('rings')} (80% range {position.get('rings_low')}-{position.get('rings_high')}), "
            f"reliability {position.get('reliability')}. {position.get('rationale', '')}"
        ).strip()

    def ledger(self) -> list[dict[str, Any]]:
        rows = []
        for claim in self.claims.values():
            challenges = [c for c in self.challenges.values() if c.claim_id == claim.claim_id]
            confirms = [self.participants[r["by"]].code for r in claim.reviews if r["verdict"] == "confirm"]
            if challenges:
                status = ", ".join(sorted({c.status for c in challenges}))
            else:
                status = "confirmed" if confirms else "unchallenged"
            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "by": self.participants[claim.by].code,
                    "text": claim.text,
                    "evidence": claim.evidence,
                    "confirmed_by": confirms,
                    "challenges": [
                        {
                            "challenge_id": c.challenge_id,
                            "by": self.participants[c.by].code,
                            "reason": c.reason,
                            "evidence": c.evidence,
                            "status": c.status,
                            "response": c.response,
                            "response_evidence": c.response_evidence,
                        }
                        for c in challenges
                    ],
                    "status": status,
                }
            )
        return rows

    def ledger_markdown(self) -> str:
        lines = ["| Id | By | Claim | Evidence | Challenged by | Reply | Status |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in self.ledger():
            challenged = "; ".join(f"{c['by']} ({c['challenge_id']}): {c['reason']}" for c in row["challenges"]) or (
                "confirmed by " + ", ".join(row["confirmed_by"]) if row["confirmed_by"] else "-"
            )
            replies = "; ".join(f"{c['challenge_id']} {c['status']}: {c['response']}" for c in row["challenges"] if c["response"]) or "-"
            cells = [row["claim_id"], row["by"], row["text"], ", ".join(row["evidence"]), challenged, replies, row["status"]]
            lines.append("| " + " | ".join(cell.replace("|", "/").replace("\n", " ") for cell in cells) + " |")
        positions = self.final_positions()
        lines += ["", "Final positions of the tool agents:"]
        lines += [f"- {self.participants[n].code}: {self.position_text(p)}" for n, p in positions.items()] or ["- none"]
        lines.append(f"\nTool agents agree: {'yes' if self.converged() else 'no'}.")
        if self.expectations:
            lines += ["", "Expectations of the article experts:"]
            lines += [f"- {self.participants[n].code}: {t}" for n, t in self.expectations.items()]
        if self.abstained:
            lines.append("\nAbstained: " + ", ".join(self.participants[n].code for n in self.abstained))
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "round": self.round,
            "cycles_done": self.cycles_done,
            "converged": self.converged(),
            "abstained": sorted(self.participants[n].code for n in self.abstained),
            "positions": {self.participants[n].code: p for n, p in self.positions.items()},
            "expectations": {self.participants[n].code: t for n, t in self.expectations.items()},
            "ledger": self.ledger(),
            "events": self.events,
        }

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, indent=2)
