"""Rule-based extractor: deterministic, no LLM required.

Every user sentence is matched against a small table of patterns. The best
(strongest) match wins; identity matches resolve the user's name so later
sentences can use it as the relationship subject. Weak matches are still
reported as ``should_remember=False`` decisions so the engine can log what
it deliberately ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .entities import extract_entities
from .classifier import classify_sentence
from .scoring import confidence_for, importance_for

from .relations import DEFAULT_SUBJECT, build_relationships

from .schemas import Conversation, MemoryCandidate, MemoryCategory, MemoryDecision


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _clean_object(obj: str) -> str:
    obj = obj.strip().strip(",.!?;:").strip()
    if obj.lower().startswith("the "):
        obj = obj[4:].strip()
    return obj


@dataclass(frozen=True)
class Rule:
    category: MemoryCategory
    regex: str
    relation: Optional[str] = None
    strength: float = 0.6
    slot: Optional[str] = None


RULES: List[Rule] = [
    # identity ---------------------------------------------------------
    Rule(MemoryCategory.IDENTITY, r"\bmy name(?:'s| is) (?P<obj>[A-Z][\w.'-]+)", strength=0.95, slot="name"),
    Rule(
        MemoryCategory.IDENTITY,
        r"\bi am (?!(?:super|very|really|so|extremely|quite|pretty|fairly|rather|somewhat|"
        r"happy|sad|tired|excited|angry|glad|upset|worried|nervous|anxious|confused|"
        r"surprised|bored|busy|free|ready|done|finished|good|bad|fine|okay|ok|well|"
        r"ill|sick|great|awesome|terrible|horrible|wonderful|amazing|fantastic)\b)"
        r"(?P<obj>[A-Z][\w.'-]+)",
        strength=0.85,
        slot="name",
    ),
    Rule(MemoryCategory.IDENTITY, r"\bi am from (?P<obj>[\w .'-]+)", strength=0.80, slot="origin"),
    Rule(MemoryCategory.IDENTITY, r"\bi live in (?P<obj>[\w .'-]+)", strength=0.80, slot="city"),
    Rule(
        MemoryCategory.IDENTITY,
        r"\bi am an? (?P<obj>(?:\w+ ){0,2}(?:developer|engineer|architect|"
        r"designer|scientist|researcher|student|consultant|manager))",
        strength=0.75,
        slot="role",
    ),
    Rule(MemoryCategory.IDENTITY, r"\bi work at (?P<obj>[\w .'-]+)", strength=0.75, slot="employer"),
    # preference -------------------------------------------------------
    Rule(
        MemoryCategory.PREFERENCE,
        r"\bi (?:really |truly |absolutely )?(?:like|love|enjoy) "
        r"(?P<obj>[\w ,'&.-]+?)(?:[.!?]| and | but | because |,)",
        strength=0.80,
    ),
    Rule(
        MemoryCategory.PREFERENCE,
        r"\bi prefer (?P<obj>[\w ,'&.-]+?)(?:[.!?]| over | to )",
        strength=0.80,
    ),
    Rule(
        MemoryCategory.PREFERENCE,
        r"\bi (?:do not|don't) (?:really )?like (?P<obj>[\w ,'&.-]+?)(?:[.!?]| and | but | because )",
        strength=0.70,
    ),
    # skill ------------------------------------------------------------
    Rule(
        MemoryCategory.SKILL,
        r"\bi work with (?P<obj>[\w .+#&'-]+?)(?:[.!?]| (?:on|for|at|and) )",
        strength=0.80,
    ),
    Rule(
        MemoryCategory.SKILL,
        r"\bi use (?P<obj>[\w .+#&'-]+?)(?:[.!?]| (?:for|to) )",
        strength=0.75,
    ),
    Rule(
        MemoryCategory.SKILL,
        r"\bi ?(?:am|'m) good at (?P<obj>[\w ,'&.-]+?)(?:[.!?]| and )",
        strength=0.80,
    ),
    Rule(
        MemoryCategory.SKILL,
        r"\bi have experience (?:working )?with (?P<obj>[\w .+#&'-]+?)(?:[.!?]| and )",
        strength=0.80,
    ),
    Rule(
        MemoryCategory.SKILL,
        r"\bi ?(?:am|'m) familiar with (?P<obj>[\w .+#&'-]+?)(?:[.!?]| and )",
        strength=0.70,
    ),
    # goal -------------------------------------------------------------
    Rule(
        MemoryCategory.GOAL,
        r"\bi (?:want|would like) to (?P<obj>\w[\w ]{1,50}?)(?:[.!?]| (?:this|that) )",
        strength=0.75,
    ),
    Rule(MemoryCategory.GOAL, r"\bmy goal is to (?P<obj>\w[\w ]{1,50}?)(?:[.!?])", strength=0.80),
    Rule(
        MemoryCategory.GOAL,
        r"\bi ?(?:am|'m) (?:trying|planning|hoping) to (?P<obj>\w[\w ]{1,50}?)(?:[.!?])",
        strength=0.70,
    ),
    # project ----------------------------------------------------------
    Rule(
        MemoryCategory.PROJECT,
        r"\bi ?(?:am|'m) (?:currently )?(?:building|developing|working on) "
        r"(?P<obj>[\w .+'&-]+?)(?:[.!?]| with | using | for )",
        strength=0.80,
    ),
    Rule(
        MemoryCategory.PROJECT,
        r"\bmy (?:current |side |main )?project (?:is )?(?P<obj>[\w .+'&-]+?)(?:[.!?]|,| which )",
        strength=0.80,
    ),
    # fact -------------------------------------------------------------
    Rule(
        MemoryCategory.FACT,
        r"\bi (?:have|own) (?P<obj>[\w ,'&-]+?)(?:[.!?]| and )",
        strength=0.60,
    ),
    Rule(MemoryCategory.FACT, r"\bi ?(?:am|'m) (?P<obj>\d+ years old)", strength=0.70),
    Rule(
        MemoryCategory.FACT,
        r"\bmy (?:email|github|website|linkedin)(?: is)? (?P<obj>[\w@.:/-]+)",
        strength=0.75,
    ),
    # event ------------------------------------------------------------
    Rule(
        MemoryCategory.EVENT,
        r"\bi (?:just |recently )?(?:created|finished|completed|started|built|wrote|"
        r"released|shipped|deployed|launched) (?P<obj>[\w .+'&-]+?)(?:[.!?]| and )",
        strength=0.60,
    ),
    # task -------------------------------------------------------------
    Rule(
        MemoryCategory.TASK,
        r"\bi (?:need|have|still have) to (?P<obj>\w[\w ]{1,50}?)(?:[.!?]| before | after )",
        strength=0.50,
    ),
    Rule(MemoryCategory.TASK, r"\bi should (?P<obj>\w[\w ]{1,50}?)(?:[.!?])", strength=0.40),
    # opinion ----------------------------------------------------------
    Rule(
        MemoryCategory.OPINION,
        r"\b(?:in my opinion|i think|i believe|i feel(?: that)?) (?P<obj>[\w ,.'-]{3,80}?)(?:[.!?])",
        strength=0.50,
    ),
    # knowledge --------------------------------------------------------
    Rule(
        MemoryCategory.KNOWLEDGE,
        r"\b(?:according to|the documentation says|the docs say|research shows) "
        r"(?P<obj>[\w ,.'-]{3,80}?)(?:[.!?])",
        strength=0.55,
    ),
]

_SUMMARY_TEMPLATES = {
    MemoryCategory.IDENTITY: "User identity: {obj}",
    MemoryCategory.PREFERENCE: "User prefers {obj}",
    MemoryCategory.SKILL: "User skill: {obj}",
    MemoryCategory.GOAL: "User goal: {obj}",
    MemoryCategory.PROJECT: "User project: {obj}",
    MemoryCategory.FACT: "User fact: {obj}",
    MemoryCategory.EVENT: "User event: {obj}",
    MemoryCategory.TASK: "User task: {obj}",
    MemoryCategory.OPINION: "User opinion: {obj}",
    MemoryCategory.KNOWLEDGE: "User knowledge: {obj}",
}


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


class RuleBasedExtractor:
    """Deterministic ``MemoryExtractor`` built on the rule table above."""

    def __init__(self, min_strength: float = 0.55) -> None:
        self.min_strength = min_strength
        self._compiled: List[Tuple[Rule, re.Pattern]] = [
            (rule, re.compile(rule.regex, re.IGNORECASE)) for rule in RULES
        ]

    # ------------------------------------------------------------------ #
    # Contract                                                           #
    # ------------------------------------------------------------------ #

    def extract(self, conversation: Conversation) -> List[MemoryDecision]:
        decisions: List[MemoryDecision] = []
        subject: Optional[str] = None
        seen: set = set()

        for turn in conversation.turns:
            if turn.role != "user":
                continue
            for sentence in split_sentences(turn.content):
                match = self._best_match(sentence)
                if match is None:
                    self._emit_keyword_signal(sentence, subject, decisions, seen)
                    continue
                rule, groups = match
                obj = _clean_object(groups.get("obj", ""))

                entities = [e for e in extract_entities(sentence) if e not in ("User",)]
                if rule.category is MemoryCategory.IDENTITY and obj:
                    subject = obj
                    if obj not in entities:
                        entities.append(obj)
                elif obj and obj not in entities and _looks_named(obj):
                    entities.append(obj)

                has_entities = bool(entities)
                slots = (
                    {"identity": {rule.slot: obj}}
                    if rule.slot and obj
                    else None
                )
                candidate = MemoryCandidate(
                    text=sentence,
                    summary=_SUMMARY_TEMPLATES[rule.category].format(obj=obj) if obj else None,
                    category=rule.category,
                    importance=importance_for(rule.category, has_entities),
                    confidence=confidence_for(rule.strength, has_entities),
                    entities=entities,
                    relationships=build_relationships(
                        rule.category, obj, subject=subject or DEFAULT_SUBJECT
                    ),
                    subject=subject,
                    reasoning=f"rule:{rule.category.value}:{rule.regex[:40]}",
                    slots=slots,
                )

                key = sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                decisions.append(
                    MemoryDecision(
                        candidate=candidate,
                        should_remember=rule.strength >= self.min_strength,
                        reason=(
                            "matched rule "
                            f"{rule.category.value} (strength {rule.strength})"
                            if rule.strength >= self.min_strength
                            else f"low signal (strength {rule.strength})"
                        ),
                    )
                )
        return decisions

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _best_match(self, sentence: str) -> Optional[Tuple[Rule, Dict[str, str]]]:
        best: Optional[Tuple[Rule, Dict[str, str]]] = None
        for rule, pattern in self._compiled:
            match = pattern.search(sentence)
            if match is None:
                continue
            groups = {k: v for k, v in match.groupdict().items() if v}
            if best is None or rule.strength > best[0].strength:
                best = (rule, groups)
        return best

    def _emit_keyword_signal(
        self,
        sentence: str,
        subject: Optional[str],
        decisions: List[MemoryDecision],
        seen: set,
    ) -> None:
        """Sentences with only a weak keyword signal are explicitly NOT remembered."""
        category = classify_sentence(sentence)
        if category is None:
            return
        key = sentence.lower()
        if key in seen:
            return
        seen.add(key)
        entities = [e for e in extract_entities(sentence) if e not in ("User",)]
        decisions.append(
            MemoryDecision(
                candidate=MemoryCandidate(
                    text=sentence,
                    category=category,
                    importance=importance_for(category, bool(entities)),
                    confidence=confidence_for(0.35, bool(entities)),
                    entities=entities,
                    subject=subject,
                    reasoning="keyword-signal only, no rule matched",
                ),
                should_remember=False,
                reason="low signal: keyword hit without a rule match",
            )
        )


def _looks_named(obj: str) -> bool:
    """Is the object worth an entity node (proper noun or known tech term)?"""
    if not obj:
        return False
    if obj[0].isupper():
        return True
    from .entities import TECH_WHITELIST

    return obj.lower() in TECH_WHITELIST or any(
        term in obj.lower() for term in TECH_WHITELIST
    )


__all__ = ["Rule", "RULES", "RuleBasedExtractor", "split_sentences"]
