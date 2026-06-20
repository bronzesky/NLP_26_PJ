"""
src/polish_advisor.py

Generate tiered natural-language polishing suggestions from linguistic feature deviations.
Four tiers: lexical → syntactic → discourse → pragmatic.
Output includes a Composite Prompt ready to paste directly into an AI.
"""
from __future__ import annotations

import re
from typing import Optional


# ── Discourse markers to detect ───────────────────────────────────────────────
_DISCOURSE_MARKERS = {
    "additive": ["furthermore", "moreover", "additionally", "in addition",
                 "besides", "likewise"],
    "contrastive": ["however", "nevertheless", "on the other hand",
                    "in contrast", "nonetheless"],
    "causal": ["therefore", "thus", "consequently", "as a result",
               "because of this", "hence"],
    "conclusive": ["in conclusion", "to summarize", "in summary",
                   "overall", "in short", "to conclude", "to sum up"],
}
_ALL_MARKERS = [m for ms in _DISCOURSE_MARKERS.values() for m in ms]

_HEDGES = [
    "it seems", "it appears", "it is suggested", "research indicates",
    "studies show", "it is worth noting", "it should be noted",
    "one might argue", "it could be argued", "generally speaking",
    "it is generally", "it has been shown",
]

_CONTRACTION_RE = re.compile(
    r"\b(don't|doesn't|didn't|won't|wouldn't|can't|couldn't|"
    r"isn't|aren't|wasn't|weren't|I'm|I've|I'll|it's|that's|"
    r"they're|we're|you're|he's|she's)\b",
    re.IGNORECASE,
)


def _find_discourse_markers_in_text(text: str) -> list[str]:
    """Return the distinct discourse markers actually found in the text."""
    text_lower = text.lower()
    found = []
    for marker in _ALL_MARKERS:
        if marker in text_lower and marker not in found:
            found.append(marker)
    return found


def _find_hedges_in_text(text: str) -> list[str]:
    text_lower = text.lower()
    return [h for h in _HEDGES if h in text_lower]


def generate_suggestions(
    features: dict[str, float],
    baselines: dict,
    text: str,
    doc_prob_ai: float,
) -> dict:
    """
    Generate tiered polishing suggestions.

    Returns:
        {
            "tier1_lexical": [...],
            "tier2_syntactic": [...],
            "tier3_discourse": [...],
            "tier4_pragmatic": [...],
            "composite_prompt": str,
            "priority_features": [...],  # top 3 most AI-like features
        }
    """
    tier1, tier2, tier3, tier4 = [], [], [], []
    priority = []

    def ai_deviation(feat: str) -> float:
        """How much closer to AI avg than human avg (0=human, 1=AI, >1=more AI than AI avg)."""
        if feat not in features or feat not in baselines:
            return 0.0
        val = features[feat]
        b = baselines[feat]
        h, a = b.get("human_mean", 0), b.get("ai_mean", 0)
        if abs(a - h) < 1e-6:
            return 0.0
        return (val - h) / (a - h)  # 0=human, 1=AI

    # ── Tier 3: Discourse (most impactful for AI detection) ────────────────────
    disc_dev = ai_deviation("discourse_total_density")
    if disc_dev > 0.5:
        found_markers = _find_discourse_markers_in_text(text)
        if found_markers:
            marker_list = ", ".join(f'"{m}"' for m in found_markers[:6])
            tier3.append({
                "feature": "discourse_total_density",
                "deviation": round(disc_dev, 2),
                "suggestion": (
                    f"Remove or restructure sentences using: {marker_list}. "
                    "These connective phrases are the strongest AI signal in this text. "
                    "Let ideas connect implicitly — start the next sentence directly "
                    "without announcing the transition."
                ),
                "prompt_fragment": (
                    f"Do NOT use these transition phrases: {marker_list}. "
                    "Instead, let each paragraph flow into the next without explicit connectors."
                ),
            })
            priority.append(("discourse_total_density", disc_dev))

    struct_dev = ai_deviation("structural_completeness")
    if struct_dev > 0.6:
        tier3.append({
            "feature": "structural_completeness",
            "deviation": round(struct_dev, 2),
            "suggestion": (
                "The text follows a rigid 5-paragraph essay structure: "
                "every paragraph has a topic sentence, evidence, and transition. "
                "Break this pattern: let one paragraph be deliberately short (1-2 sentences), "
                "or end the piece mid-thought without a formal conclusion paragraph."
            ),
            "prompt_fragment": (
                "Do not write a formal conclusion paragraph. "
                "Allow one body paragraph to remain underdeveloped. "
                "End where the last substantive point ends."
            ),
        })

    # ── Tier 4: Pragmatic (deep human-like signals) ────────────────────────────
    cont_dev = ai_deviation("contraction_ratio")
    if cont_dev < -0.3:  # much less than AI average (AI avoids contractions)
        # flip: for contractions, LOWER = more AI
        human_mean = baselines.get("contraction_ratio", {}).get("human_mean", 0.005)
        curr = features.get("contraction_ratio", 0)
        if curr < human_mean * 0.4:
            tier4.append({
                "feature": "contraction_ratio",
                "deviation": round(abs(cont_dev), 2),
                "suggestion": (
                    f"No contractions found (current rate: {curr:.3f}, "
                    f"human average: {human_mean:.3f}). "
                    "Add contractions where they sound natural: "
                    "don't, it's, I've, can't, we're, I'm, that's, they're."
                ),
                "prompt_fragment": (
                    "Use contractions freely throughout: don't instead of do not, "
                    "it's instead of it is, I've instead of I have, "
                    "can't instead of cannot, we're instead of we are."
                ),
            })
            priority.append(("contraction_ratio", abs(ai_deviation("contraction_ratio"))))

    fp_dev = ai_deviation("first_person_ratio")
    if fp_dev < -0.4:
        human_mean = baselines.get("first_person_ratio", {}).get("human_mean", 0.008)
        curr = features.get("first_person_ratio", 0)
        if curr < human_mean * 0.3:
            tier4.append({
                "feature": "first_person_ratio",
                "deviation": round(abs(fp_dev), 2),
                "suggestion": (
                    f"No first-person voice found (current: {curr:.3f}, "
                    f"human average: {human_mean:.3f}). "
                    "Add 1-2 sentences with personal perspective: "
                    "'I think...', 'In my experience...', 'Personally, I find...', "
                    "'I'm not entirely sure, but...'"
                ),
                "prompt_fragment": (
                    "Include at least one sentence with 'I think' or 'I find' "
                    "that expresses a genuine personal opinion with some uncertainty. "
                    "Add one specific concrete detail that implies personal experience."
                ),
            })
            priority.append(("first_person_ratio", abs(ai_deviation("first_person_ratio"))))

    hedge_dev = ai_deviation("hedge_position_std")
    if hedge_dev < -0.3:
        tier4.append({
            "feature": "hedge_position_std",
            "deviation": round(abs(hedge_dev), 2),
            "suggestion": (
                "Hedging phrases are clustered only at the start/end of the text "
                "(typical AI pattern). Human writers hedge where they're actually uncertain. "
                "Move one qualification mid-argument: "
                "'though I'm not certain this applies universally', "
                "'this might be wrong in edge cases'."
            ),
            "prompt_fragment": (
                "Add one qualification mid-argument (not at the end) "
                "where you express genuine uncertainty: "
                "'though I'm not entirely sure about this', 'this may not apply everywhere'."
            ),
        })

    # ── Tier 2: Syntactic ──────────────────────────────────────────────────────
    cv_dev = ai_deviation("sentence_length_cv")
    if cv_dev < -0.4:
        curr_cv = features.get("sentence_length_cv", 0)
        human_cv = baselines.get("sentence_length_cv", {}).get("human_mean", 0.5)
        tier2.append({
            "feature": "sentence_length_cv",
            "deviation": round(abs(cv_dev), 2),
            "suggestion": (
                f"Sentence length is too uniform (CV={curr_cv:.2f}, "
                f"human average={human_cv:.2f}). "
                "AI text has rhythmically consistent sentences. "
                "After every 2-3 medium sentences, add either: "
                "a short sentence under 8 words making a direct point, OR "
                "a long sentence over 35 words with embedded clauses."
            ),
            "prompt_fragment": (
                "Vary sentence length significantly: "
                "after every 2-3 regular sentences, add one sentence under 8 words. "
                "Occasionally use a longer sentence over 35 words with subordinate clauses."
            ),
        })
        priority.append(("sentence_length_cv", abs(cv_dev)))

    passive_dev = ai_deviation("passive_ratio")
    if passive_dev > 0.5:
        curr_p = features.get("passive_ratio", 0)
        tier2.append({
            "feature": "passive_ratio",
            "deviation": round(passive_dev, 2),
            "suggestion": (
                f"Passive voice rate is high ({curr_p:.1%}). "
                "Convert passive constructions to active: "
                "'it was found that X' → 'X showed that', "
                "'results were obtained' → 'we obtained results'."
            ),
            "prompt_fragment": (
                "Prefer active voice. Rephrase passive constructions: "
                "'it was found' → name who found it, "
                "'was demonstrated' → 'showed', 'were observed' → 'we observed'."
            ),
        })

    # ── Tier 1: Lexical ────────────────────────────────────────────────────────
    mattr_dev = ai_deviation("mattr")
    if mattr_dev < -0.3:
        curr_m = features.get("mattr", 0)
        human_m = baselines.get("mattr", {}).get("human_mean", 0.78)
        tier1.append({
            "feature": "mattr",
            "deviation": round(abs(mattr_dev), 2),
            "suggestion": (
                f"Vocabulary diversity is low (MATTR={curr_m:.3f}, "
                f"human average={human_m:.3f}). "
                "AI text reuses the same structural phrases repeatedly. "
                "Vary your word choices: if you've used 'important' twice, "
                "use 'crucial', 'worth noting', or 'significant' instead."
            ),
            "prompt_fragment": (
                "Avoid repeating the same adjectives or structural phrases. "
                "If a word or phrase appears more than once, "
                "replace subsequent uses with alternatives."
            ),
        })

    lat_dev = ai_deviation("latinate_ratio")
    if lat_dev > 0.5:
        tier1.append({
            "feature": "latinate_ratio",
            "deviation": round(lat_dev, 2),
            "suggestion": (
                "Too many Latinate/formal words (words ending in -tion, -ment, -ance, etc.). "
                "Human writing mixes formal and informal vocabulary naturally. "
                "Replace some: 'implementation' → 'how it works', "
                "'examination' → 'look at', 'utilization' → 'use'."
            ),
            "prompt_fragment": (
                "Replace some Latinate words with simpler alternatives: "
                "implementation → how it works, utilization → use, "
                "examination → look at, demonstration → show."
            ),
        })

    # ── Build Composite Prompt ─────────────────────────────────────────────────
    all_fragments = []
    section_order = [
        ("[LEXICAL]", tier1),
        ("[SYNTACTIC]", tier2),
        ("[DISCOURSE]", tier3),
        ("[PRAGMATIC]", tier4),
    ]
    for label, tier in section_order:
        for item in tier:
            if "prompt_fragment" in item:
                all_fragments.append(f"{label} {item['prompt_fragment']}")

    if all_fragments:
        composite = (
            "Rewrite the following text to sound more naturally human-written. "
            "Apply ALL of these changes:\n\n"
            + "\n\n".join(all_fragments)
            + "\n\nText to rewrite:\n{TEXT}"
        )
    else:
        composite = (
            "This text already shows relatively human-like writing patterns. "
            "Minor suggestion: add one personal observation or concrete example "
            "that only a direct participant would know."
        )

    # Sort priority features by deviation magnitude
    priority.sort(key=lambda x: -x[1])

    return {
        "tier1_lexical": tier1,
        "tier2_syntactic": tier2,
        "tier3_discourse": tier3,
        "tier4_pragmatic": tier4,
        "composite_prompt": composite,
        "priority_features": priority[:3],
        "total_suggestions": len(tier1) + len(tier2) + len(tier3) + len(tier4),
    }
