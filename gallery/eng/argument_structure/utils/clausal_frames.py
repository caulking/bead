"""Clausal frame mapping for VerbNet patterns.

This module maps VerbNet clausal complement patterns to template structures
based on the MegaAttitude frame inventory. The MegaAttitude project provides
comprehensive coverage of English clause-embedding constructions.

MegaAttitude Frame Types Covered:
1. Finite complements: that/whether + indicative/subjunctive/conditional
2. Non-finite complements: to-infinitive, gerund, perfect infinitive, bare infinitive
3. Wh-complements: finite and infinitival
4. PP complements with clausal objects
5. Null/pro-clausal complements
6. NP complements
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClausalTemplate:
    """Template structure for clausal complements.

    Attributes
    ----------
    frame_type : str
        Type of clausal complement (e.g., "finite_that", "infinitival_to").
    template_string : str
        Template with slot placeholders (e.g., "{subj} {verb} that {comp_clause}").
    slots : dict[str, str]
        Mapping from slot name to slot type/constraint.
    complementizer : str | None
        Optional complementizer (e.g., "that", "whether", "for").
    mood : str | None
        Clause mood (e.g., "indicative", "subjunctive", "conditional").

    Examples
    --------
    >>> template = ClausalTemplate(
    ...     frame_type="finite_that_indicative",
    ...     template_string="{subj} {verb} that {comp_subj} {comp_verb} {comp_obj}",
    ...     slots={"subj": "noun", "verb": "verb", "comp_subj": "noun",
    ...            "comp_verb": "verb", "comp_obj": "noun"},
    ...     complementizer="that",
    ...     mood="indicative"
    ... )
    """

    frame_type: str
    template_string: str
    slots: dict[str, str]
    complementizer: str | None = None
    mood: str | None = None


# megaattitude frame inventory (from latex specification)
MEGAATTITUDE_FRAMES = {
    # 1. finite complements with that
    "that_indicative_past": ClausalTemplate(
        frame_type="finite_that_indicative_past",
        template_string="{subj} {verb} that {comp_subj} {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_subj": "noun",
            "comp_verb": "verb_past",
            "comp_obj": "noun",
        },
        complementizer="that",
        mood="indicative",
    ),
    "that_indicative_future": ClausalTemplate(
        frame_type="finite_that_indicative_future",
        template_string="{subj} {verb} that {comp_subj} would {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_subj": "noun",
            "comp_verb": "verb_base",
            "comp_obj": "noun",
        },
        complementizer="that",
        mood="conditional",
    ),
    "that_subjunctive": ClausalTemplate(
        frame_type="finite_that_subjunctive",
        template_string="{subj} {verb} that {comp_subj} {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_subj": "noun",
            "comp_verb": "verb_subjunctive",  # base form
            "comp_obj": "noun",
        },
        complementizer="that",
        mood="subjunctive",
    ),
    # 2. finite complements with whether
    "whether_indicative_past": ClausalTemplate(
        frame_type="finite_whether_indicative_past",
        template_string="{subj} {verb} whether {comp_subj} {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_subj": "noun",
            "comp_verb": "verb_past",
            "comp_obj": "noun",
        },
        complementizer="whether",
        mood="indicative",
    ),
    "whether_subjunctive": ClausalTemplate(
        frame_type="finite_whether_subjunctive",
        template_string="{subj} {verb} whether {comp_subj} {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_subj": "noun",
            "comp_verb": "verb_subjunctive",
            "comp_obj": "noun",
        },
        complementizer="whether",
        mood="subjunctive",
    ),
    # 3. non-finite complements: gerund
    "gerund": ClausalTemplate(
        frame_type="nonfinite_gerund",
        template_string="{subj} {verb} {comp_verb_gerund} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_verb_gerund": "verb_present_participle",
            "comp_obj": "noun",
        },
        complementizer=None,
        mood=None,
    ),
    # 4. non-finite complements: to-infinitive
    "to_infinitive": ClausalTemplate(
        frame_type="nonfinite_to_infinitive",
        template_string="{subj} {verb} to {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_verb": "verb_base",
            "comp_obj": "noun",
        },
        complementizer="to",
        mood=None,
    ),
    # 5. non-finite complements: to have (perfect infinitive)
    "to_have_infinitive": ClausalTemplate(
        frame_type="nonfinite_to_have",
        template_string="{subj} {verb} to have {comp_verb_participle} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_verb_participle": "verb_past_participle",
            "comp_obj": "noun",
        },
        complementizer="to",
        mood=None,
    ),
    # 6. bare infinitive (e.g., "see someone do something")
    "bare_infinitive": ClausalTemplate(
        frame_type="nonfinite_bare_infinitive",
        template_string="{subj} {verb} {comp_subj} {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_subj": "noun",
            "comp_verb": "verb_base",
            "comp_obj": "noun",
        },
        complementizer=None,
        mood=None,
    ),
    # 7. for-to infinitive
    "for_to_infinitive": ClausalTemplate(
        frame_type="nonfinite_for_to",
        template_string="{subj} {verb} for {comp_subj} to {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_subj": "noun",
            "comp_verb": "verb_base",
            "comp_obj": "noun",
        },
        complementizer="for",
        mood=None,
    ),
    # 8. wh-complements: finite
    "wh_finite": ClausalTemplate(
        frame_type="wh_finite",
        template_string="{subj} {verb} which {wh_noun} {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "wh_noun": "noun",
            "comp_verb": "verb_past",
            "comp_obj": "noun",
        },
        complementizer="which",
        mood="indicative",
    ),
    # 9. wh-complements: infinitival
    "wh_infinitive": ClausalTemplate(
        frame_type="wh_infinitive",
        template_string="{subj} {verb} which {wh_noun} to {comp_verb}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "wh_noun": "noun",
            "comp_verb": "verb_base",
        },
        complementizer="which",
        mood=None,
    ),
    # 10. PP complements with clausal objects
    "about_whether": ClausalTemplate(
        frame_type="pp_about_whether",
        template_string="{subj} {verb} about whether {comp_subj} {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "comp_subj": "noun",
            "comp_verb": "verb_past",
            "comp_obj": "noun",
        },
        complementizer="whether",
        mood="indicative",
    ),
    # 11. dative with finite complement
    "dative_that": ClausalTemplate(
        frame_type="dative_that",
        template_string="{subj} {verb} to {recipient} that {comp_subj} {comp_verb} {comp_obj}",
        slots={
            "subj": "noun",
            "verb": "verb",
            "recipient": "noun",
            "comp_subj": "noun",
            "comp_verb": "verb_past",
            "comp_obj": "noun",
        },
        complementizer="that",
        mood="indicative",
    ),
    # 12. null complement
    "null_complement": ClausalTemplate(
        frame_type="null_complement",
        template_string="{subj} {verb}",
        slots={"subj": "noun", "verb": "verb"},
        complementizer=None,
        mood=None,
    ),
    # 13. pro-clausal "so"
    "so_proclause": ClausalTemplate(
        frame_type="proclause_so",
        template_string="{subj} {verb} so",
        slots={"subj": "noun", "verb": "verb"},
        complementizer="so",
        mood=None,
    ),
}


def map_verbnet_to_clausal_templates(
    frame_primary: str, syntax_elements: list[tuple[str, str | None]] | None = None
) -> list[ClausalTemplate]:
    """Map VerbNet frame patterns to MegaAttitude clausal templates.

    Parameters
    ----------
    frame_primary
        Primary VerbNet frame description (e.g., "NP V that S", "NP V NP to VP").
    syntax_elements
        Optional detailed syntax from VerbNet frame.

    Returns
    -------
    list[ClausalTemplate]
        Matching clausal templates (may return multiple variants).

    Examples
    --------
    >>> templates = map_verbnet_to_clausal_templates("NP V that S")
    >>> len(templates) > 0
    True
    >>> templates[0].complementizer
    'that'
    """
    templates = []

    frame_lower = frame_primary.lower()

    # finite complements with "that"
    if "that s" in frame_lower or "that-s" in frame_lower:
        templates.extend(
            [
                MEGAATTITUDE_FRAMES["that_indicative_past"],
                MEGAATTITUDE_FRAMES["that_indicative_future"],
                MEGAATTITUDE_FRAMES["that_subjunctive"],
            ]
        )

    # finite complements with "whether"
    if "whether" in frame_lower:
        if "to" in frame_lower:
            # "whether to VP" - different pattern
            templates.append(
                ClausalTemplate(
                    frame_type="whether_to_infinitive",
                    template_string="{subj} {verb} whether to {comp_verb} {comp_obj}",
                    slots={
                        "subj": "noun",
                        "verb": "verb",
                        "comp_verb": "verb_base",
                        "comp_obj": "noun",
                    },
                    complementizer="whether",
                    mood=None,
                )
            )
        else:
            templates.extend(
                [
                    MEGAATTITUDE_FRAMES["whether_indicative_past"],
                    MEGAATTITUDE_FRAMES["whether_subjunctive"],
                ]
            )

    # infinitival "to VP"
    if ("to vp" in frame_lower or "to inf" in frame_lower) and "for" not in frame_lower:
        templates.append(MEGAATTITUDE_FRAMES["to_infinitive"])

    # for-to infinitive
    if "for np to" in frame_lower or "for-np-to-inf" in frame_lower:
        templates.append(MEGAATTITUDE_FRAMES["for_to_infinitive"])

    # gerund
    if "v-ing" in frame_lower or "ving" in frame_lower or "ing" in frame_lower:
        templates.append(MEGAATTITUDE_FRAMES["gerund"])

    # wh-complements
    if any(wh in frame_lower for wh in ["wh-", "which", "what", "who", "how"]):
        if "to" in frame_lower:
            templates.append(MEGAATTITUDE_FRAMES["wh_infinitive"])
        else:
            templates.append(MEGAATTITUDE_FRAMES["wh_finite"])

    # small clauses / bare infinitives (e.g., "NP V NP VP")
    if "np v np vp" in frame_lower.replace(
        " ", ""
    ) or "np v np inf" in frame_lower.replace(" ", ""):
        templates.append(MEGAATTITUDE_FRAMES["bare_infinitive"])

    # PP + clausal complement
    if "about" in frame_lower and ("whether" in frame_lower or "wh" in frame_lower):
        templates.append(MEGAATTITUDE_FRAMES["about_whether"])

    # dative + clausal
    if (
        "to np" in frame_lower or "pp.recipient" in frame_lower
    ) and "that" in frame_lower:
        templates.append(MEGAATTITUDE_FRAMES["dative_that"])

    return templates


def get_all_clausal_frame_types() -> dict[str, ClausalTemplate]:
    """Get all available clausal frame templates.

    Returns
    -------
    dict[str, ClausalTemplate]
        Dictionary mapping frame type names to templates.

    Examples
    --------
    >>> frames = get_all_clausal_frame_types()
    >>> "that_indicative_past" in frames
    True
    """
    return MEGAATTITUDE_FRAMES.copy()


def is_clausal_frame(frame_primary: str) -> bool:
    """Check if a VerbNet frame contains clausal complements.

    Parameters
    ----------
    frame_primary
        VerbNet frame description.

    Returns
    -------
    bool
        True if frame contains clausal patterns.

    Examples
    --------
    >>> is_clausal_frame("NP V that S")
    True
    >>> is_clausal_frame("NP V NP")
    False
    """
    clausal_keywords = [
        "that",
        "whether",
        "if",
        " s",  # sentence
        "to vp",
        "to inf",
        "v-ing",
        "ving",
        "wh-",
        "which",
        "what",
        "who",
        "how",
        "for np to",
        "scomp",
        "vcomp",
    ]

    frame_lower = frame_primary.lower()
    return any(keyword in frame_lower for keyword in clausal_keywords)
