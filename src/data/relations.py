#!/usr/bin/env python3
"""
Canonical Relation Definition & Ontology for Centrode.
Defines the top 20 canonical semantic relations with Persian & English labels,
their descriptions, and category mappings for visual knowledge graph completion.
"""

from typing import Dict, List, Set

CANONICAL_RELATIONS: Dict[str, Dict] = {
    "IsA": {
        "en": "is a",
        "fa": "نوعی_از",
        "category": "taxonomic",
        "description": "Hierarchical categorization (e.g. Tehran is a city)"
    },
    "PartOf": {
        "en": "part of",
        "fa": "بخشی_از",
        "category": "structural",
        "description": "Component or sub-element (e.g. Heart part of circulatory system)"
    },
    "HasProperty": {
        "en": "has property",
        "fa": "دارای_ویژگی",
        "category": "attribute",
        "description": "Inherent quality or characteristic (e.g. Earth has gravity)"
    },
    "UsedFor": {
        "en": "used for",
        "fa": "کاربرد_در",
        "category": "functional",
        "description": "Purpose, usage, or utility (e.g. Telescope used for astronomy)"
    },
    "CapableOf": {
        "en": "capable of",
        "fa": "توانایی",
        "category": "functional",
        "description": "Ability or action of an entity (e.g. Bird capable of flying)"
    },
    "AtLocation": {
        "en": "located at",
        "fa": "موقعیت_در",
        "category": "spatial",
        "description": "Physical or logical location (e.g. Brain located in head)"
    },
    "Causes": {
        "en": "causes",
        "fa": "علت",
        "category": "causal",
        "description": "Causal link or consequence (e.g. Virus causes disease)"
    },
    "HasPrerequisite": {
        "en": "requires",
        "fa": "پیش‌نیاز",
        "category": "logical",
        "description": "Prerequisite or necessity (e.g. Learning requires practice)"
    },
    "CreatedBy": {
        "en": "created by",
        "fa": "ساخته_شده_توسط",
        "category": "origin",
        "description": "Author, inventor, or origin (e.g. Theory of relativity created by Einstein)"
    },
    "DerivedFrom": {
        "en": "derived from",
        "fa": "مشتق_از",
        "category": "origin",
        "description": "Etymology or origin (e.g. Gasoline derived from crude oil)"
    },
    "SubfieldOf": {
        "en": "subfield of",
        "fa": "شاخه‌ای_از",
        "category": "taxonomic",
        "description": "Academic or domain hierarchy (e.g. Deep learning subfield of AI)"
    },
    "HasSubfield": {
        "en": "has subfield",
        "fa": "دارای_شاخه",
        "category": "taxonomic",
        "description": "Broad field containing subdomains (e.g. Physics has subfield Quantum mechanics)"
    },
    "Treats": {
        "en": "treats",
        "fa": "درمان_کننده",
        "category": "functional",
        "description": "Medical or technical remedy (e.g. Insulin treats diabetes)"
    },
    "SimilarTo": {
        "en": "similar to",
        "fa": "مشابه_با",
        "category": "relational",
        "description": "Conceptual analogy or similarity (e.g. Brain similar to neural network)"
    },
    "Synonym": {
        "en": "synonym of",
        "fa": "مترادف",
        "category": "linguistic",
        "description": "Identical semantic meaning (e.g. Physician synonym of doctor)"
    },
    "Antonym": {
        "en": "antonym of",
        "fa": "متضاد",
        "category": "linguistic",
        "description": "Opposite semantic meaning (e.g. Cold antonym of hot)"
    },
    "InfluencedBy": {
        "en": "influenced by",
        "fa": "تاثیرپذیر_از",
        "category": "relational",
        "description": "Influence or dependency (e.g. Earth climate influenced by sun)"
    },
    "RelatesTo": {
        "en": "relates to",
        "fa": "مرتبط_با",
        "category": "general",
        "description": "General associative relationship (e.g. Relativity relates to Spacetime)"
    }
}

CANONICAL_RELATION_NAMES: Set[str] = set(CANONICAL_RELATIONS.keys())

# Normalized alias lookup to map noisy ConceptNet relations to clean canonical ones
RELATION_ALIAS_MAP: Dict[str, str] = {
    "isa": "IsA",
    "is_a": "IsA",
    "instanceof": "IsA",
    "instance_of": "IsA",
    "partof": "PartOf",
    "part_of": "PartOf",
    "haspart": "PartOf",
    "has_part": "PartOf",
    "madeof": "PartOf",
    "made_of": "PartOf",
    "hasproperty": "HasProperty",
    "has_property": "HasProperty",
    "hasattribute": "HasProperty",
    "usedfor": "UsedFor",
    "used_for": "UsedFor",
    "capableof": "CapableOf",
    "capable_of": "CapableOf",
    "atlocation": "AtLocation",
    "at_location": "AtLocation",
    "locatedat": "AtLocation",
    "causes": "Causes",
    "causedby": "Causes",
    "hasprerequisite": "HasPrerequisite",
    "has_prerequisite": "HasPrerequisite",
    "createdby": "CreatedBy",
    "created_by": "CreatedBy",
    "derivedfrom": "DerivedFrom",
    "derived_from": "DerivedFrom",
    "subfieldof": "SubfieldOf",
    "subfield_of": "SubfieldOf",
    "hassubfield": "HasSubfield",
    "has_subfield": "HasSubfield",
    "treats": "Treats",
    "similarto": "SimilarTo",
    "similar_to": "SimilarTo",
    "synonym": "Synonym",
    "antonym": "Antonym",
    "relatedto": "RelatesTo",
    "relatesto": "RelatesTo",
    "distinctfrom": "Antonym",
    # Persian aliases
    "نوعی_از": "IsA",
    "بخشی_از": "PartOf",
    "دارای_ویژگی": "HasProperty",
    "کاربرد_در": "UsedFor",
    "توانایی": "CapableOf",
    "موقعیت_در": "AtLocation",
    "علت": "Causes",
    "پیش‌نیاز": "HasPrerequisite",
    "مترادف": "Synonym",
    "متضاد": "Antonym",
    "مرتبط_با": "RelatesTo",
}

def canonicalize_relation(raw_rel: str) -> str:
    """Maps any relation string to a standardized canonical relation."""
    clean = raw_rel.strip().lower().replace("-", "_").replace(" ", "_")
    return RELATION_ALIAS_MAP.get(clean, "")
