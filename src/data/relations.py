#!/usr/bin/env python3
"""
Canonical Relation Definition & Ontology for Centrode.
Defines 32 high-precision canonical semantic relations with:
  1. Natural language verbalizer templates (Persian & English)
  2. Bidirectional inverse relation pairs
  3. Category mappings and rich alias dictionary
"""

from typing import Dict, List, Set, Tuple, Optional
import re

PERSIAN_CHAR_REGEX = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

def is_persian_text(text: str) -> bool:
    """Returns True if text contains Persian/Arabic characters."""
    return bool(PERSIAN_CHAR_REGEX.search(text))

CANONICAL_RELATIONS: Dict[str, Dict] = {
    # 1. Taxonomic & Categorization
    "IsA": {
        "en_label": "is a",
        "fa_label": "نوعی_از",
        "en_template": "{head} is a type of",
        "fa_template": "{head} نوعی از",
        "inverse": "HasInstance",
        "category": "taxonomic",
        "description": "Hierarchical categorization (e.g. Tehran is a city)"
    },
    "HasInstance": {
        "en_label": "has instance",
        "fa_label": "دارای_نمونه",
        "en_template": "{head} includes instance",
        "fa_template": "{head} شامل نمونه",
        "inverse": "IsA",
        "category": "taxonomic",
        "description": "Category containing instances (e.g. City includes Tehran)"
    },
    "SubfieldOf": {
        "en_label": "subfield of",
        "fa_label": "شاخه‌ای_از",
        "en_template": "{head} is a subfield of",
        "fa_template": "{head} شاخه‌ای از",
        "inverse": "HasSubfield",
        "category": "taxonomic",
        "description": "Academic or domain hierarchy (e.g. Deep learning subfield of AI)"
    },
    "HasSubfield": {
        "en_label": "has subfield",
        "fa_label": "دارای_شاخه",
        "en_template": "{head} has subfield",
        "fa_template": "{head} دارای شاخه",
        "inverse": "SubfieldOf",
        "category": "taxonomic",
        "description": "Domain containing subdisciplines (e.g. Physics has subfield Quantum mechanics)"
    },
    "DefinedAs": {
        "en_label": "defined as",
        "fa_label": "تعریف_شده_به_عنوان",
        "en_template": "{head} is defined as",
        "fa_template": "{head} تعریف می‌شود به عنوان",
        "inverse": "DefinitionOf",
        "category": "conceptual",
        "description": "Exact definition or identity (e.g. Velocity is defined as rate of change)"
    },
    "DefinitionOf": {
        "en_label": "definition of",
        "fa_label": "تعریف_برای",
        "en_template": "{head} is the definition of",
        "fa_template": "{head} تعریفی است برای",
        "inverse": "DefinedAs",
        "category": "conceptual",
        "description": "Inverse definition mapping"
    },

    # 2. Structural & Compositional (Mereology)
    "PartOf": {
        "en_label": "part of",
        "fa_label": "بخشی_از",
        "en_template": "{head} is a part of",
        "fa_template": "{head} بخشی از",
        "inverse": "HasPart",
        "category": "structural",
        "description": "Component of a system (e.g. Heart part of circulatory system)"
    },
    "HasPart": {
        "en_label": "has part",
        "fa_label": "دارای_بخش",
        "en_template": "{head} contains part",
        "fa_template": "{head} دارای بخش",
        "inverse": "PartOf",
        "category": "structural",
        "description": "Entity containing sub-components (e.g. Car has part engine)"
    },
    "MadeOf": {
        "en_label": "made of",
        "fa_label": "ساخته_شده_از",
        "en_template": "{head} is made of",
        "fa_template": "{head} ساخته شده از",
        "inverse": "MaterialFor",
        "category": "structural",
        "description": "Material or physical composition (e.g. Table made of wood)"
    },
    "MaterialFor": {
        "en_label": "material for",
        "fa_label": "ماده_اولیه_برای",
        "en_template": "{head} is a material for",
        "fa_template": "{head} ماده اولیه است برای",
        "inverse": "MadeOf",
        "category": "structural",
        "description": "Material used in construction (e.g. Wood material for furniture)"
    },
    "ConsistsOf": {
        "en_label": "consists of",
        "fa_label": "تشکیل_شده_از",
        "en_template": "{head} consists of",
        "fa_template": "{head} تشکیل شده از",
        "inverse": "ElementOf",
        "category": "structural",
        "description": "Substances or conceptual members (e.g. Water consists of hydrogen and oxygen)"
    },
    "ElementOf": {
        "en_label": "element of",
        "fa_label": "عنصر_تشکیل_دهنده",
        "en_template": "{head} is an element of",
        "fa_template": "{head} عنصر تشکیل‌دهنده است در",
        "inverse": "ConsistsOf",
        "category": "structural",
        "description": "Member or constituent element"
    },

    # 3. Attributes & Properties
    "HasProperty": {
        "en_label": "has property",
        "fa_label": "دارای_ویژگی",
        "en_template": "{head} has the property of",
        "fa_template": "{head} دارای ویژگی",
        "inverse": "PropertyOf",
        "category": "attribute",
        "description": "Inherent quality or characteristic (e.g. Earth has gravity)"
    },
    "PropertyOf": {
        "en_label": "property of",
        "fa_label": "ویژگی_مربوط_به",
        "en_template": "{head} is a property of",
        "fa_template": "{head} ویژگی است برای",
        "inverse": "HasProperty",
        "category": "attribute",
        "description": "Property attributed to an entity (e.g. Gravity is property of mass)"
    },

    # 4. Functional & Utility
    "UsedFor": {
        "en_label": "used for",
        "fa_label": "کاربرد_در",
        "en_template": "{head} is used for",
        "fa_template": "{head} کاربرد دارد در",
        "inverse": "UsedBy",
        "category": "functional",
        "description": "Purpose, usage, or utility (e.g. Telescope used for astronomy)"
    },
    "UsedBy": {
        "en_label": "used by",
        "fa_label": "مورد_استفاده_توسط",
        "en_template": "{head} is used by",
        "fa_template": "{head} مورد استفاده است توسط",
        "inverse": "UsedFor",
        "category": "functional",
        "description": "Entity employing tool or instrument"
    },
    "CapableOf": {
        "en_label": "capable of",
        "fa_label": "توانایی_انجام",
        "en_template": "{head} is capable of",
        "fa_template": "{head} توانایی انجام",
        "inverse": "CapabilityOf",
        "category": "functional",
        "description": "Action or behavior entity can perform (e.g. Bird capable of flying)"
    },
    "CapabilityOf": {
        "en_label": "capability of",
        "fa_label": "توانایی_مربوط_به",
        "en_template": "{head} is a capability of",
        "fa_template": "{head} توانایی است مربوط به",
        "inverse": "CapableOf",
        "category": "functional",
        "description": "Action associated with agent"
    },
    "Produces": {
        "en_label": "produces",
        "fa_label": "تولید_می‌کند",
        "en_template": "{head} produces",
        "fa_template": "{head} تولید می‌کند",
        "inverse": "ProducedBy",
        "category": "functional",
        "description": "Output or production (e.g. Tree produces oxygen)"
    },
    "ProducedBy": {
        "en_label": "produced by",
        "fa_label": "تولید_شده_توسط",
        "en_template": "{head} is produced by",
        "fa_template": "{head} تولید شده توسط",
        "inverse": "Produces",
        "category": "functional",
        "description": "Origin of production"
    },
    "Treats": {
        "en_label": "treats",
        "fa_label": "درمان_می‌کند",
        "en_template": "{head} treats",
        "fa_template": "{head} درمان می‌کند",
        "inverse": "TreatedBy",
        "category": "functional",
        "description": "Remedy or solution (e.g. Insulin treats diabetes)"
    },
    "TreatedBy": {
        "en_label": "treated by",
        "fa_label": "درمان_شده_با",
        "en_template": "{head} is treated by",
        "fa_template": "{head} درمان می‌شود با",
        "inverse": "Treats",
        "category": "functional",
        "description": "Condition managed by remedy"
    },

    # 5. Causal, Process & Temporal
    "Causes": {
        "en_label": "causes",
        "fa_label": "علت_ایجاد",
        "en_template": "{head} causes",
        "fa_template": "{head} باعث ایجاد",
        "inverse": "CausedBy",
        "category": "causal",
        "description": "Causal link or consequence (e.g. Friction causes heat)"
    },
    "CausedBy": {
        "en_label": "caused by",
        "fa_label": "ایجاد_شده_توسط",
        "en_template": "{head} is caused by",
        "fa_template": "{head} ایجاد شده توسط",
        "inverse": "Causes",
        "category": "causal",
        "description": "Consequence resulting from cause"
    },
    "HasPrerequisite": {
        "en_label": "requires",
        "fa_label": "پیش‌نیاز_دارد",
        "en_template": "{head} requires prerequisite",
        "fa_template": "{head} پیش‌نیاز دارد",
        "inverse": "PrerequisiteFor",
        "category": "causal",
        "description": "Logical or physical requirement (e.g. Fire requires oxygen)"
    },
    "PrerequisiteFor": {
        "en_label": "prerequisite for",
        "fa_label": "پیش‌نیاز_برای",
        "en_template": "{head} is a prerequisite for",
        "fa_template": "{head} پیش‌نیاز است برای",
        "inverse": "HasPrerequisite",
        "category": "causal",
        "description": "Requirement enabling an outcome"
    },
    "Enables": {
        "en_label": "enables",
        "fa_label": "امکان‌پذیر_می‌کند",
        "en_template": "{head} enables",
        "fa_template": "{head} امکان‌پذیر می‌کند",
        "inverse": "EnabledBy",
        "category": "causal",
        "description": "Condition enabling capability (e.g. Internet enables remote work)"
    },
    "EnabledBy": {
        "en_label": "enabled by",
        "fa_label": "امکان‌پذیر_شده_با",
        "en_template": "{head} is enabled by",
        "fa_template": "{head} امکان‌پذیر شده با",
        "inverse": "Enables",
        "category": "causal",
        "description": "Capability enabled by technology"
    },
    "Prevents": {
        "en_label": "prevents",
        "fa_label": "جلوگیری_می‌کند_از",
        "en_template": "{head} prevents",
        "fa_template": "{head} جلوگیری می‌کند از",
        "inverse": "PreventedBy",
        "category": "causal",
        "description": "Inhibitor or preventive barrier (e.g. Vaccine prevents infection)"
    },
    "PreventedBy": {
        "en_label": "prevented by",
        "fa_label": "جلوگیری_شده_توسط",
        "en_template": "{head} is prevented by",
        "fa_template": "{head} جلوگیری شده توسط",
        "inverse": "Prevents",
        "category": "causal",
        "description": "Condition prevented by action"
    },
    "Precedes": {
        "en_label": "precedes",
        "fa_label": "قبل_از",
        "en_template": "{head} precedes",
        "fa_template": "{head} رخ می‌دهد قبل از",
        "inverse": "Follows",
        "category": "temporal",
        "description": "Temporal order (e.g. Design precedes implementation)"
    },
    "Follows": {
        "en_label": "follows",
        "fa_label": "بعد_از",
        "en_template": "{head} follows",
        "fa_template": "{head} رخ می‌دهد بعد از",
        "inverse": "Precedes",
        "category": "temporal",
        "description": "Temporal succession (e.g. Implementation follows design)"
    },

    # 6. Spatial & Locational
    "AtLocation": {
        "en_label": "located at",
        "fa_label": "موقعیت_در",
        "en_template": "{head} is located at",
        "fa_template": "{head} قرار دارد در",
        "inverse": "LocationOf",
        "category": "spatial",
        "description": "Spatial location (e.g. CPU located at motherboard)"
    },
    "LocationOf": {
        "en_label": "location of",
        "fa_label": "مکان_برای",
        "en_template": "{head} is the location of",
        "fa_template": "{head} مکانی است برای",
        "inverse": "AtLocation",
        "category": "spatial",
        "description": "Location housing entities"
    },
    "LocatedIn": {
        "en_label": "located in",
        "fa_label": "واقع_در",
        "en_template": "{head} is located in",
        "fa_template": "{head} واقع شده درون",
        "inverse": "ContainsLocation",
        "category": "spatial",
        "description": "Geographical inclusion (e.g. Paris located in France)"
    },
    "ContainsLocation": {
        "en_label": "contains location",
        "fa_label": "شامل_مکان",
        "en_template": "{head} contains location",
        "fa_template": "{head} دربرگیرنده مکان",
        "inverse": "LocatedIn",
        "category": "spatial",
        "description": "Region containing places"
    },
    "AdjacentTo": {
        "en_label": "adjacent to",
        "fa_label": "مجاور_با",
        "en_template": "{head} is adjacent to",
        "fa_template": "{head} مجاور است با",
        "inverse": "AdjacentTo",
        "category": "spatial",
        "description": "Symmetric adjacency (e.g. Canada adjacent to USA)"
    },

    # 7. Origin, Agency & Association
    "CreatedBy": {
        "en_label": "created by",
        "fa_label": "ساخته_شده_توسط",
        "en_template": "{head} was created by",
        "fa_template": "{head} ساخته شده توسط",
        "inverse": "CreatorOf",
        "category": "origin",
        "description": "Author, creator, or inventor (e.g. General Relativity created by Einstein)"
    },
    "CreatorOf": {
        "en_label": "creator of",
        "fa_label": "سازنده",
        "en_template": "{head} is the creator of",
        "fa_template": "{head} سازنده است برای",
        "inverse": "CreatedBy",
        "category": "origin",
        "description": "Agent responsible for creation"
    },
    "DerivedFrom": {
        "en_label": "derived from",
        "fa_label": "مشتق_از",
        "en_template": "{head} is derived from",
        "fa_template": "{head} مشتق شده از",
        "inverse": "SourceOf",
        "category": "origin",
        "description": "Etymology or origin (e.g. Gasoline derived from crude oil)"
    },
    "SourceOf": {
        "en_label": "source of",
        "fa_label": "منبع_برای",
        "en_template": "{head} is the source of",
        "fa_template": "{head} منبع است برای",
        "inverse": "DerivedFrom",
        "category": "origin",
        "description": "Origin producing derivation"
    },
    "InfluencedBy": {
        "en_label": "influenced by",
        "fa_label": "تاثیرپذیر_از",
        "en_template": "{head} is influenced by",
        "fa_template": "{head} تاثیرپذیر از",
        "inverse": "Influences",
        "category": "relational",
        "description": "Influence or dependency (e.g. Modern physics influenced by Newton)"
    },
    "Influences": {
        "en_label": "influences",
        "fa_label": "تاثیرگذار_بر",
        "en_template": "{head} influences",
        "fa_template": "{head} تاثیرگذار است بر",
        "inverse": "InfluencedBy",
        "category": "relational",
        "description": "Agent or concept shaping others"
    },
    "OwnedBy": {
        "en_label": "owned by",
        "fa_label": "متعلق_به",
        "en_template": "{head} is owned by",
        "fa_template": "{head} متعلق است به",
        "inverse": "Owns",
        "category": "relational",
        "description": "Ownership or possession"
    },
    "Owns": {
        "en_label": "owns",
        "fa_label": "مالک",
        "en_template": "{head} owns",
        "fa_template": "{head} مالک است برای",
        "inverse": "OwnedBy",
        "category": "relational",
        "description": "Owner of property"
    },
    "DependsOn": {
        "en_label": "depends on",
        "fa_label": "وابسته_به",
        "en_template": "{head} depends on",
        "fa_template": "{head} وابسته است به",
        "inverse": "DependedOnBy",
        "category": "relational",
        "description": "Technical or logical dependency (e.g. Flutter depends on Dart)"
    },
    "DependedOnBy": {
        "en_label": "depended on by",
        "fa_label": "مورد_نیاز_برای",
        "en_template": "{head} is depended on by",
        "fa_template": "{head} مورد نیاز است برای",
        "inverse": "DependsOn",
        "category": "relational",
        "description": "Prerequisite base supporting dependencies"
    },
    "InheritsFrom": {
        "en_label": "inherits from",
        "fa_label": "ارث‌بری_از",
        "en_template": "{head} inherits from",
        "fa_template": "{head} ارث‌بری دارد از",
        "inverse": "InheritedBy",
        "category": "structural",
        "description": "OOP or genetic inheritance"
    },
    "InheritedBy": {
        "en_label": "inherited by",
        "fa_label": "ارث‌بری_شده_توسط",
        "en_template": "{head} is inherited by",
        "fa_template": "{head} ارث‌بری شده توسط",
        "inverse": "InheritsFrom",
        "category": "structural",
        "description": "Base entity giving inheritance"
    },

    # 8. Human & Psychological
    "Desires": {
        "en_label": "desires",
        "fa_label": "تمایل_به",
        "en_template": "{head} desires",
        "fa_template": "{head} تمایل دارد به",
        "inverse": "DesiredBy",
        "category": "cognitive",
        "description": "Goal or desire of conscious agent (e.g. Person desires happiness)"
    },
    "DesiredBy": {
        "en_label": "desired by",
        "fa_label": "مطلوب_برای",
        "en_template": "{head} is desired by",
        "fa_template": "{head} مطلوب است برای",
        "inverse": "Desires",
        "category": "cognitive",
        "description": "Object of aspiration"
    },
    "MotivatedBy": {
        "en_label": "motivated by",
        "fa_label": "با_انگیزه",
        "en_template": "{head} is motivated by",
        "fa_template": "{head} با انگیزه",
        "inverse": "Motivates",
        "category": "cognitive",
        "description": "Motivation behind action"
    },
    "Motivates": {
        "en_label": "motivates",
        "fa_label": "انگیزه_می‌دهد_به",
        "en_template": "{head} motivates",
        "fa_template": "{head} انگیزه می‌دهد به",
        "inverse": "MotivatedBy",
        "category": "cognitive",
        "description": "Factor driving behavior"
    },
    "SymbolOf": {
        "en_label": "symbol of",
        "fa_label": "نماد",
        "en_template": "{head} is a symbol of",
        "fa_template": "{head} نمادی است از",
        "inverse": "SymbolizedBy",
        "category": "conceptual",
        "description": "Symbolic representation (e.g. Dove symbol of peace)"
    },
    "SymbolizedBy": {
        "en_label": "symbolized by",
        "fa_label": "نمادگذاری_شده_با",
        "en_template": "{head} is symbolized by",
        "fa_template": "{head} نمادگذاری شده با",
        "inverse": "SymbolOf",
        "category": "conceptual",
        "description": "Concept represented by symbol"
    },

    # 9. Linguistic & Comparative
    "SimilarTo": {
        "en_label": "similar to",
        "fa_label": "مشابه_با",
        "en_template": "{head} is similar to",
        "fa_template": "{head} مشابه است با",
        "inverse": "SimilarTo",
        "category": "linguistic",
        "description": "Symmetric similarity (e.g. Brain similar to computer)"
    },
    "Synonym": {
        "en_label": "synonym of",
        "fa_label": "مترادف",
        "en_template": "{head} is a synonym of",
        "fa_template": "{head} مترادف است با",
        "inverse": "Synonym",
        "category": "linguistic",
        "description": "Symmetric synonymy (e.g. Fast synonym of quick)"
    },
    "Antonym": {
        "en_label": "antonym of",
        "fa_label": "متضاد",
        "en_template": "{head} is the antonym of",
        "fa_template": "{head} متضاد است با",
        "inverse": "Antonym",
        "category": "linguistic",
        "description": "Symmetric antonymy (e.g. Cold antonym of hot)"
    },
    "DistinctFrom": {
        "en_label": "distinct from",
        "fa_label": "متمایز_از",
        "en_template": "{head} is distinct from",
        "fa_template": "{head} متمایز است از",
        "inverse": "DistinctFrom",
        "category": "linguistic",
        "description": "Explicit distinction between commonly confused concepts"
    }
}

CANONICAL_RELATION_NAMES: Set[str] = set(CANONICAL_RELATIONS.keys())

# Comprehensive alias lookup table mapping noisy ConceptNet terms & Persian phrases to canonical relations
RELATION_ALIAS_MAP: Dict[str, str] = {
    # IsA & Types
    "isa": "IsA",
    "is_a": "IsA",
    "instanceof": "IsA",
    "instance_of": "IsA",
    "typeof": "IsA",
    "type_of": "IsA",
    "hasinstance": "HasInstance",
    "has_instance": "HasInstance",
    
    # PartOf & Mereology
    "partof": "PartOf",
    "part_of": "PartOf",
    "haspart": "HasPart",
    "has_part": "HasPart",
    "madeof": "MadeOf",
    "made_of": "MadeOf",
    "materialfor": "MaterialFor",
    "material_for": "MaterialFor",
    "consistsof": "ConsistsOf",
    "consists_of": "ConsistsOf",
    "elementof": "ElementOf",
    "element_of": "ElementOf",
    
    # Properties
    "hasproperty": "HasProperty",
    "has_property": "HasProperty",
    "hasattribute": "HasProperty",
    "has_attribute": "HasProperty",
    "propertyof": "PropertyOf",
    "property_of": "PropertyOf",
    
    # Definition
    "definedas": "DefinedAs",
    "defined_as": "DefinedAs",
    "definitionof": "DefinitionOf",
    "definition_of": "DefinitionOf",
    
    # Functions & Utility
    "usedfor": "UsedFor",
    "used_for": "UsedFor",
    "usedby": "UsedBy",
    "used_by": "UsedBy",
    "capableof": "CapableOf",
    "capable_of": "CapableOf",
    "capabilityof": "CapabilityOf",
    "capability_of": "CapabilityOf",
    "produces": "Produces",
    "producedby": "ProducedBy",
    "produced_by": "ProducedBy",
    "treats": "Treats",
    "treatedby": "TreatedBy",
    "treated_by": "TreatedBy",
    
    # Causality & Process
    "causes": "Causes",
    "causedby": "CausedBy",
    "caused_by": "CausedBy",
    "hasprerequisite": "HasPrerequisite",
    "has_prerequisite": "HasPrerequisite",
    "prerequisitefor": "PrerequisiteFor",
    "prerequisite_for": "PrerequisiteFor",
    "enables": "Enables",
    "enabledby": "EnabledBy",
    "enabled_by": "EnabledBy",
    "prevents": "Prevents",
    "preventedby": "PreventedBy",
    "prevented_by": "PreventedBy",
    "precedes": "Precedes",
    "follows": "Follows",
    
    # Spatial
    "atlocation": "AtLocation",
    "at_location": "AtLocation",
    "locatedat": "AtLocation",
    "locationof": "LocationOf",
    "location_of": "LocationOf",
    "locatedin": "LocatedIn",
    "located_in": "LocatedIn",
    "containslocation": "ContainsLocation",
    "contains_location": "ContainsLocation",
    "adjacentto": "AdjacentTo",
    "adjacent_to": "AdjacentTo",
    
    # Origin & Agency
    "createdby": "CreatedBy",
    "created_by": "CreatedBy",
    "creatorof": "CreatorOf",
    "creator_of": "CreatorOf",
    "derivedfrom": "DerivedFrom",
    "derived_from": "DerivedFrom",
    "sourceof": "SourceOf",
    "source_of": "SourceOf",
    "influencedby": "InfluencedBy",
    "influenced_by": "InfluencedBy",
    "influences": "Influences",
    "ownedby": "OwnedBy",
    "owned_by": "OwnedBy",
    "owns": "Owns",
    "subfieldof": "SubfieldOf",
    "subfield_of": "SubfieldOf",
    "hassubfield": "HasSubfield",
    "has_subfield": "HasSubfield",
    "dependson": "DependsOn",
    "depends_on": "DependsOn",
    "dependedonby": "DependedOnBy",
    "depended_on_by": "DependedOnBy",
    "inheritsfrom": "InheritsFrom",
    "inherits_from": "InheritsFrom",
    "inheritedby": "InheritedBy",
    "inherited_by": "InheritedBy",
    
    # Cognitive & Symbolic
    "desires": "Desires",
    "desiredby": "DesiredBy",
    "desired_by": "DesiredBy",
    "motivatedby": "MotivatedBy",
    "motivated_by": "MotivatedBy",
    "motivates": "Motivates",
    "symbolof": "SymbolOf",
    "symbol_of": "SymbolOf",
    "symbolizedby": "SymbolizedBy",
    "symbolized_by": "SymbolizedBy",
    
    # Linguistic & Comparative
    "similarto": "SimilarTo",
    "similar_to": "SimilarTo",
    "synonym": "Synonym",
    "antonym": "Antonym",
    "distinctfrom": "DistinctFrom",
    "distinct_from": "DistinctFrom",
    
    # Persian Aliases
    "نوعی_از": "IsA",
    "شامل_نمونه": "HasInstance",
    "بخشی_از": "PartOf",
    "دارای_بخش": "HasPart",
    "ساخته_شده_از": "MadeOf",
    "ماده_اولیه": "MaterialFor",
    "تشکیل_شده_از": "ConsistsOf",
    "عنصر_تشکیل_دهنده": "ElementOf",
    "دارای_ویژگی": "HasProperty",
    "ویژگی_مربوط_به": "PropertyOf",
    "تعریف_به_عنوان": "DefinedAs",
    "کاربرد_در": "UsedFor",
    "مورد_استفاده_توسط": "UsedBy",
    "توانایی": "CapableOf",
    "توانایی_انجام": "CapableOf",
    "تولید_می‌کند": "Produces",
    "تولید_شده_توسط": "ProducedBy",
    "درمان_کننده": "Treats",
    "درمان_می‌کند": "Treats",
    "درمان_شده_با": "TreatedBy",
    "علت": "Causes",
    "علت_ایجاد": "Causes",
    "ایجاد_شده_توسط": "CausedBy",
    "پیش‌نیاز": "HasPrerequisite",
    "پیش‌نیاز_برای": "PrerequisiteFor",
    "امکان‌پذیر_می‌کند": "Enables",
    "جلوگیری_می‌کند": "Prevents",
    "قبل_از": "Precedes",
    "بعد_از": "Follows",
    "موقعیت_در": "AtLocation",
    "مکان_برای": "LocationOf",
    "واقع_در": "LocatedIn",
    "مجاور_با": "AdjacentTo",
    "ساخته_شده_توسط": "CreatedBy",
    "سازنده": "CreatorOf",
    "مشتق_از": "DerivedFrom",
    "منبع_برای": "SourceOf",
    "تاثیرپذیر_از": "InfluencedBy",
    "تاثیرگذار_بر": "Influences",
    "متعلق_به": "OwnedBy",
    "مالک": "Owns",
    "شاخه‌ای_از": "SubfieldOf",
    "دارای_شاخه": "HasSubfield",
    "وابسته_به": "DependsOn",
    "ارث‌بری_از": "InheritsFrom",
    "تمایل_به": "Desires",
    "با_انگیزه": "MotivatedBy",
    "نماد": "SymbolOf",
    "مشابه_با": "SimilarTo",
    "مترادف": "Synonym",
    "متضاد": "Antonym",
    "متمایز_از": "DistinctFrom",
}

def canonicalize_relation(raw_rel: str) -> str:
    """Maps any raw relation string to a standardized canonical relation."""
    clean = raw_rel.strip().lower().replace("-", "_").replace(" ", "_")
    return RELATION_ALIAS_MAP.get(clean, "")

def get_inverse_relation(rel_name: str) -> str:
    """Returns the canonical inverse relation name."""
    info = CANONICAL_RELATIONS.get(rel_name)
    if info and "inverse" in info:
        return info["inverse"]
    return ""

def format_verbalizer_prompt(head: str, rel_name: str) -> str:
    """
    Formats the natural language prompt for (Head + Relation).
    Selects Persian or English template automatically based on script detection.
    """
    info = CANONICAL_RELATIONS.get(rel_name)
    if not info:
        return f"{head} [SEP] {rel_name}"
        
    if is_persian_text(head):
        template = info.get("fa_template", f"{{head}} {info.get('fa_label', rel_name)}")
    else:
        template = info.get("en_template", f"{{head}} {info.get('en_label', rel_name)}")
        
    return template.format(head=head)
