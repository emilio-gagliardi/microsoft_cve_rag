from neomodel import (
    AsyncStructuredNode,
    StringProperty,
    ArrayProperty,
    DateTimeProperty,
    DateTimeFormatProperty,
    JSONProperty,
    FloatProperty,
    IntegerProperty,
    AliasProperty,
    install_all_labels,
)
from neomodel.async_.relationship_manager import (
    AsyncRelationshipFrom,
    AsyncRelationshipTo,
)
import json
import uuid
from datetime import datetime


# =============== BGEIN Relationship Classes

from neomodel.async_.relationship import AsyncStructuredRel


class AsyncMetadataRel(AsyncStructuredRel):
    created_at = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    updated_at = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty())
    relationship_type = StringProperty()


class AsyncMetadataCVERel(AsyncStructuredRel):
    created_at = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    updated_at = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty())
    relationship_type = StringProperty()
    impact_type = StringProperty()
    remediation_level = StringProperty()
    severity_type = StringProperty
    max_severity = StringProperty()


# =============== BGEIN Node Classes


class Product(AsyncStructuredNode):
    name_choices = {
        "windows_10": "Windows 10",
        "windows_11": "Windows 11",
        "edge": "Microsoft Edge (Chromium-based)",
    }
    archi_choices = {"x86": "32-bit system", "x64": "64-bit system"}
    product_versions = {
        "21H2": "Windows 10, Windows 11",
        "22H2": "Windows 10, Windows 11",
        "23H2": "Windows 11",
        "24H2": "Windows 11",
    }
    product_name = StringProperty(required=True, choices=name_choices)
    product_architecture = StringProperty(required=True, choices=archi_choices)
    product_version = StringProperty(required=True, choices=product_versions)
    node_id = StringProperty(default=uuid.uuid4)
    product_id = AliasProperty(to="node_id")
    description = StringProperty()
    node_label = StringProperty()
    build_numbers = StringProperty()
    product_build_ids = ArrayProperty(StringProperty())

    # Relationships
    symptoms_affect = AsyncRelationshipFrom(
        "Symptom", "AFFECTS_PRODUCT", model=AsyncMetadataRel
    )

    has_builds = AsyncRelationshipTo(
        "ProductBuild", "HAS_BUILD", model=AsyncMetadataRel
    )

    def set_build_numbers(self, build_numbers_list):
        """Serialize and set the build numbers as a JSON string."""
        self.build_numbers = json.dumps(build_numbers_list)

    def get_build_numbers(self):
        """Deserialize and return the build numbers as a list of lists."""
        return json.loads(self.build_numbers) if self.build_numbers else []


class ProductBuild(AsyncStructuredNode):
    name_choices = {
        "windows_10": "Windows 10",
        "windows_11": "Windows 11",
        "edge": "Microsoft Edge (Chromium-based)",
    }
    archi_choices = {"x86": "32-bit systems", "x64": "64-bit systems"}
    product_versions = {
        "21H2": "Windows 10, Windows 11",
        "22H2": "Windows 10, Windows 11",
        "23H2": "Windows 11",
        "24H2": "Windows 11",
    }
    impact_type_choices = {
        "tampering": "Tampering",
        "spoofing": "Spoofing",
        "availability": "Availability",
        "privilege elevation": "Elevation of Privilege",
        "dos": "Denial of Service",
        "disclosure": "Information Disclosure",
        "remote_code_execution": "Remote Code Execution",
        "security_feature_bypass": "Security Feature Bypass",
    }

    severity_type_choices = {
        "critical": "Critical",
        "important": "Important",
        "moderate": "Moderate",
        "low": "Low",
    }

    attack_complexity_choices = {"low": "Low", "high": "High"}

    attack_vector_choices = {
        "network": "Network",
        "adjacent_network": "Adjacent Network",
        "local": "Local",
        "physical": "Physical",
    }

    remediation_level_choices = {
        "official_fix": "Official Fix",
        "temporary_fix": "Temporary Fix",
        "workaround": "Workaround",
        "unavailable": "Unavailable",
    }
    product_name = StringProperty(required=True, choices=name_choices)
    product = StringProperty(required=True)
    product_architecture = StringProperty(required=True, choices=archi_choices)
    product_version = StringProperty(required=True, choices=product_versions)
    build_number = ArrayProperty(IntegerProperty())
    cve_id = StringProperty()
    kb_id = StringProperty()
    node_id = StringProperty(required=True)
    product_build_id = AliasProperty(to="node_id")
    published = DateTimeFormatProperty(
        format="%Y-%m-%d %H:%M:%S",
    )
    node_label = StringProperty()
    summary = StringProperty()
    article_url = StringProperty()
    cve_url = StringProperty()
    impact_type = StringProperty(choices=impact_type_choices)
    severity_type = StringProperty(choices=severity_type_choices)
    attack_vector = StringProperty(choices=attack_vector_choices)
    remediation_level = StringProperty(choices=remediation_level_choices)

    # Relationships
    products_have = AsyncRelationshipFrom(
        "Product", "HAS_BUILD", model=AsyncMetadataRel
    )

    references_msrcs = AsyncRelationshipTo(
        "MSRCPost", "REFERENCES", model=AsyncMetadataRel
    )
    references_kbs = AsyncRelationshipTo(
        "KBArticle", "REFERENCES", model=AsyncMetadataRel
    )
    has_update_packages = AsyncRelationshipTo(
        "UpdatePackage", "HAS_UPDATE_PACKAGE", model=AsyncMetadataRel
    )


class MSRCPost(AsyncStructuredNode):
    impact_type_choices = {
        "tampering": "Tampering",
        "spoofing": "Spoofing",
        "availability": "Availability",
        "privilege elevation": "Elevation of Privilege",
        "dos": "Denial of Service",
        "disclosure": "Information Disclosure",
        "remote_code_execution": "Remote Code Execution",
        "security_feature_bypass": "Security Feature Bypass",
    }

    severity_type_choices = {
        "critical": "Critical",
        "important": "Important",
        "moderate": "Moderate",
        "low": "Low",
    }

    attack_complexity_choices = {"low": "Low", "high": "High"}

    attack_vector_choices = {
        "network": "Network",
        "adjacent_network": "Adjacent Network",
        "local": "Local",
        "physical": "Physical",
    }

    remediation_level_choices = {
        "official_fix": "Official Fix",
        "temporary_fix": "Temporary Fix",
        "workaround": "Workaround",
        "unavailable": "Unavailable",
    }
    node_id = StringProperty(required=True)
    mongo_id = AliasProperty(to="node_id")
    text = StringProperty()
    embedding = ArrayProperty(FloatProperty())
    metadata = JSONProperty()
    published = DateTimeFormatProperty(
        format="%Y-%m-%d %H:%M:%S",
    )
    node_label = StringProperty()
    revision = StringProperty()
    title = StringProperty()
    description = StringProperty()
    source = StringProperty()
    impact_type = StringProperty(choices=impact_type_choices)
    severity_type = StringProperty(choices=severity_type_choices)
    post_type = StringProperty()
    attack_complexity = StringProperty(choices=attack_complexity_choices)
    attack_vector = StringProperty(choices=attack_vector_choices)
    exploit_code_maturity = StringProperty()
    exploitability = StringProperty()
    post_id = StringProperty()
    max_severity = StringProperty()
    remediation_level = StringProperty(choices=remediation_level_choices)
    report_confidence = StringProperty()
    summary = StringProperty()
    build_numbers = ArrayProperty(StringProperty())
    # Relationships
    next_version_has = AsyncRelationshipFrom(
        "MSRCPost", "NEXT_VERSION", model=AsyncMetadataRel
    )

    has_symptoms = AsyncRelationshipTo("Symptom", "HAS_SYMPTOM", model=AsyncMetadataRel)
    has_causes = AsyncRelationshipTo("Cause", "HAS_CAUSE", model=AsyncMetadataRel)
    has_fixes = AsyncRelationshipTo("Fix", "HAS_FIX", model=AsyncMetadataRel)
    has_faqs = AsyncRelationshipTo("FAQ", "HAS_FAQ", model=AsyncMetadataRel)
    has_tools = AsyncRelationshipTo("Tool", "HAS_TOOL", model=AsyncMetadataRel)
    has_kb_articles = AsyncRelationshipTo("KBArticle", "HAS_KB", model=AsyncMetadataRel)
    has_update_packages = AsyncRelationshipTo(
        "UpdatePackage", "HAS_UPDATE_PACKAGE", model=AsyncMetadataRel
    )
    previous_version_has = AsyncRelationshipTo(
        "MSRCPost", "PREVIOUS_VERSION", model=AsyncMetadataRel
    )
    affects_products = AsyncRelationshipTo(
        "Product", "AFFECTS_PRODUCT", model=AsyncMetadataRel
    )

    def set_build_numbers(self, build_numbers_list):
        """Serialize and set the build numbers as a JSON string."""
        self.build_numbers = json.dumps(build_numbers_list)

    def get_build_numbers(self):
        """Deserialize and return the build numbers as a list of lists."""
        return json.loads(self.build_numbers) if self.build_numbers else []


class Symptom(AsyncStructuredNode):
    node_id = StringProperty(required=True)
    symptom_id = AliasProperty(to="node_id")
    description = StringProperty(required=True)
    node_label = StringProperty()
    source_id = StringProperty()
    source_type = StringProperty()
    embedding = ArrayProperty(FloatProperty())
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty())
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_SYMPTOM", model=AsyncMetadataRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_SYMPTOM", model=AsyncMetadataRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_SYMPTOM", model=AsyncMetadataRel
    )
    affects_products = AsyncRelationshipTo(
        "Product", "AFFECTS_PRODUCT", model=AsyncMetadataRel
    )


class Cause(AsyncStructuredNode):
    node_id = StringProperty(unique_index=True, default=uuid.uuid4)
    cause_id = AliasProperty(to="node_id")
    description = StringProperty(required=True)
    node_label = StringProperty()
    source_id = StringProperty()
    source_type = StringProperty()
    embedding = ArrayProperty(FloatProperty())
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty())
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_CAUSE", model=AsyncMetadataRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_CAUSE", model=AsyncMetadataRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_CAUSE", model=AsyncMetadataRel
    )


class Fix(AsyncStructuredNode):
    node_id = StringProperty(unique_index=True, default=uuid.uuid4)
    fix_id = AliasProperty(to="node_id")
    description = StringProperty(required=True)
    node_label = StringProperty()
    source_id = StringProperty()
    source_type = StringProperty()
    embedding = ArrayProperty(FloatProperty())
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty())
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_FIX", model=AsyncMetadataRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_FIX", model=AsyncMetadataRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_FIX", model=AsyncMetadataRel
    )
    update_packages_have = AsyncRelationshipFrom(
        "UpdatePackage", "HAS_FIX", model=AsyncMetadataRel
    )

    references_kb_articles = AsyncRelationshipTo(
        "KBArticle", "REFERENCES", model=AsyncMetadataRel
    )
    references_tools = AsyncRelationshipTo("Tool", "REFERENCES", model=AsyncMetadataRel)


class FAQ(AsyncStructuredNode):
    node_id = StringProperty(unique_index=True, default=uuid.uuid4)
    faq_id = AliasProperty(to="node_id")
    node_label = StringProperty()
    question = StringProperty(required=True)
    answer = StringProperty(required=True)
    source_id = StringProperty()
    source_type = StringProperty()
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty())
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_FAQ", model=AsyncMetadataRel
    )


class Tool(AsyncStructuredNode):
    node_id = StringProperty(unique_index=True, default=uuid.uuid4)
    tool_id = AliasProperty(to="node_id")
    name = StringProperty(required=True)
    node_label = StringProperty()
    description = StringProperty(required=True)
    download_url = StringProperty()
    source_id = StringProperty()
    source_type = StringProperty()
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty())
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_TOOL", model=AsyncMetadataRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_TOOL", model=AsyncMetadataRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_TOOL", model=AsyncMetadataRel
    )


class KBArticle(AsyncStructuredNode):
    node_id = StringProperty(required=True)  # id key coming from mongo
    kb_article_id = AliasProperty(to="node_id")
    kb_id = StringProperty()  # 5040442 same in mongo
    build_number = ArrayProperty(IntegerProperty())
    published = DateTimeFormatProperty(
        format="%Y-%m-%d %H:%M:%S",
    )
    node_label = StringProperty()
    product_build_id = StringProperty(required=True)
    article_url = StringProperty()
    title = StringProperty()
    text = StringProperty()
    embedding = ArrayProperty(FloatProperty())
    tags = ArrayProperty(StringProperty())
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_KB", model=AsyncMetadataRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_KB", model=AsyncMetadataRel
    )
    builds_reference = AsyncRelationshipFrom(
        "ProductBuild", "REFERENCES", model=AsyncMetadataRel
    )

    affects_product = AsyncRelationshipTo(
        "Product", "AFFECTS_PRODUCT", model=AsyncMetadataRel
    )
    has_symptoms = AsyncRelationshipTo("Symptom", "HAS_SYMPTOM", model=AsyncMetadataRel)
    has_causes = AsyncRelationshipTo("Cause", "HAS_CAUSE", model=AsyncMetadataRel)
    has_fixes = AsyncRelationshipTo("Fix", "HAS_FIX", model=AsyncMetadataRel)
    has_tools = AsyncRelationshipTo("Tool", "HAS_TOOL", model=AsyncMetadataRel)
    has_update_packages = AsyncRelationshipTo(
        "UpdatePackage", "HAS_UPDATE_PACKAGE", model=AsyncMetadataRel
    )


class UpdatePackage(AsyncStructuredNode):
    package_type_choices = {
        "security_update": "Security Update",
        "security_hotpatch": "Security Hotpatch Update",
        "monthly_rollup": "Monthly Rollup",
        "security_only": "Security Only",
    }
    node_id = StringProperty(required=True)
    update_id = AliasProperty(to="node_id")
    package_type = StringProperty(choices=package_type_choices)
    package_url = StringProperty()
    build_number = ArrayProperty(IntegerProperty())
    product_build_id = StringProperty(required=True)
    downloadable_packages = ArrayProperty(JSONProperty())
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    node_label = StringProperty()
    tags = ArrayProperty(StringProperty())
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_UPDATE_PACKAGE", model=AsyncMetadataRel
    )
    product_builds_have = AsyncRelationshipFrom(
        "ProductBuild", "HAS_UPDATE_PACKAGE", model=AsyncMetadataRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_UPDATE_PACKAGE", model=AsyncMetadataRel
    )


class PatchManagementPost(AsyncStructuredNode):
    node_id = StringProperty(required=True)  # either id_ or metadata.id from mongo
    patch_id = AliasProperty(to="node_id")
    receivedDateTime = StringProperty()
    published = DateTimeFormatProperty(
        format="%Y-%m-%d %H:%M:%S",
    )
    topic = StringProperty()
    subject = StringProperty()
    post_type = StringProperty()
    conversation_link = StringProperty()
    cve_mentions = StringProperty()
    metadata = JSONProperty()
    keywords = ArrayProperty(StringProperty())
    noun_chunks = ArrayProperty(StringProperty())
    embedding = ArrayProperty(FloatProperty())
    text = StringProperty()
    node_label = StringProperty()
    tags = ArrayProperty(StringProperty())
    # Relationships
    next_message_has = AsyncRelationshipFrom(
        "PatchManagementPost", "NEXT_MESSAGE", model=AsyncMetadataRel
    )

    has_symptoms = AsyncRelationshipTo("Symptom", "HAS_SYMPTOM", model=AsyncMetadataRel)
    has_causes = AsyncRelationshipTo("Cause", "HAS_CAUSE", model=AsyncMetadataRel)
    has_fixes = AsyncRelationshipTo("Fix", "HAS_FIX", model=AsyncMetadataRel)
    references_kb_articles = AsyncRelationshipTo(
        "KBArticle", "REFERENCES", model=AsyncMetadataRel
    )
    has_tools = AsyncRelationshipTo("Tool", "HAS_TOOL", model=AsyncMetadataRel)
    previous_message_has = AsyncRelationshipTo(
        "PatchManagementPost", "PREVIOUS_MESSAGE", model=AsyncMetadataRel
    )
    references_msrc_posts = AsyncRelationshipTo(
        "MSRCPost", "REFERENCES", model=AsyncMetadataRel
    )
    affects_products = AsyncRelationshipTo(
        "Product", "AFFECTS_PRODUCT", model=AsyncMetadataRel
    )


# Mapping of labels to their corresponding classes
LABEL_TO_CLASS_MAP = {
    "Product": Product,
    "ProductBuild": ProductBuild,
    "MSRCPost": MSRCPost,
    "Symptom": Symptom,
    "Cause": Cause,
    "Fix": Fix,
    "FAQ": FAQ,
    "Tool": Tool,
    "KBArticle": KBArticle,
    "UpdatePackage": UpdatePackage,
    "PatchManagementPost": PatchManagementPost,
}

# needed if constraints aren't set up
# install_all_labels()
