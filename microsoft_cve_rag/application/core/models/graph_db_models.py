from neomodel import (
    AsyncStructuredNode,
    StringProperty,
    ArrayProperty,
    BooleanProperty,
    DateTimeProperty,
    DateTimeFormatProperty,
    DateTimeNeo4jFormatProperty,
    JSONProperty,
    FloatProperty,
    IntegerProperty,
    AliasProperty,
    install_all_labels,
    # AsyncRelationship,
    # AsyncCardinality,
)
from neomodel.async_.relationship import AsyncStructuredRel
from neomodel.async_.relationship_manager import (
    AsyncRelationshipFrom,
    AsyncRelationshipTo,
)
import json
import uuid
from datetime import datetime


# =============== BGEIN Relationship Classes


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class AsyncZeroToManyRel(AsyncStructuredRel):
    created_at = DateTimeFormatProperty(
        default=lambda: datetime.now(), format="%Y-%m-%d %H:%M:%S"
    )
    updated_at = DateTimeFormatProperty(
        default=lambda: datetime.now(), format="%Y-%m-%d %H:%M:%S"
    )
    tags = ArrayProperty(StringProperty())
    relationship_type = StringProperty()
    relationship_id = StringProperty()


class AsyncOneToOneRel(AsyncStructuredRel):
    created_at = DateTimeFormatProperty(
        default=lambda: datetime.now(), format="%Y-%m-%d %H:%M:%S"
    )
    updated_at = DateTimeFormatProperty(
        default=lambda: datetime.now(), format="%Y-%m-%d %H:%M:%S"
    )
    relationship_type = StringProperty()
    relationship_id = StringProperty()


class AsyncSymptomCauseFixRel(AsyncZeroToManyRel):
    severity = StringProperty(
        choices={"low": "low", "medium": "medium", "high": "high", "important":"important", "nst":"nst", "NST":"NST"}
    )
    confidence = IntegerProperty(default=50)  # 0-100%
    description = StringProperty()
    reported_date = DateTimeFormatProperty(format="%Y-%m-%d")


class AsyncAffectsProductRel(AsyncZeroToManyRel):
    __type__ = "AFFECTS_PRODUCT"
    impact_rating = StringProperty()
    severity = StringProperty()
    affected_versions = ArrayProperty(StringProperty())
    patched_in_version = StringProperty()


class AsyncHasUpdatePackageRel(AsyncZeroToManyRel):
    __type__ = "HAS_UPDATE_PACKAGE"
    release_date = DateTimeFormatProperty(format="%Y-%m-%d")
    has_cumulative = BooleanProperty(default=False)
    has_dynamic = BooleanProperty(default=False)


class AsyncPreviousVersionRel(AsyncOneToOneRel):
    __type__ = "PREVIOUS_VERSION"
    version_difference = StringProperty()
    changes_summary = StringProperty()
    previous_version_id = StringProperty()


class AsyncPreviousMessageRel(AsyncOneToOneRel):
    __type__ = "PREVIOUS_MESSAGE"
    previous_id = StringProperty(default=None)


class AsyncReferencesRel(AsyncZeroToManyRel):
    __type__ = "REFERENCES"
    relevance_score = IntegerProperty()  # 0-100
    context = StringProperty()
    cited_section = StringProperty()


# =============== BEGIN Node Classes ======================


class Product(AsyncStructuredNode):
    name_choices = {
        "windows_10": "Windows 10",
        "windows_11": "Windows 11",
        "edge": "Microsoft Edge (Chromium-based)",
        "edge_ext": "Microsoft Edge (Chromium-based) Extended Edition",
    }
    archi_choices = {
        "x86": "32-bit system",
        "x64": "64-bit system",
        "NA": "No Architecture",
    }
    product_versions = {
        "21H2": "Windows 10, Windows 11",
        "22H2": "Windows 10, Windows 11",
        "23H2": "Windows 11",
        "24H2": "Windows 11",
        "NV": "No Version",
    }
    product_name = StringProperty(required=True, choices=name_choices)
    product_architecture = StringProperty(required=True, choices=archi_choices)
    product_version = StringProperty(required=True, choices=product_versions)
    node_id = StringProperty(default=uuid.uuid4, unique_index=True)
    product_id = AliasProperty(to="node_id")
    description = StringProperty(default="")
    node_label = StringProperty()
    build_numbers = ArrayProperty(StringProperty())
    product_build_ids = ArrayProperty(StringProperty())
    cve_ids = ArrayProperty(StringProperty(), default=[])
    kb_ids = ArrayProperty(StringProperty(), default=[])
    published = DateTimeProperty()
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    # Relationships
    symptoms_affect = AsyncRelationshipFrom(
        "Symptom", "AFFECTS_PRODUCT", model=AsyncSymptomCauseFixRel
    )

    has_builds = AsyncRelationshipTo(
        "ProductBuild", "HAS_BUILD", model=AsyncZeroToManyRel
    )

    class Meta:
        unique_together = [("product_name", "product_version", "product_architecture")]

    def set_build_numbers(self, build_numbers_list):
        """Serialize and set the build numbers as a JSON string."""
        if not build_numbers_list:
            self.build_numbers = []
        else:
            self.build_numbers = [json.dumps(sublist) for sublist in build_numbers_list]

    def get_build_numbers(self):
        """Deserialize and return the build numbers as a list of lists."""
        return (
            [json.loads(sublist) for sublist in self.build_numbers]
            if self.build_numbers
            else []
        )


class ProductBuild(AsyncStructuredNode):
    name_choices = {
        "windows_10": "Windows 10",
        "windows_11": "Windows 11",
        "edge": "Microsoft Edge (Chromium-based)",
        "edge_ext": "Microsoft Edge (Chromium-based) Extended",
    }
    archi_choices = {
        "x86": "32-bit systems",
        "x64": "64-bit systems",
        "NA": "No Architecture",
        "na": "No Architecture",
    }
    product_versions = {
        "21H2": "Windows 10, Windows 11",
        "22H2": "Windows 10, Windows 11",
        "23H2": "Windows 11",
        "24H2": "Windows 11",
        "NV": "No Version",
        "nv": "No Version",
    }
    impact_type_choices = {
        "tampering": "Tampering",
        "spoofing": "Spoofing",
        "availability": "Availability",
        "privilege_elevation": "Elevation of Privilege",
        "denial_of_service": "Denial of Service",
        "disclosure": "Information Disclosure",
        "remote_code_execution": "Remote Code Execution",
        "security_feature_bypass": "Security Feature Bypass",
        "NIT": "No impact type",
        "nit": "No impact type",
    }

    severity_type_choices = {
        "critical": "Critical",
        "important": "Important",
        "moderate": "Moderate",
        "low": "Low",
        "NST": "No Severity Type",
        "nst": "No Severity Type",
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
    product_architecture = StringProperty(required=True, choices=archi_choices)
    product_version = StringProperty(required=True, choices=product_versions)
    build_number = ArrayProperty(IntegerProperty())
    cve_id = StringProperty()
    kb_id = StringProperty()
    node_id = StringProperty(default=lambda: str(uuid.uuid4()), unique_index=True)
    product_build_id = StringProperty(required=True)

    published = DateTimeProperty()
    node_label = StringProperty()
    impact_type = StringProperty(choices=impact_type_choices)
    severity_type = StringProperty(choices=severity_type_choices)
    attack_vector = StringProperty(choices=attack_vector_choices)
    remediation_level = StringProperty(choices=remediation_level_choices)
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    # Relationships
    products_have = AsyncRelationshipFrom(
        "Product", "HAS_BUILD", model=AsyncZeroToManyRel
    )

    references_msrcs = AsyncRelationshipTo(
        "MSRCPost", "REFERENCES", model=AsyncReferencesRel
    )
    references_kbs = AsyncRelationshipTo(
        "KBArticle", "REFERENCES", model=AsyncReferencesRel
    )
    has_update_packages = AsyncRelationshipTo(
        "UpdatePackage", "HAS_UPDATE_PACKAGE", model=AsyncHasUpdatePackageRel
    )

    class Meta:
        unique_together = [
            (
                "product_name",
                "product_version",
                "product_architecture",
                "cve_id",
                "product_build_id",
            )
        ]


class MSRCPost(AsyncStructuredNode):
    cve_category_choices = {
        "tampering": "Tampering",
        "spoofing": "Spoofing",
        "availability": "Availability",
        "privilege_elevation": "Elevation of Privilege",
        "denial_of_service": "Denial of Service",
        "dos": "Denial of Service",
        "disclosure": "Information Disclosure",
        "remote_code_execution": "Remote Code Execution",
        "rce": "Remote Code Execution",
        "feature_bypass": "Security Feature Bypass",
        "NC": "No Category",
        "none": "None",
    }

    severity_type_choices = {
        "critical": "Critical",
        "important": "Important",
        "moderate": "Moderate",
        "low": "Low",
        "NST": "No Severity Type",
        "none": "None",
        "None": "None"
    }

    attack_complexity_choices = {
        "low": "Low",
        "high": "High",
        "none": "None",
        "None": "None",
    }

    attack_vector_choices = {
        "network": "Network",
        "adjacent_network": "Adjacent Network",
        "local": "Local",
        "physical": "Physical",
        "none": "None",
        "None": "None",
    }

    remediation_level_choices = {
        "official_fix": "Official Fix",
        "temporary_fix": "Temporary Fix",
        "workaround": "Workaround",
        "unavailable": "Unavailable",
        "in_the_wild": "In the wild actively being exploited",
    }
    base_score_rating_choices = {
        "none": "None",
        "None": "None",
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low"
    }
    cvss_attack_complexity_choices = {
        "low": "Low",
        "high": "High",
        "none": "None",
        "None": "None",
    }
    cvss_attack_vector_choices = {
        "network": "Network",
        "adjacent": "Adjacent Network",
        "local": "Local",
        "physical": "Physical",
        "none": "None",
        "None": "None",
    }
    cvss_impact_choices = {
        "none": "None",
        "None": "None",
        "low": "Low",
        "high": "High",
    }
    cvss_scope_choices = {
        "unchanged": "Unchanged",
        "changed": "Changed",
        "none": "None",
        "None": "None",
    }
    cvss_user_interaction_choices = {
        "none": "None",
        "None": "None",
        "required": "Required",
    }
    cvss_privileges_required_choices = {
        "none": "None",
        "None": "None",
        "low": "Low",
        "high": "High",
    }
    
    node_id = StringProperty(required=True, unique_index=True)
    msrc_id = AliasProperty(to="node_id")
    text = StringProperty()
    embedding = ArrayProperty(FloatProperty())
    metadata = JSONProperty()
    published = DateTimeProperty()
    node_label = StringProperty()
    revision = StringProperty()
    title = StringProperty()
    description = StringProperty(default="")
    source = StringProperty()
    cve_category = StringProperty(choices=cve_category_choices)
    severity_type = StringProperty(choices=severity_type_choices)
    post_type = StringProperty()
    attack_complexity = StringProperty(choices=attack_complexity_choices)
    attack_vector = StringProperty(choices=attack_vector_choices)
    exploit_code_maturity = StringProperty(default="")
    exploitability = StringProperty(default="")
    post_id = StringProperty()
    max_severity = StringProperty(default="")
    remediation_level = StringProperty(choices=remediation_level_choices)
    report_confidence = StringProperty(default="")
    summary = StringProperty(default="")
    kb_ids = ArrayProperty(StringProperty(), default=[])
    product_build_ids = ArrayProperty(StringProperty(), default=[])
    build_numbers = ArrayProperty(StringProperty(), default=[])
    next_version_id = StringProperty(default="")
    previous_version_id = StringProperty(default="")
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    nvd_published_date = DateTimeProperty()
    nvd_description = StringProperty()
    # NIST CVSS Properties
    nist_attack_complexity = StringProperty(choices=cvss_attack_complexity_choices)
    nist_attack_vector = StringProperty(choices=cvss_attack_vector_choices)
    nist_availability = StringProperty(choices=cvss_impact_choices)
    nist_base_score_num = FloatProperty()
    nist_base_score_rating = StringProperty(choices=base_score_rating_choices)
    nist_confidentiality = StringProperty(choices=cvss_impact_choices)
    nist_exploitability_score = FloatProperty()
    nist_impact_score = FloatProperty()
    nist_integrity = StringProperty(choices=cvss_impact_choices)
    nist_privileges_required = StringProperty(choices=cvss_privileges_required_choices)
    nist_scope = StringProperty(choices=cvss_scope_choices)
    nist_user_interaction = StringProperty(choices=cvss_user_interaction_choices)
    nist_vector = StringProperty()

    # CNA CVSS Properties
    cna_attack_complexity = StringProperty(choices=cvss_attack_complexity_choices)
    cna_attack_vector = StringProperty(choices=cvss_attack_vector_choices)
    cna_availability = StringProperty(choices=cvss_impact_choices)
    cna_base_score_num = FloatProperty()
    cna_base_score_rating = StringProperty(choices=base_score_rating_choices)
    cna_confidentiality = StringProperty(choices=cvss_impact_choices)
    cna_exploitability_score = FloatProperty()
    cna_impact_score = FloatProperty()
    cna_integrity = StringProperty(choices=cvss_impact_choices)
    cna_privileges_required = StringProperty(choices=cvss_privileges_required_choices)
    cna_scope = StringProperty(choices=cvss_scope_choices)
    cna_user_interaction = StringProperty(choices=cvss_user_interaction_choices)
    cna_vector = StringProperty()

    # ADP CVSS Properties
    adp_attack_complexity = StringProperty(choices=cvss_attack_complexity_choices)
    adp_attack_vector = StringProperty(choices=cvss_attack_vector_choices)
    adp_availability = StringProperty(choices=cvss_impact_choices)
    adp_base_score_num = FloatProperty()
    adp_base_score_rating = StringProperty(choices=base_score_rating_choices)
    adp_confidentiality = StringProperty(choices=cvss_impact_choices)
    adp_exploitability_score = FloatProperty()
    adp_impact_score = FloatProperty()
    adp_integrity = StringProperty(choices=cvss_impact_choices)
    adp_privileges_required = StringProperty(choices=cvss_privileges_required_choices)
    adp_scope = StringProperty(choices=cvss_scope_choices)
    adp_user_interaction = StringProperty(choices=cvss_user_interaction_choices)
    adp_vector = StringProperty()

    # CWE Properties
    cwe_id = StringProperty()
    cwe_name = StringProperty()
    cwe_source = StringProperty()
    cwe_url = StringProperty()
    
    # Relationships
    has_symptoms = AsyncRelationshipTo(
        "Symptom", "HAS_SYMPTOM", model=AsyncSymptomCauseFixRel
    )
    has_causes = AsyncRelationshipTo(
        "Cause", "HAS_CAUSE", model=AsyncSymptomCauseFixRel
    )
    has_fixes = AsyncRelationshipTo("Fix", "HAS_FIX", model=AsyncSymptomCauseFixRel)
    has_faqs = AsyncRelationshipTo("FAQ", "HAS_FAQ", model=AsyncZeroToManyRel)
    has_tools = AsyncRelationshipTo("Tool", "HAS_TOOL", model=AsyncZeroToManyRel)
    has_kb_articles = AsyncRelationshipTo(
        "KBArticle", "HAS_KB", model=AsyncZeroToManyRel
    )
    has_update_packages = AsyncRelationshipTo(
        "UpdatePackage", "HAS_UPDATE_PACKAGE", model=AsyncHasUpdatePackageRel
    )
    previous_version_has = AsyncRelationshipTo(
        "MSRCPost", "PREVIOUS_VERSION", model=AsyncPreviousVersionRel
    )
    affects_products = AsyncRelationshipTo(
        "Product", "AFFECTS_PRODUCT", model=AsyncAffectsProductRel
    )

    def set_build_numbers(self, build_numbers_list):
        """Serialize and set the build numbers as a JSON string."""
        if not build_numbers_list:
            self.build_numbers = []
        else:
            self.build_numbers = [json.dumps(sublist) for sublist in build_numbers_list]

    def get_build_numbers(self):
        """Deserialize and return the build numbers as a list of lists."""
        return (
            [json.loads(sublist) for sublist in self.build_numbers]
            if self.build_numbers
            else []
        )


class Symptom(AsyncStructuredNode):
    severity_type_choices = {
        "critical": "Critical",
        "important": "Important",
        "moderate": "Moderate",
        "low": "Low",
        "NST": "No Severity Type",
    }
    node_id = StringProperty(default=uuid.uuid4, unique_index=True)
    symptom_label = StringProperty(unique_index=True)
    description = StringProperty(required=True)
    node_label = StringProperty()
    source_id = StringProperty(default="")
    source_type = StringProperty(default="")
    reliability = StringProperty(default="")
    readability = FloatProperty(default=0.0)
    severity_type = StringProperty(choices=severity_type_choices)
    embedding = ArrayProperty(FloatProperty(), default=None)
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty(), default=None)
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_SYMPTOM", model=AsyncSymptomCauseFixRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_SYMPTOM", model=AsyncSymptomCauseFixRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_SYMPTOM", model=AsyncSymptomCauseFixRel
    )
    affects_products = AsyncRelationshipTo(
        "Product", "AFFECTS_PRODUCT", model=AsyncAffectsProductRel
    )


class Cause(AsyncStructuredNode):
    severity_type_choices = {
        "critical": "Critical",
        "important": "Important",
        "moderate": "Moderate",
        "low": "Low",
        "NST": "No Severity Type",
    }
    node_id = StringProperty(unique_index=True, default=uuid.uuid4)
    cause_id = AliasProperty(to="node_id")
    description = StringProperty(required=True)
    node_label = StringProperty()
    cause_label = StringProperty()
    source_id = StringProperty(default="")
    source_type = StringProperty(default="")
    reliability = StringProperty(default="")
    readability = FloatProperty(default=0.0)
    severity_type = StringProperty(choices=severity_type_choices)
    embedding = ArrayProperty(FloatProperty(), default=None)
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty())
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_CAUSE", model=AsyncSymptomCauseFixRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_CAUSE", model=AsyncSymptomCauseFixRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_CAUSE", model=AsyncSymptomCauseFixRel
    )


class Fix(AsyncStructuredNode):
    severity_type_choices = {
        "critical": "Critical",
        "important": "Important",
        "moderate": "Moderate",
        "low": "Low",
        "NST": "No Severity Type",
    }
    node_id = StringProperty(unique_index=True, default=uuid.uuid4)
    fix_id = AliasProperty(to="node_id")
    description = StringProperty(required=True)
    node_label = StringProperty()
    fix_label = StringProperty()
    source_id = StringProperty(default=None)
    source_type = StringProperty(default=None)
    reliability = StringProperty(default="")
    readability = FloatProperty(default=0.0)
    severity_type = StringProperty(choices=severity_type_choices)
    embedding = ArrayProperty(FloatProperty(), default=None)
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty(), default=[])
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_FIX", model=AsyncSymptomCauseFixRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_FIX", model=AsyncSymptomCauseFixRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_FIX", model=AsyncSymptomCauseFixRel
    )
    update_packages_have = AsyncRelationshipFrom(
        "UpdatePackage", "HAS_FIX", model=AsyncSymptomCauseFixRel
    )

    references_kb_articles = AsyncRelationshipTo(
        "KBArticle", "REFERENCES", model=AsyncReferencesRel
    )
    references_tools = AsyncRelationshipTo(
        "Tool", "REFERENCES", model=AsyncReferencesRel
    )


class FAQ(AsyncStructuredNode):
    node_id = StringProperty(unique_index=True, default=uuid.uuid4)
    faq_id = AliasProperty(to="node_id")
    node_label = StringProperty()
    question = StringProperty(required=True)
    answer = StringProperty(required=True)
    source_id = StringProperty(default="")
    source_type = StringProperty(default="")
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty(), default=[])
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_FAQ", model=AsyncZeroToManyRel
    )


class Tool(AsyncStructuredNode):
    node_id = StringProperty(unique_index=True, default=uuid.uuid4)
    tool_id = AliasProperty(to="node_id")
    tool_label = StringProperty(required=True)
    node_label = StringProperty()
    description = StringProperty(required=True)
    source_url = StringProperty(default="")
    source_id = StringProperty(default="")
    source_type = StringProperty(default="")
    reliability = StringProperty(default="")
    readability = FloatProperty(default=0.0)
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    tags = ArrayProperty(StringProperty(), default=[])
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_TOOL", model=AsyncZeroToManyRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_TOOL", model=AsyncZeroToManyRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_TOOL", model=AsyncZeroToManyRel
    )


class KBArticle(AsyncStructuredNode):
    severity_type_choices = {
        "critical": "Critical",
        "important": "Important",
        "moderate": "Moderate",
        "low": "Low",
        "NST": "No Severity Type",
    }
    node_id = StringProperty(
        required=True, unique_index=True
    )  # id key coming from mongo
    kb_article_id = AliasProperty(to="node_id")
    kb_id = StringProperty()  # 5040442 same in mongo
    build_number = ArrayProperty(IntegerProperty())
    published = DateTimeProperty()
    node_label = StringProperty()
    product_build_id = StringProperty(required=True)
    product_build_ids = ArrayProperty(StringProperty())
    article_url = StringProperty()
    cve_ids = ArrayProperty(StringProperty(), default=[])
    title = StringProperty()
    text = StringProperty()
    severity_type = StringProperty(choices=severity_type_choices)
    embedding = ArrayProperty(FloatProperty(), default=[])
    tags = ArrayProperty(StringProperty(), default=[])
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_KB", model=AsyncZeroToManyRel
    )
    emails_have = AsyncRelationshipFrom(
        "PatchManagementPost", "HAS_KB", model=AsyncZeroToManyRel
    )
    builds_reference = AsyncRelationshipFrom(
        "ProductBuild", "REFERENCES", model=AsyncZeroToManyRel
    )

    affects_product = AsyncRelationshipTo(
        "Product", "AFFECTS_PRODUCT", model=AsyncAffectsProductRel
    )
    has_symptoms = AsyncRelationshipTo(
        "Symptom", "HAS_SYMPTOM", model=AsyncSymptomCauseFixRel
    )
    has_causes = AsyncRelationshipTo(
        "Cause", "HAS_CAUSE", model=AsyncSymptomCauseFixRel
    )
    has_fixes = AsyncRelationshipTo("Fix", "HAS_FIX", model=AsyncSymptomCauseFixRel)
    has_tools = AsyncRelationshipTo("Tool", "HAS_TOOL", model=AsyncZeroToManyRel)
    has_update_packages = AsyncRelationshipTo(
        "UpdatePackage", "HAS_UPDATE_PACKAGE", model=AsyncHasUpdatePackageRel
    )


class UpdatePackage(AsyncStructuredNode):
    package_type_choices = {
        "security_update": "Security Update",
        "security_hotpatch": "Security Hotpatch Update",
        "monthly_rollup": "Monthly Rollup",
        "security_only": "Security Only",
    }
    node_id = StringProperty(required=True, unique_index=True)
    update_id = AliasProperty(to="node_id")
    package_type = StringProperty(choices=package_type_choices)
    package_url = StringProperty()
    build_number = ArrayProperty(IntegerProperty())
    product_build_ids = ArrayProperty(StringProperty(), required=True)
    downloadable_packages = ArrayProperty(StringProperty(), default=[])
    published = DateTimeProperty()
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    node_label = StringProperty()
    tags = ArrayProperty(StringProperty(), default=None)
    # Relationships
    msrc_posts_have = AsyncRelationshipFrom(
        "MSRCPost", "HAS_UPDATE_PACKAGE", model=AsyncHasUpdatePackageRel
    )
    product_builds_have = AsyncRelationshipFrom(
        "ProductBuild", "HAS_UPDATE_PACKAGE", model=AsyncHasUpdatePackageRel
    )
    kb_articles_have = AsyncRelationshipFrom(
        "KBArticle", "HAS_UPDATE_PACKAGE", model=AsyncHasUpdatePackageRel
    )

    def set_downloadable_packages(self, packages_list):
        """
        Serialize each package dictionary to a JSON string and set the downloadable_packages.

        :param packages_list: List of dictionaries containing package information
        """
        serialized_packages = [
            json.dumps(package, default=json_serial) for package in packages_list
        ]
        self.downloadable_packages = serialized_packages

    def get_downloadable_packages(self):
        """
        Deserialize and return the downloadable_packages as a list of dictionaries.
        """
        return [json.loads(package_str) for package_str in self.downloadable_packages]


class PatchManagementPost(AsyncStructuredNode):
    severity_type_choices = {
        "critical": "Critical",
        "important": "Important",
        "moderate": "Moderate",
        "low": "Low",
        "NST": "No Severity Type",
    }
    node_id = StringProperty(required=True, unique_index=True)
    patch_id = AliasProperty(to="node_id")
    receivedDateTime = StringProperty()
    published = DateTimeProperty()
    thread_id = StringProperty()
    subject = StringProperty()
    post_type = StringProperty()
    conversation_link = StringProperty(default="")
    cve_ids = ArrayProperty(StringProperty(), default=[])
    kb_ids = ArrayProperty(StringProperty(), default=[])
    metadata = JSONProperty()
    keywords = ArrayProperty(StringProperty(), default=[])
    product_mentions = ArrayProperty(StringProperty(), default=[])
    build_numbers = ArrayProperty(StringProperty(), default=[])
    noun_chunks = ArrayProperty(StringProperty(), default=[])
    reliability = StringProperty(default="")
    readability = FloatProperty(default=0.0)
    severity_type = StringProperty(choices=severity_type_choices)
    embedding = ArrayProperty(FloatProperty(), default=[])
    text = StringProperty()
    summary = StringProperty()
    node_label = StringProperty()
    previous_id = StringProperty(default="")
    next_id = StringProperty(default="")
    tags = ArrayProperty(StringProperty(), default=[])
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    # Relationships
    has_symptoms = AsyncRelationshipTo(
        "Symptom", "HAS_SYMPTOM", model=AsyncSymptomCauseFixRel
    )
    has_causes = AsyncRelationshipTo(
        "Cause", "HAS_CAUSE", model=AsyncSymptomCauseFixRel
    )
    has_fixes = AsyncRelationshipTo("Fix", "HAS_FIX", model=AsyncSymptomCauseFixRel)
    references_kbs = AsyncRelationshipTo(
        "KBArticle", "REFERENCES", model=AsyncReferencesRel
    )
    has_tools = AsyncRelationshipTo("Tool", "HAS_TOOL", model=AsyncZeroToManyRel)
    previous_message_has = AsyncRelationshipTo(
        "PatchManagementPost", "PREVIOUS_MESSAGE", model=AsyncPreviousVersionRel
    )
    references_msrcs = AsyncRelationshipTo(
        "MSRCPost", "REFERENCES", model=AsyncReferencesRel
    )
    affects_products = AsyncRelationshipTo(
        "Product", "AFFECTS_PRODUCT", model=AsyncAffectsProductRel
    )

    def set_build_numbers(self, build_numbers_list):
        """Serialize and set the build numbers as a JSON string."""
        if not build_numbers_list:
            self.build_numbers = []
        else:
            self.build_numbers = [json.dumps(sublist) for sublist in build_numbers_list]

    def get_build_numbers(self):
        """Deserialize and return the build numbers as a list of lists."""
        return (
            [json.loads(sublist) for sublist in self.build_numbers]
            if self.build_numbers
            else []
        )


class Technology(AsyncStructuredNode):
    node_id = StringProperty(required=True, unique_index=True)
    name = StringProperty(required=True)
    version = StringProperty()
    architecture = StringProperty()
    build_number = StringProperty()
    description = StringProperty()
    original_id = StringProperty(required=True, unique_index=True)
    source_id = StringProperty()
    tags = ArrayProperty(StringProperty(), default=[])
    created_on = DateTimeFormatProperty(
        default=lambda: datetime.now(),
        format="%Y-%m-%d %H:%M:%S",
    )
    related_to = AsyncRelationshipTo("Product", "RELATED_TO")


# Mapping of labels to their corresponding classes
LABEL_TO_CLASS_MAPPING = {
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
    "Technology": Technology,
}

RELATIONSHIP_MAPPING = {
    "MSRCPost": {
        "has_symptoms": ("Symptom", "HAS_SYMPTOM", AsyncSymptomCauseFixRel),
        "has_causes": ("Cause", "HAS_CAUSE", AsyncSymptomCauseFixRel),
        "has_fixes": ("Fix", "HAS_FIX", AsyncSymptomCauseFixRel),
        "has_faqs": ("FAQ", "HAS_FAQ", AsyncZeroToManyRel),
        "has_tools": ("Tool", "HAS_TOOL", AsyncZeroToManyRel),
        "has_kb_articles": ("KBArticle", "HAS_KB", AsyncZeroToManyRel),
        "has_update_packages": (
            "UpdatePackage",
            "HAS_UPDATE_PACKAGE",
            AsyncHasUpdatePackageRel,
        ),
        "affects_products": ("Product", "AFFECTS_PRODUCT", AsyncAffectsProductRel),
        "previous_version_has": (
            "MSRCPost",
            "PREVIOUS_VERSION",
            AsyncPreviousVersionRel,
        ),
    },
    "Symptom": {
        "affects_products": ("Product", "AFFECTS_PRODUCT", AsyncAffectsProductRel),
    },
    "Cause": {},
    "Fix": {
        "references_kb_articles": ("KBArticle", "REFERENCES", AsyncReferencesRel),
    },
    "FAQ": {},
    "Tool": {},
    "KBArticle": {
        "affects_product": ("Product", "AFFECTS_PRODUCT", AsyncAffectsProductRel),
        "has_symptoms": ("Symptom", "HAS_SYMPTOM", AsyncSymptomCauseFixRel),
        "has_causes": ("Cause", "HAS_CAUSE", AsyncSymptomCauseFixRel),
        "has_fixes": ("Fix", "HAS_FIX", AsyncSymptomCauseFixRel),
        "has_tools": ("Tool", "HAS_TOOL", AsyncZeroToManyRel),
        "has_update_packages": (
            "UpdatePackage",
            "HAS_UPDATE_PACKAGE",
            AsyncHasUpdatePackageRel,
        ),
    },
    "UpdatePackage": {},
    "PatchManagementPost": {
        "previous_message_has": (
            "PatchManagementPost",
            "PREVIOUS_MESSAGE",
            AsyncPreviousMessageRel,
        ),
        "has_symptoms": ("Symptom", "HAS_SYMPTOM", AsyncSymptomCauseFixRel),
        "has_causes": ("Cause", "HAS_CAUSE", AsyncSymptomCauseFixRel),
        "has_fixes": ("Fix", "HAS_FIX", AsyncSymptomCauseFixRel),
        "references_kbs": ("KBArticle", "REFERENCES", AsyncReferencesRel),
        "has_tools": ("Tool", "HAS_TOOL", AsyncZeroToManyRel),
        "references_msrcs": ("MSRCPost", "REFERENCES", AsyncReferencesRel),
        "affects_products": ("Product", "AFFECTS_PRODUCT", AsyncAffectsProductRel),
    },
    "Product": {
        "has_builds": ("ProductBuild", "HAS_BUILD", AsyncZeroToManyRel),
    },
    "ProductBuild": {
        "references_msrcs": ("MSRCPost", "REFERENCES", AsyncReferencesRel),
        "references_kbs": ("KBArticle", "REFERENCES", AsyncReferencesRel),
        "has_update_packages": (
            "UpdatePackage",
            "HAS_UPDATE_PACKAGE",
            AsyncHasUpdatePackageRel,
        ),
    },
    "Technology": {"related_to_product": ("Product", "RELATED_TO")},
}
