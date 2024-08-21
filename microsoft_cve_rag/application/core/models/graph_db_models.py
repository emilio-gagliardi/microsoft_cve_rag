from neomodel import (
    AsyncStructuredNode,
    StringProperty,
    ArrayProperty,
    UniqueIdProperty,
    DateTimeProperty,
    JSONProperty,
)
from neomodel.async_.relationship_manager import (
    AsyncRelationshipFrom,
    AsyncRelationshipTo,
)
import json


class Product(AsyncStructuredNode):
    product_name = StringProperty(required=True)
    product_architecture = StringProperty(required=True)
    product_version = StringProperty(required=True)
    id = UniqueIdProperty()
    description = StringProperty()
    build_numbers = StringProperty()
    product_build_ids = ArrayProperty(StringProperty())

    # Relationships
    builds = AsyncRelationshipTo("ProductBuild", "HAS_BUILD")

    def set_build_numbers(self, build_numbers_list):
        """Serialize and set the build numbers as a JSON string."""
        self.build_numbers = json.dumps(build_numbers_list)

    def get_build_numbers(self):
        """Deserialize and return the build numbers as a list of lists."""
        return json.loads(self.build_numbers) if self.build_numbers else []


class ProductBuild(AsyncStructuredNode):
    product_name = StringProperty(required=True)
    product_architecture = StringProperty(required=True)
    product_version = StringProperty(required=True)
    build_number = StringProperty()
    cve_id = StringProperty()
    kb_id = StringProperty()
    product_build_id = StringProperty()
    published = DateTimeProperty()
    summary = StringProperty()
    article_url = StringProperty()
    cve_url = StringProperty()

    # Relationships
    product = AsyncRelationshipFrom("Product", "HAS_BUILD")
    references_msrc = AsyncRelationshipTo("MSRCPost", "REFERENCES")
    references_kb = AsyncRelationshipTo("KBArticle", "REFERENCES")
    update_packages = AsyncRelationshipTo("UpdatePackage", "HAS_UPDATE_PACKAGE")

    def set_build_number(self, build_number_list):
        """Serialize and set the build number as a JSON string."""
        self.build_number = json.dumps(build_number_list)

    def get_build_number(self):
        """Deserialize and return the build number as a list."""
        return json.loads(self.build_number) if self.build_number else []


class MSRCPost(AsyncStructuredNode):
    id = UniqueIdProperty()
    text = StringProperty()
    embedding = ArrayProperty(float)
    metadata = JSONProperty()
    revision = StringProperty(required=True)
    title = StringProperty(required=True)
    description = StringProperty()
    source = StringProperty()
    impact_type = StringProperty()
    severity_type = StringProperty()
    post_type = StringProperty()
    attack_complexity = StringProperty()
    attack_vector = StringProperty()
    exploit_code_maturity = StringProperty()
    exploitability = StringProperty()
    post_id = StringProperty()
    max_severity = StringProperty()
    remediation_level = StringProperty()
    report_confidence = StringProperty()
    summary = StringProperty()

    # Relationships
    symptoms = AsyncRelationshipTo("Symptom", "HAS_SYMPTOM")
    causes = AsyncRelationshipTo("Cause", "HAS_CAUSE")
    fixes = AsyncRelationshipTo("Fix", "HAS_FIX")
    faqs = AsyncRelationshipTo("FAQ", "HAS_FAQ")
    kb_articles = AsyncRelationshipTo("KBArticle", "HAS_KBARTICLE")
    update_packages = AsyncRelationshipTo("UpdatePackage", "HAS_UPDATE_PACKAGE")
    previous_version = AsyncRelationshipTo("MSRCPost", "PREVIOUS_VERSION")
    next_version = AsyncRelationshipFrom("MSRCPost", "NEXT_VERSION")


class Symptom(AsyncStructuredNode):
    id = UniqueIdProperty()
    description = StringProperty(required=True)
    source_id = StringProperty()
    source_type = StringProperty()

    # Relationships
    msrc_posts = AsyncRelationshipFrom("MSRCPost", "HAS_SYMPTOM")
    emails = AsyncRelationshipFrom("PatchManagementEmail", "HAS_SYMPTOM")


class Cause(AsyncStructuredNode):
    id = UniqueIdProperty()
    description = StringProperty(required=True)
    source_id = StringProperty()
    source_type = StringProperty()

    # Relationships
    msrc_posts = AsyncRelationshipFrom("MSRCPost", "HAS_CAUSE")
    emails = AsyncRelationshipFrom("PatchManagementEmail", "HAS_CAUSE")


class Fix(AsyncStructuredNode):
    id = UniqueIdProperty()
    description = StringProperty(required=True)
    source_id = StringProperty()
    source_type = StringProperty()

    # Relationships
    msrc_posts = AsyncRelationshipFrom("MSRCPost", "HAS_FIX")
    emails = AsyncRelationshipFrom("PatchManagementEmail", "HAS_FIX")


class FAQ(AsyncStructuredNode):
    id = UniqueIdProperty()
    question = StringProperty(required=True)
    answer = StringProperty(required=True)
    source_id = StringProperty()
    source_type = StringProperty()

    # Relationships
    msrc_posts = AsyncRelationshipFrom("MSRCPost", "HAS_FAQ")
    emails = AsyncRelationshipFrom("PatchManagementEmail", "HAS_FAQ")


class Tool(AsyncStructuredNode):
    id = UniqueIdProperty()
    name = StringProperty(required=True)
    description = StringProperty(required=True)
    download_url = StringProperty()
    source_id = StringProperty()
    source_type = StringProperty()

    # Relationships
    msrc_posts = AsyncRelationshipFrom("MSRCPost", "HAS_TOOL")
    emails = AsyncRelationshipFrom("PatchManagementEmail", "HAS_TOOL")


class KBArticle(AsyncStructuredNode):
    id = UniqueIdProperty()
    kb_id = StringProperty()
    build_number = ArrayProperty()
    published = DateTimeProperty()
    product_build_id = StringProperty()
    article_url = StringProperty()

    # Relationships
    msrc_posts = AsyncRelationshipFrom("MSRCPost", "HAS_KBARTICLE")
    emails = AsyncRelationshipFrom("PatchManagementEmail", "HAS_KBARTICLE")
    product_builds = AsyncRelationshipFrom("ProductBuild", "REFERENCES")


class UpdatePackage(AsyncStructuredNode):
    id = UniqueIdProperty()
    package_type = StringProperty()
    package_url = StringProperty()
    build_number = ArrayProperty()
    product_build_id = StringProperty()
    downloadable_packages = ArrayProperty(JSONProperty())

    # Relationships
    msrc_posts = AsyncRelationshipFrom("MSRCPost", "HAS_UPDATE_PACKAGE")
    product_builds = AsyncRelationshipFrom("ProductBuild", "HAS_UPDATE_PACKAGE")


class PatchManagementEmail(AsyncStructuredNode):
    id = UniqueIdProperty()
    receivedDateTime = DateTimeProperty()
    topic = StringProperty()
    subject = StringProperty()
    email_text_original = StringProperty()
    evaluated_keywords = ArrayProperty(StringProperty())
    evaluated_noun_chunks = ArrayProperty(StringProperty())
    post_type = StringProperty()
    conversation_link = StringProperty()
    cve_mentions = StringProperty()
    metadata = JSONProperty()

    # Relationships
    symptoms = AsyncRelationshipTo("Symptom", "HAS_SYMPTOM")
    causes = AsyncRelationshipTo("Cause", "HAS_CAUSE")
    fixes = AsyncRelationshipTo("Fix", "HAS_FIX")
    faqs = AsyncRelationshipTo("FAQ", "HAS_FAQ")
    kb_articles = AsyncRelationshipTo("KBArticle", "HAS_KBARTICLE")
    previous_message = AsyncRelationshipTo("PatchManagementEmail", "PREVIOUS_MESSAGE")
    next_message = AsyncRelationshipFrom("PatchManagementEmail", "NEXT_MESSAGE")
    msrc_posts = AsyncRelationshipTo("MSRCPost", "REFERENCES")
