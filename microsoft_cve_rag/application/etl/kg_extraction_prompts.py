from typing import Dict


def get_prompt(
    entity_type: str,
    document_type: str,
    context_str: str,
    source_id: str,
    source_type: str,
    prompt_type: str = "user",
    **kwargs,
) -> str:

    if (
        document_type in PROMPT_TEMPLATES
        and entity_type in PROMPT_TEMPLATES[document_type]
    ):
        prompt_template = PROMPT_TEMPLATES[document_type][entity_type][prompt_type]
        prompt = prompt_template.format(
            context_str=context_str,
            source_id=source_id,
            source_type=source_type,
            **kwargs,
        )
        return prompt
    else:
        raise ValueError(
            f"No prompt template found for entity '{entity_type}', document type '{document_type}', "
            f"or prompt type '{prompt_type}'."
        )


def get_prompt_template(
    entity_type: str,
    document_type: str,
    prompt_type: str = "user",
) -> str:

    if (
        document_type in PROMPT_TEMPLATES
        and entity_type in PROMPT_TEMPLATES[document_type]
    ):
        prompt_template = PROMPT_TEMPLATES[document_type][entity_type][prompt_type]

        return prompt_template
    else:
        raise ValueError(
            f"No prompt template found for entity '{entity_type}', document type '{document_type}', "
            f"or prompt type '{prompt_type}'."
        )


PROMPT_TEMPLATES: Dict[str, Dict[str, Dict[str, str]]] = {}

# Symptom extraction from MSRCPost
PROMPT_TEMPLATES = {
    "MSRCPost": {
        "Symptom": {
            "user": (
                """Given the following Microsoft Security Response Center (MSRC) post and its metadata, extract the core \
Symptom that is described or can be inferred. A Symptom is an observable behavior, an error message, or any \
indication that something is going wrong in the system, as experienced by end users or system administrators. \
**It is not a vulnerability or technical exploit, but rather what the user notices as a result of the underlying \
issue.** For example, a Symptom could be system crashes, poor performance, unexpected reboots, failed updates, \
or error messages seen by the user. Symptoms help people identify issues that are affecting their environments. \n
Do not describe vulnerabilities, exploits, or technical flaws directly. Instead, describe the **impact or \
observable behavior** that a system administrator or end user would see as a consequence of the security issue. \
Focus on the **user's perspective**, not the attacker's. \n
Be thorough and consider subtle aspects that may not be explicitly stated. Describe how the security update \
affects a particular system or software product from the perspective of the end user. For instance:\n
- 'The computer fails to boot after installing a security patch.'\n
'Network communication is intermittently lost on systems using the affected driver.'\n
- 'The system experiences slow performance and occasional reboots.'\n
Do not restate or reference the original post directly; the Symptom should stand alone and specify the \
**observable behavior** or **impact** rather than describing the vulnerability itself. \n
The `post_type` key in the metadata determines whether a Symptom should be extracted:\n
- If `post_type` is 'Information only', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Problem statement', extract the Symptom.\n
- If `post_type` is 'Solution provided', extract the Symptom.\n
- If `post_type` is 'Helpful tool', do not extract anything and return a null JSON object.\n\n
For each Symptom, generate a six-word CamelCase label, e.g., 'WindowsFailsToBootWithSignedWdacPolicy'. \
**Avoid including words such as 'vulnerability', 'exploit', or 'security'** in the label. Generate a complete and \
thorough description of at least 3 sentences. The description should stand alone as a precise explanation of the \
Symptom, and must include the products, build numbers, and KB Article IDs (eg., KB5040442) provided in the context. \n
---------------------
<start context>
{context_str}

Build Numbers:
{build_numbers}
KB Articles:
{kb_ids}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "symptom_label": "6 word camel case label for the symptom",
        "description": "A concise and precise technical description of the symptom. Be as specific as possible. Include build numbers, kb ids, and update package links.",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "tags": ["search", "prioritized", "terms"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Symptom",
        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the symptom describes the CVE.",
        "severity_type": "one of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
        "Cause": {
            "user": (
                """Given the following Microsoft Security Response Center (MSRC) post and its metadata, extract the core Cause \
that is described or can be inferred. A Cause is any technical or situational condition responsible for the \
Symptom or issue described in the CVE. This could be a flaw in software, a misconfiguration, or any contributing \
factor that underpins the identified Symptom. Focus on the technical reasons driving the issue, and avoid \
restating the full text of the post. Provide a complete description of the Cause that would allow Microsoft \
experts to trace back the root of the issue. Ensure to describe the what, where, and how. \n
The `post_type` key in the metadata determines whether a Cause should be extracted:\n
- If `post_type` is 'Information only', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Problem statement', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Solution provided', extract the Cause.\n
- If `post_type` is 'Helpful tool', do not extract anything and return a null JSON object.\n\n
A Cause explains the underlying technical condition or flaw causing the Symptom. Include the products, build \
numbers, and KB Article IDs provided in the context. For each Cause, generate a six-word CamelCase label, e.g., \
'IncorrectRegistryValueCausesRebootError'. **Avoid including words such as 'Cause', 'vulnerability', 'exploit', or \
'security'** in the label. Generate a complete and thorough description of at least 5 sentences. \n
---------------------
<start context>
{context_str}

Build Numbers:
{build_numbers}
KB Articles:
{kb_ids}
<end context>
---------------------
Provide your answer in the following JSON format:
    {{
        "description": "A concise, thorough and precise technical description of the cause. Include the build numbers and kb ids contained in the context.",
        "cause_label": "The 6 word CamelCase label for the cause",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "tags": ["search", "prioritized", "terms"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Cause",
        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the cause is the true and canonical source of the CVE.",
        "severity_type": "one of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
        "Fix": {
            "user": (
                """Given the following Microsoft Security Response Center (MSRC) post, extract the core Fix that is described \
or can be inferred. \n
The `post_type` key in the metadata determines whether a Fix should be extracted:\n
- If `post_type` is 'Information only', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Problem statement', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Solution provided', extract the Fix.\n
- If `post_type` is 'Helpful tool', do not extract anything and return a null JSON object.\n\n
A Fix is a technical action, configuration change, or patch available to address the Cause or mitigate the \
Symptom. Focus on the specific technical response to the issue affecting the CVE, referencing affected \
systems or products without repeating the post's full text. Generate a complete and thorough description of \
at least 5 sentences. The description should stand alone as a precise explanation of the steps necessary to \
implement the fix, it must include the products, build numbers, KB Article IDs (eg., KB5040442), and update \
package links provided in the context. \n
For each Fix, generate a six-word CamelCase label. Do not include build numbers or verbalizations in the \
label, e.g., 'ApplyKbPatchToResolveRebootIssue. Also provide a concise description'. **Avoid including words \
such as 'Fix', 'vulnerability', 'exploit', or 'security'** in the label. Generate a complete and thorough description of at least 5 sentences. \
Finally, generate up to 3 tags or keywords that describe the Fix and add them as a list to the output. \n
Ensure the update package URLs are explicitly mentioned in the `description` field. This extraction is \
optional, do not fabricate or hallucinate a fix if it is not stated. Return an empty JSON dictionary if no \
fix is found.\n
---------------------
<start context>
{context_str}

Build Numbers:
{build_numbers}
KB Articles:
{kb_ids}
Update Packages:
{update_package_urls}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "description": "A concise and precise technical description of the fix, including the build numbers and update package links.",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "fix_label": "The 6 word CamelCase label for the fix",
        "tags": ["search", "prioritized", "terms"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Fix",
        "reliability": "A floating score between 0.0 and 1.0 that indicates how reliably the fix mitigates or solves the CVE.",
        "severity_type": "One of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
        "Tool": {
            "user": (
                """Given the following Microsoft Security Response Center (MSRC) post that has been filtered to reference \
only Windows 10, Windows 11, and Microsoft Edge products. Proceed in two steps. First, evaluate the context \
and determine if a tool is mentioned within the text. A Tool is any software, command-line utility, or \
similar that Microsoft or end users may use to diagnose, mitigate, or resolve the issue mentioned in the CVE. \
Focus on names, configurations, or commands necessary for addressing the symptoms or causes of the CVE. \n
The `post_type` key in the metadata determines whether a Tool should be attempted to be extracted:\n
- If `post_type` is 'Information only', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Problem statement', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Solution provided', attempt to extract a Tool. If there is no tool mentioned, return a null JSON object.\n
- If `post_type` is 'Helpful tool', extract the Tool.\n\n
If you find a tool, generate a six-word CamelCase label, e.g., 'EnableFirewallUsingPowerShellCmd'. **Avoid \
including words such as 'Tool', 'vulnerability', 'exploit', or 'security'** in the label. Generate a complete \
description of the tool of at least 3 sentences including how an end-user may find or use the tool. Evaluate \
the text, look for an external URL for the tool. The tool URL is NOT the `source` url from the document \
metadata. Do not insert the document's source url into the `tool_url` field. \n
The `source_id` and `source_type` fields have been filled in for you. This extraction is optional, do not \
fabricate or hallucinate a tool if it is not stated, return an empty json dictionary if no tool is found. \n
---------------------
<start context>
{context_str}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "tool_label": "The 6 word CamelCase label for the tool.",
        "description": "A concise description of the tool and its purpose.",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "source_url": "If a URL for a tool is mentioned, include it here.",
        "tags": ["search", "prioritized", "terms"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Tool",
        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the tool contributes to mitigating or supporting the work needed to resolve the core issue in the Patch Management post.",
        "severity_type": "one of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
        "Technology": {
            "user": (
                """Given the following Microsoft Security Response Center (MSRC) post, evaluate it to determine if a tertiary technology is referenced in relation to the CVE. If there is, extract it as a technology entity. A Technology refers to any Microsoft product, service, or platform that is relevant to the context. Technologies are separate from the core Microsoft product families. Focus on identifying technologies referenced or implied that could relate to the post. Attempt to extract separate pieces for the `name`, `version`, `architecture`, and `build_number` fields. Generate a complete and thorough description of at least 3 sentences.  This entity extraction is optional therefore do not fabricate or hallucinate a technology, if there is no technology mentioned return an empty json dictionary. The `source_id` and `source_type` fields have been filled in for you. Finally, generate up to 3 tags or keywords that describe the Technology and add them as a list to the output.\n
<start context>
{context_str}

Build Numbers:
{build_numbers}
KB Articles:
{kb_ids}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "name": "Name of the technology.",
        "description": "A concise description of the technology and its relevance to the issue.",
        "version": "A product version number if available.",
        "architecture": "A product architecture if available. E.g., x86, x64, ARM, etc.",
        "build_number": "A list of integers representing a product build numbers if available. E.g., [10, 0, 19041, 450]",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "tags": ["search", "prioritized", "terms"],
        "node_label": "Technology",
        "node_id": "placeholder_node_id_leave_as_is",
        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the Technology is described in the Patch Management post.",
        "severity_type": "one of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
    },
    "PatchManagementPost": {
        "Symptom": {
            "user": (
                """Given the following Patch Management post and its metadata, extract the core Symptom that is described or \
can be inferred. \n
Patch Management posts are often very limited in their text and contain irrelevant text in the form of email \
signatures. When the text of the post is not sufficient to extract a Symptom, you may infer and combine your \
own knowledge of the topic to fill in gaps, but do not fabricate entities for completeness. All your answers \
must be technically correct in the domain of Microsoft patches and systems.\n
The `post_type` key in the metadata determines whether a Symptom should be extracted:\n
- If `post_type` is 'Information only', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Problem statement', extract the Symptom.\n
- If `post_type` is 'Solution provided', extract the Symptom.\n
- If `post_type` is 'Helpful tool', do not extract anything and return a null JSON object.\n\n
A Symptom is an observable behavior, error message, or any indication that something is going wrong in the \
system, especially as experienced by end users or system administrators. Symptoms help people identify \
issues affecting their environments. Be thorough and consider subtle aspects that may not be explicitly \
stated. Describe how the post affects a particular system or software product (e.g., How an attacker gains \
access to a firewall). Do not restate or reference the original post; the Symptom should stand alone and \
specify the product or technology affected by the post. \n
For each Symptom, generate a six-word CamelCase label, e.g., 'WindowsFailsToBootWithSignedWdacPolicy'. \
**Avoid including words such as 'Symptom', 'vulnerability', 'exploit', or 'security'** in the label. Generate \
a complete and thorough description of at least 3 sentences. The description should stand alone as a precise \
explanation of the Symptom, and must include the products, build numbers, and KB Article IDs (eg., KB5040442) \
provided in the context. \n
The `source_id` and `source_type` fields have been filled in for you. Note. When estimating the reliability \
of the post, factor in the rating: language usage, the mastery of the topic, the completeness of the answer, \
and supporting links. Finally, generate up to 3 tags or keywords that describe the Symptom and add them as a \
list to the output.\n
---------------------
<start context>
{context_str}

Build Numbers:
{build_numbers}
KB Articles:
{kb_ids}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "symptom_label": "6 word camel case label for the symptom",
        "description": "A concise and precise technical description of the symptom. Be as specific as possible.",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "tags": ["search", "prioritized", "terms"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Symptom",
        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the symptom describes the core issue in the Patch Management post.",
        "severity_type": "one of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
        "Cause": {
            "user": (
                """Given the following Patch Management post and its metadata, extract the core Cause that is described or \
inferred. \n
The `post_type` key in the metadata determines whether a Cause should be extracted:\n
- If `post_type` is 'Information only', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Problem statement', do not extract a Cause.\n
- If `post_type` is 'Solution provided', extract the Cause.\n
- If `post_type` is 'Helpful tool', do not extract anything and return a null JSON object.\n\n
A Cause is the underlying reason or root of the issue, which could be a bug, vulnerability, misconfiguration, \
or any other factor contributing to the problem. Be thorough and consider subtle aspects that may not be \
explicitly stated. Provide a concise description of the Cause that would allow Microsoft experts to trace \
back the root of the issue. \n
For each Cause, generate a six-word CamelCase label, e.g., 'IncorrectRegistryValueYieldsRebootError'. \
**Avoid including words such as 'Cause', 'vulnerability', 'exploit', or 'security'** in the label. Generate a \
complete and thorough description of at least 5 sentences. The description should stand alone as a precise \
explanation of the Cause, and must include the products, build numbers, and KB Article IDs (eg., KB5040442) \
provided in the context. \n
Patch Management posts are often very limited in their text and contain irrelevant text in the form of email \
signatures. All your answers must be technically correct in the domain of Microsoft patches and systems. \n
Note. When estimating the reliability of the post, factor in the rating: language usage, the mastery of the \
topic, the completeness of the answer, and supporting links.\n
The `source_id` and `source_type` fields have been filled in for you.\n
---------------------
<start context>
{context_str}

Build Numbers:
{build_numbers}
KB Articles:
{kb_ids}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "description": "A concise and precise technical description of the cause.",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "cause_label": "The 6 word CamelCase label for the cause",
        "tags": ["search", "prioritized", "terms"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Cause",
        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the cause explains and solves the core issue in the Patch Management post.",
        "severity_type": "one of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
        "Fix": {
            "user": (
                """Given the following Patch Management post and its metadata, extract the core Fix described or inferred. \
Patch Management posts are often very limited in their text and contain irrelevant text in the form of email \
signatures. The `post_type` key in the metadata determines whether a Fix should be extracted:\n
- If `post_type` is 'Information only', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Problem statement', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Solution provided', extract the Fix.\n
- If `post_type` is 'Helpful tool', do not extract anything and return a null JSON object.\n\n
A Fix is a technical action, configuration change, or patch available to address the Cause or mitigate the Symptom. \
Focus on the specific technical response to the Cause, referencing affected systems or products without repeating \
the post's full text. The description should stand alone as a precise explanation of the Fix. \
Include the products, build numbers, KB Article IDs, and update package links provided in the context. \
For each Fix, generate a six-word CamelCase label and provide a concise description, e.g., 'ApplyKbPatchToResolveRebootIssue'. \
**Avoid including words such as 'Fix', 'vulnerability', 'exploit', or 'security'** in the label. \
Generate a complete and thorough description of at least 5 sentences. \
Finally, generate up to 3 tags or keywords that describe the Fix and add that as a list to the output.\n\n
---------------------
<start context>
{context_str}

Build Numbers:
{build_numbers}
KB Articles:
{kb_ids}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "description": "A concise and precise technical description of the fix.",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "fix_label": "The 6 word CamelCase label for the fix",
        "tags": ["search", "prioritized", "terms"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Fix",
        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the fix explains and solves the core issue in the Patch Management post.",
        "severity_type": "one of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
        "Tool": {
            "user": (
                """Given the following Patch Management post and its metadata, attempt to extract any Tool described or inferred. \
Patch Management posts are often very limited in their text and contain irrelevant text in the form of email \
signatures. Tools are not required to extract; it is important you do not fabricate or hallucinate a tool if it \
doesn't exist. The `post_type` key in the metadata determines whether a Tool should be attempted to be extracted:\n
- If `post_type` is 'Information only', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Problem statement', do not extract anything and return a null JSON object.\n
- If `post_type` is 'Solution provided', attempt to extract a Tool. If there is no tool mentioned, return a null JSON object.\n
- If `post_type` is 'Helpful tool', extract the Tool.\n\n
A Tool is any software, command-line utility, or similar that Microsoft or end users may use to diagnose, mitigate, \
or resolve the issue. Focus on names, configurations, or commands necessary for addressing the symptoms or causes \
of the issue. If you find a tool, generate a six-word CamelCase label, e.g., 'EnableFirewallUsingPowerShellCmd'. \
**Avoid including words such as 'Tool', 'vulnerability', 'exploit', or 'security'** in the label. \
Generate a complete description of the tool and how an end-user may find or use the tool. \
Include its name, description, and any relevant external URLs. The `tool_url` is NOT the same as the `source_url`. If a tool is referenced in a source document, the url of the source document should be assigned to the `source_url` field. Whereas the url of the tool itself should be assigned to the `tool_url` field. The priority is the `tool_url` field.\n\n
---------------------
<start context>
{context_str}

Build Numbers:
{build_numbers}
KB Articles:
{kb_ids}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "tool_label": "The 6 word CamelCase label for the tool.",
        "description": "A concise description of the tool and its purpose.",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "source_url": "If a URL is included assign it here.",
        "tags": ["search", "prioritized", "terms"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Tool",
        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the tool contributes to mitigating or supporting the work needed to resolve the core issue in the Patch Management post.",
        "severity_type": "one of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
        "Technology": {
            "user": (
                """Given the following Patch Management post, evaluate it and determine if a tertiary technology is mentioned in relation to the central topic of the post. Note. the text is messy email text with poor grammar and in many cases irrelevant text like email signatures. Ignore irrelevant text. Additionally, the only official 'Products' we track in this system filter for Windows 10, Windows 11, and Microsoft Edge, as such technologies tend to be outside of the core Microsoft product families. If you identify a technology, extract it as a technology entity. A Technology refers to any product, service, or platform that is relevant to the context but is not the locus or core issue. Focus on identifying technologies referenced or implied that could relate to the post. Attempt to extract separate variables for the `name`, `version`, `architecture`, and `build_number` fields. The `source_id`,  `source_type`, and `node_label` fields have been filled in for you. Do not modify the node_label value.\n
Note. When estimating the reliability of the post, factor in the rating: language usage, the mastery of the topic, the completeness of the answer and supporting links.\n
This extraction is optional, do not fabricate or hallucinate a technology if it is not stated, return an empty json dictionary if no technology is found. Finally, generate up to 3 tags or keywords that will help end-users search for the Technology and add them as a list to the output.\n
---------------------
<start context>
{context_str}

Build Numbers:
{build_numbers}
KB Articles:
{kb_ids}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "name": "Name of the technology.",
        "description": "A concise description of the technology and its relevance to the issue.",
        "version": "A product version number if available.",
        "architecture": "A product architecture if available. E.g., x86, x64, ARM, etc.",
        "build_number": "A list of integers representing a product build numbers if available. E.g., [10, 0, 19041, 450]",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "tags": ["search", "prioritized", "terms"],
        "node_label": "Technology",
        "node_id": "placeholder_node_id_leave_as_is",
        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the Technology is described in the Patch Management post.",
        "severity_type": "one of [critical, important, moderate, low]"
    }},

Do not output any other dialog or text outside of the json object. The JSON object must be valid."""
            ),
            "system": """You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post."""
        },
    },
}


PROMPT_TEMPLATES_v2 = {
    "manual": {
        "MSRCPost": {
            "Symptom": {
                "user": (
                    "Given the following Microsoft Security Response Center (MSRC) post, extract the core Symptom that is described or can be inferred. A Symptom is an observable behavior, an error message, or any indication that something is going wrong in the system, as experienced by end users or system administrators. **It is not a vulnerability or technical exploit, but rather what the user notices as a result of the underlying issue.** For example, a Symptom could be system crashes, poor performance, unexpected reboots, failed updates, or error messages seen by the user. Symptoms help people identify issues that are affecting their environments. \nDo not describe vulnerabilities, exploits, or technical flaws directly. Instead, describe the **impact or observable behavior** that a system administrator or end user would see as a consequence of the security issue. Focus on the **user's perspective**, not the attacker's. \nBe thorough and consider subtle aspects that may not be explicitly stated. Describe how the security update affects a particular system or software product from the perspective of the end user. For instance:\n- 'The computer fails to boot after installing a security patch.'\n'Network communication is intermittently lost on systems using the affected driver.'\n- 'The system experiences slow performance and occasional reboots.'\nDo not restate or reference the original post directly; the Symptom should stand alone and specify the **observable behavior** or **impact** rather than describing the vulnerability itself. For each Symptom, generate a six-word CamelCase label and provide a concise description, e.g., 'WindowsFailsToBootWithSignedWdacPolicy'. **Avoid including words such as 'vulnerability', 'exploit', or 'security'** in the label. \nThe `source_id` and `source_type` fields have been filled in already. Finally, generate up to 3 tags or keywords that describe the Symptom and add them as a list to the output.\n"
                    "---------------------\n"
                    "Post metadata and text:\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Provide your answer in the following JSON format:\n"
                    "\n"
                    "    {{\n"
                    '        "symptom_label": "6 word camel case label for the symptom",\n'
                    '        "description": "A concise and precise technical description of the symptom. Be as specific as possible.",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Symptom"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
            "Cause": {
                "user": (
                    "Given the following Microsoft Security Response Center (MSRC) post, extract the core Cause that is described or can be inferred. A Cause is any technical or situational condition responsible for the Symptom or issue described in the CVE. This could be a flaw in software, a misconfiguration, or any contributing factor that underpins the identified Symptom. Focus on the technical reasons driving the issue, and avoid restating the full text of the post. Provide a complete description of the Cause that would allow Microsoft experts to trace back the root of the issue. Ensure to describe the what, where, and how. For each Cause, generate a six-word CamelCase label, e.g., 'IncorrectRegistryValueCausesRebootError'. Do not include the word 'Cause' in the label. The `source_id` and `source_type` fields have been filled in for you. Finally, generate up to 3 tags or keywords that describe the Cause and add that as a list to the output\n"
                    "---------------------\n"
                    "Post metadata and text:\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Provide your answer in the following JSON format:\n"
                    "    {{\n"
                    '        "description": "A concise, thorough and precise technical description of the cause.",\n'
                    '        "cause_label": "The 6 word CamelCase label for the cause",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Cause"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
            "Fix": {
                "user": (
                    "Given the following Microsoft Security Response Center (MSRC) post, extract the core Fix that is described or can be inferred. In the document metadata is the key `post_type`, if it's value is 'Solution provided' the post likely contains an explicit fix, if the value is 'Critical' then there is likely no fix. A Fix is a technical action, configuration change, or patch available to address the Cause or mitigate the Symptom. Focus on the specific technical response to the Cause, referencing affected systems or products without repeating the post's full text. This description should stand alone as a precise explanation of the Fix. For each Fix, generate a six-word CamelCase label and provide a concise description, e.g., 'ApplyKbPatchToResolveRebootIssue'. Do not include the word 'Fix' in the label. The `source_id` and `source_type` fields have been filled in for you. Finally, generate up to 3 tags or keywords that describe the Fix and add them as a list to the output. This extraction is optional, do not fabricate or hallucinate a fix if it is not stated, return an empty json dictionary if no fix is found.\n"
                    "---------------------\n"
                    "Post metadata and text:\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Provide your answer in the following JSON format:\n"
                    "\n"
                    "    {{\n"
                    '        "description": "A concise and precise technical description of the fix.",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "fix_label": "The 6 word CamelCase label for the fix",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Fix"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
            "Tool": {
                "user": (
                    "Given the following Microsoft Security Response Center (MSRC) post that has been filtered to reference only Windows 10, Windows 11, and Microsoft Edge products. Proceed in two steps. First, evaluate the context and determine if a tool is mentioned within the text. A Tool is any software, command-line utility, or similar that Microsoft or end users may use to diagnose, mitigate, or resolve the issue. Focus on names, configurations, or commands necessary for addressing the symptoms or causes of the CVE. If you find a tool, generate a six-word CamelCase label, e.g., 'EnableFirewallUsingPowerShellCmd'. Do not include the word 'Tool' in the label. Generate a complete description of the tool and how an end-user may find or use the tool. The `source_id` and `source_type` fields have been filled in for you. This extraction is optional, do not fabricate or hallucinate a tool if it is not stated, return an empty json dictionary if no tool is found. Finally, generate up to 3 tags or keywords that will help end-users search for the Tool and add them as a list to the output.\n"
                    "---------------------\n"
                    "Post metadata and text:\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Provide your answer in the following JSON format:\n"
                    "\n"
                    "    {{\n"
                    '        "tool_label": "The 6 word CamelCase label for the tool.",\n'
                    '        "description": "A concise description of the tool and its purpose.",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "source_url": "If a URL is included assign it here.",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Tool"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
            "Technology": {
                "user": (
                    "Given the following Microsoft Security Response Center (MSRC) post, evaluate it to determine if a tertiary technology is referenced in relation to the CVE. If there is, extract it as a technology entity. A Technology refers to any Microsoft product, service, or platform that is relevant to the context. Technologies are separate from the core Microsoft product families. Focus on identifying technologies referenced or implied that could relate to the post. Attempt to extract separate pieces for the `name`, `version`, `architecture`, and `build_number` fields. This entity extraction is optional therefore do not fabricate or hallucinate a technology, if there is no technology mentioned return an empty json dictionary. The `source_id` and `source_type` fields have been filled in for you. Finally, generate up to 3 tags or keywords that describe the Technology and add them as a list to the output.\n"
                    "Provide your answer in the following JSON format:\n"
                    "\n"
                    "    {{\n"
                    '        "name": "Name of the technology.",\n'
                    '        "description": "A concise description of the technology and its relevance to the issue.",\n'
                    '        "version": "A product version nyumber if available.",\n'
                    '        "architecture": "A product architecture if available. E.g., x86, x64, ARM, etc.",\n'
                    '        "build_number": "A list of integers representing a product build numbers if available. E.g., [10, 0, 19041, 450]",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Technology"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
        },
        "PatchManagementPost": {
            "Symptom": {
                "user": (
                    "Given the following Patch Management Google Group post, extract the core Symptom that is described or can be inferred.\n"
                    "Patch Management posts are often very limited in their text and contain irrelevant text in the form of email signatures. When the text of the post is not sufficient to extract a Symptom, you may infer and combine your own knowledge of the topic to fill-in gaps, but do not fabricate entities for completeness. All your answers must be technically correct in the domain of Microsoft patches and systems.\n"
                    "A Symptom is an observable behavior, error message, or any indication that something is going wrong in the system, especially as experienced by end users or system administrators. Symptoms help people identify issues affecting their environments. Be thorough and consider subtle aspects that may not be explicitly stated. Describe how the security update affects a particular system or software product (e.g., How an attacker gains access to a firewall). Do not restate or reference the original post; the Symptom should stand alone and specify the product or technology affected by the security update. For each Symptom, generate a six-word CamelCase label and provide a concise description, e.g., 'WindowsFailsToBootWithSignedWdacPolicy'. Do not include the word 'Symptom' in the label. The `source_id` and `source_type` fields have been filled in for you.  Finally, generate up to 3 tags or keywords that describe the Symptom and add that as a list to the output.\n"
                    "---------------------\n"
                    "Post metadata and text:\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Provide your answer in the following JSON format:\n"
                    "\n"
                    "    {{\n"
                    '        "symptom_label": "6 word camel case label for the symptom",\n'
                    '        "description": "A concise and precise technical description of the symptom. Be as specific as possible.",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Symptom"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
            "Cause": {
                "user": (
                    "Given the following Patch Management post, attempt to extract the core **Cause** that is described if it can be inferred. Do not fabricate or hallcuinate a cause if it is not stated, return an empty json dictionary if no cause is found.\n"
                    "A **Cause** is the underlying reason or root of the issue, which could be a bug, vulnerability, misconfiguration, or any other factor contributing to the problem. "
                    "Be thorough and consider subtle aspects that may not be explicitly stated. Provide a concise description of the Cause that would allow Microsoft experts to trace back the root of the issue. For each Cause, generate a six-word CamelCase label and provide a concise description, e.g., 'IncorrectRegistryValueYieldsRebootError'. Do not include the word 'Cause' in the label"
                    "Patch Management posts are often very limited in their text and contain irrelevant text in the form of email signatures. All your answers must be technically correct in the domain of Microsoft patches and systems.\n"
                    "The `source_id` and `source_type` fields have been filled in for you."
                    "Provide your answer in the following JSON format:\n"
                    "\n"
                    "    {{\n"
                    '        "description": "A concise and precise technical description of the cause.",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "cause_label": "The 6 word CamelCase label for the cause",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Cause",\n'
                    '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the cause is the true and canonical source of the issue described in the Patch Management post.",\n'
                    '        "severity_type": "one of [critical, important, moderate, low]"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
            "Fix": {
                "user": (
                    "Given the following Patch Management post, attempt extract the core Fix that is described or can be inferred. In the document metadata is the key `post_type`, if it's value is 'Solution provided' the post likely contains an explicit fix, if the value is 'Problem statement' then there is no fix detected in the text.\n"
                    "Patch Management posts are often very limited in their text and contain irrelevant text in the form of email signatures. All your answers must be technically correct in the domain of Microsoft patches and systems. Do not fabricate or hallucinate a fix if it is not stated, return an empty json dictionary if no fix is found.\n"
                    "The `source_id` and `source_type` fields have been filled in for you."
                    "A Fix is a technical action, configuration change, or patch available to address the Cause or mitigate the Symptom. Focus on the specific technical response to the Cause, referencing affected systems or products without repeating the post's full text. This description should stand alone as a precise explanation of the Fix. For each Fix, generate a six-word CamelCase label and provide a concise description, e.g., 'ApplyKbPatchToResolveRebootIssue'. Do not include the word 'Fix' in the label. Finally, generate up to 3 tags or keywords that describe the Fix and add that as a list to the output.\n"
                    "---------------------\n"
                    "Post metadata and text:\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Provide your answer in the following JSON format:\n"
                    "\n"
                    "    {{\n"
                    '        "description": "A concise and precise technical description of the fix.",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "fix_label": "The 6 word CamelCase label for the fix",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Fix",\n'
                    '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the fix mitigates or solves the core issue in the Patch Management post.",\n'
                    '        "severity_type": "one of [critical, important, moderate, low]"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
            "Tool": {
                "user": (
                    "Given the following Patch Management post, attempt to extract any useful tool. Your goals are to identify and, if possible, extract any tool mentioned in the text. Note. the text is messy email text with poor grammar and in many cases irrelavant text like email signatures. Ignore irrelevant text. Proceed in two steps. First, evaluate the context and determine if a tool is mentioned within the text. Inspect the post metadata for a key `post_type`, if it is 'Helpful tool' then a tool has been detected in the text. A Tool is any software, command-line utility, or similar that Microsoft or end users may use to diagnose, mitigate, or resolve the issue. Focus on names, configurations, or commands necessary for addressing the symptoms or causes of the CVE. If you find a tool, generate a six-word CamelCase label, e.g., 'EnableFirewallUsingPowerShellCmd'. Do not include the word 'Tool' in the label. Generate a complete description of the tool and how an end-user may find or use the tool. The `source_id` and `source_type` fields have been filled in for you. This extraction is optional, do not fabricate or hallucinate a tool if it is not stated, return an empty json dictionary if no tool is found. Finally, generate up to 3 tags or keywords that will help end-users search for the Tool and add them as a list to the output.\n"
                    "---------------------\n"
                    "Post metadata and text:\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Provide your answer in the following JSON format:\n"
                    "\n"
                    "    {{\n"
                    '        "tool_label": "The 6 word CamelCase label for the tool.",\n'
                    '        "description": "A concise description of the tool and its purpose.",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "source_url": "If a URL is included assign it here.",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Tool"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
            "Technology": {
                "user": (
                    "Given the following Patch Management post, evaluate it and determine if a tertiary technology is mentioned in relation to the central topic of the post. Note. the text is messy email text with poor grammar and in many cases irrelavant text like email signatures. Ignore irrelevant text. Additionally, the only official 'Products' we track in this system filter for Windows 10, Windows 11, and Microsoft Edge, as such technologies tend to be outside of the core Microsoft product families. If you identify a technology, extract it as a technology entity. A Technology refers to any product, service, or platform that is relevant to the context but is not the locus or core issue. Focus on identifying technologies referenced or implied that could relate to the post. Attempt to extract separate variables for the `name`, `version`, `architecture`, and `build_number` fields. The `source_id` and `source_type` fields have been filled in for you. This extraction is optional, do not fabricate or hallucinate a technology if it is not stated, return an empty json dictionary if no technology is found. Finally, generate up to 3 tags or keywords that will help end-users search for the Technology and add them as a list to the output.\n"
                    "Provide your answer in the following JSON format:\n"
                    "\n"
                    "    {{\n"
                    '        "name": "Name of the technology.",\n'
                    '        "description": "A concise description of the technology and its relevance to the issue.",\n'
                    '        "source_id": "{source_id}",\n'
                    '        "source_type": "{source_type}",\n'
                    '        "tags": ["search", "prioritized", "terms"],\n'
                    '        "node_label": "Technology"\n'
                    "    }},\n"
                    "\n"
                    "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
                ),
                "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
            },
        },
    },
    "llama": {
        "MSRCPost": {
            "Symptom": {
                "user": "<placeholder for llama MSRCPost Symptom user prompt>",
                "system": "<placeholder for llama MSRCPost Symptom system prompt>",
            },
            "Cause": {
                "user": "<placeholder for llama MSRCPost Cause user prompt>",
                "system": "<placeholder for llama MSRCPost Cause system prompt>",
            },
            "Fix": {
                "user": "<placeholder for llama MSRCPost Fix user prompt>",
                "system": "<placeholder for llama MSRCPost Fix system prompt>",
            },
            "Tool": {
                "user": "<placeholder for llama MSRCPost Tool user prompt>",
                "system": "<placeholder for llama MSRCPost Tool system prompt>",
            },
            "Technology": {
                "user": "<placeholder for llama MSRCPost Technology user prompt>",
                "system": "<placeholder for llama MSRCPost Technology system prompt>",
            },
        },
        "PatchManagementPost": {
            "Symptom": {
                "user": "<placeholder for llama PatchManagementPost Symptom user prompt>",
                "system": "<placeholder for llama PatchManagementPost Symptom system prompt>",
            },
            "Cause": {
                "user": "<placeholder for llama PatchManagementPost Cause user prompt>",
                "system": "<placeholder for llama PatchManagementPost Cause system prompt>",
            },
            "Fix": {
                "user": "<placeholder for llama PatchManagementPost Fix user prompt>",
                "system": "<placeholder for llama PatchManagementPost Fix system prompt>",
            },
            "Tool": {
                "user": "<placeholder for llama PatchManagementPost Tool user prompt>",
                "system": "<placeholder for llama PatchManagementPost Tool system prompt>",
            },
            "Technology": {
                "user": "<placeholder for llama PatchManagementPost Technology user prompt>",
                "system": "<placeholder for llama PatchManagementPost Technology system prompt>",
            },
        },
    },
}
