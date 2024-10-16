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


PROMPT_TEMPLATES: Dict[str, str] = {}

# Symptom extraction from MSRCPost
PROMPT_TEMPLATES = {
    "MSRCPost": {
        "Symptom": {
            "user": (
                "Given the following Microsoft Security Response Center (MSRC) post, extract the core Symptom that is described or can be inferred. A Symptom is an observable behavior, an error message, or any indication that something is going wrong in the system, as experienced by end users or system administrators. **It is not a vulnerability or technical exploit, but rather what the user notices as a result of the underlying issue.** For example, a Symptom could be system crashes, poor performance, unexpected reboots, failed updates, or error messages seen by the user. Symptoms help people identify issues that are affecting their environments. \nDo not describe vulnerabilities, exploits, or technical flaws directly. Instead, describe the **impact or observable behavior** that a system administrator or end user would see as a consequence of the security issue. Focus on the **user's perspective**, not the attacker's. \nBe thorough and consider subtle aspects that may not be explicitly stated. Describe how the security update affects a particular system or software product from the perspective of the end user. For instance:\n- 'The computer fails to boot after installing a security patch.'\n'Network communication is intermittently lost on systems using the affected driver.'\n- 'The system experiences slow performance and occasional reboots.'\nDo not restate or reference the original post directly; the Symptom should stand alone and specify the **observable behavior** or **impact** rather than describing the vulnerability itself. For each Symptom, generate a six-word CamelCase label, e.g., 'WindowsFailsToBootWithSignedWdacPolicy'. **Avoid including words such as 'vulnerability', 'exploit', or 'security'** in the label. Generate a complete and thorough description of at least 3 sentences. The description should stand alone as a precise explanation of the Symptom, and must include the products, build numbers and KB Article ids (eg., KB5040442) provided in the context. \nThe `source_id` and `source_type` fields have been filled in already. Finally, generate up to 3 tags or keywords that describe the Symptom and add them as a list to the output.\n"
                "---------------------\n"
                "<start context>\n"
                "{context_str}\n"
                "\n"
                "Build Numbers:\n"
                "{build_numbers}\n"
                "KB Articles:\n"
                "{kb_ids}\n"
                "<end context>\n"
                "---------------------\n"
                "Provide your answer in the following JSON format:\n"
                "\n"
                "    {{\n"
                '        "symptom_label": "6 word camel case label for the symptom",\n'
                '        "description": "A complete and precise technical description of the symptom. Be as specific as possible. Include build numbers, kb ids, and update package links.",\n'
                '        "source_id": "{source_id}",\n'
                '        "source_type": "{source_type}",\n'
                '        "tags": ["search", "prioritized", "terms"],\n'
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
                '        "node_label": "Symptom"\n'
                '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the symptom describes the CVE.",\n'
                '        "severity_type": "one of [critical, important, moderate, low]",\n'
                "    }},\n"
                "\n"
                "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
            ),
            "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
        },
        "Cause": {
            "user": (
                "Given the following Microsoft Security Response Center (MSRC) post, extract the core Cause that is described or can be inferred. A Cause is any technical or situational condition responsible for the Symptom or issue described in the CVE. This could be a flaw in software, a misconfiguration, or any contributing factor that underpins the identified Symptom. Focus on the technical reasons driving the issue, and avoid restating the full text of the post. Provide a complete description of the Cause that would allow Microsoft experts to trace back the root of the issue. Ensure to describe the what, where, and how. For each Cause, generate a six-word CamelCase label, e.g., 'IncorrectRegistryValueCausesRebootError'. **Avoid including words such as 'Cause', 'vulnerability', 'exploit', or 'security'** in the label.  Generate a complete and thorough description of at least 3 sentences. The description should stand alone as a precise explanation of the Cause, and must include the products, build numbers and KB Article ids (eg., KB5040442) provided in the context. \nThe `source_id` and `source_type` fields have been filled in for you. Finally, generate up to 3 tags or keywords that describe the Cause and add that as a list to the output\n"
                "---------------------\n"
                "<start context>\n"
                "{context_str}\n"
                "\n"
                "Build Numbers:\n"
                "{build_numbers}\n"
                "KB Articles:\n"
                "{kb_ids}\n"
                "<end context>\n"
                "---------------------\n"
                "Provide your answer in the following JSON format:\n"
                "    {{\n"
                '        "description": "A concise, thorough and precise technical description of the cause. Include the build numbers and kb ids contained in the context.",\n'
                '        "cause_label": "The 6 word CamelCase label for the cause",\n'
                '        "source_id": "{source_id}",\n'
                '        "source_type": "{source_type}",\n'
                '        "tags": ["search", "prioritized", "terms"],\n'
                '        "node_label": "Cause"\n'
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
                '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the cause is the true and canonical source of the CVE.",\n'
                '        "severity_type": "one of [critical, important, moderate, low]",\n'
                "    }},\n"
                "\n"
                "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
            ),
            "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
        },
        "Fix": {
            "user": (
                "Given the following Microsoft Security Response Center (MSRC) post, extract the core Fix that is described or can be inferred. "
                "In the document metadata is the key `post_type`. If its value is 'Solution provided', the post likely contains an explicit fix. "
                "If the value is 'Critical', then there is likely no fix. A Fix is a technical action, configuration change, or patch available to address the Cause or mitigate the Symptom. "
                "Focus on the specific technical response to the issue affecting the CVE, referencing affected systems or products without repeating the post's full text. "
                "Generate a complete and thorough description of at least 3 sentences. The description should stand alone as a precise explanation of the steps necessary to implement the fix, it must include the products, build numbers,KB Article ids (eg., KB5040442), and update package links provided in the context."
                "For each Fix, generate a six-word CamelCase label and provide a concise description, e.g., 'ApplyKbPatchToResolveRebootIssue'. "
                "**Avoid including words such as 'Fix', 'vulnerability', 'exploit', or 'security'** in the label. The `source_id` and `source_type` fields have been filled in for you. "
                "Finally, generate up to 3 tags or keywords that describe the Fix and add them as a list to the output. "
                "Ensure the update package urls are explicitly mentioned in the `description` field. "
                "This extraction is optional, do not fabricate or hallucinate a fix if it is not stated. Return an empty JSON dictionary if no fix is found.\n"
                "---------------------\n"
                "<start context>\n"
                "{context_str}\n"
                "\n"
                "Build Numbers:\n"
                "{build_numbers}\n"
                "KB Articles:\n"
                "{kb_ids}\n"
                "Update Packages:\n"
                "{update_package_urls}\n"
                "<end context>\n"
                "---------------------\n"
                "Provide your answer in the following JSON format:\n"
                "\n"
                "    {{\n"
                '        "description": "A concise and precise technical description of the fix, including the build numbers and update package links.",\n'
                '        "source_id": "{source_id}",\n'
                '        "source_type": "{source_type}",\n'
                '        "fix_label": "The 6-word CamelCase label for the fix",\n'
                '        "tags": ["search", "prioritized", "terms"],\n'
                '        "node_label": "Fix",\n'
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
                '        "reliability": "A floating score between 0.0 and 1.0 that indicates how reliably the fix mitigates or solves the CVE.",\n'
                '        "severity_type": "One of [critical, important, moderate, low]",\n'
                "    }},\n"
                "\n"
                "Do not output any other dialog or text outside of the JSON object. The JSON object must be valid."
            ),
            "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
        },
        "Tool": {
            "user": (
                "Given the following Microsoft Security Response Center (MSRC) post that has been filtered to reference only Windows 10, Windows 11, and Microsoft Edge products. Proceed in two steps. First, evaluate the context and determine if a tool is mentioned within the text. A Tool is any software, command-line utility, or similar that Microsoft or end users may use to diagnose, mitigate, or resolve the issue mentioned in the CVE. Focus on names, configurations, or commands necessary for addressing the symptoms or causes of the CVE. If you find a tool, generate a six-word CamelCase label, e.g., 'EnableFirewallUsingPowerShellCmd'. **Avoid including words such as 'Tool', 'vulnerability', 'exploit', or 'security'** in the label. Generate a complete description of the tool of at least 3 sentences including how an end-user may find or use the tool. Evaluate the text, look for an external URL for the tool. The tool URL is NOT the `source` url from the document metadata. Do not insert the document's source url into the `tool_url` field. The `source_id` and `source_type` fields have been filled in for you. This extraction is optional, do not fabricate or hallucinate a tool if it is not stated, return an empty json dictionary if no tool is found. Finally, generate up to 3 tags or keywords that will help end-users search for the Tool and add them as a list to the output.\n"
                "---------------------\n"
                "<start context>\n"
                "{context_str}\n"
                "<end context>\n"
                "---------------------\n"
                "Provide your answer in the following JSON format:\n"
                "\n"
                "    {{\n"
                '        "name": "Name of the tool.",\n'
                '        "description": "A concise description of the tool and its purpose.",\n'
                '        "source_id": "{source_id}",\n'
                '        "source_type": "{source_type}",\n'
                '        "tool_url": "If a URL for a tool is mentioned, included it here.",\n'
                '        "tags": ["search", "prioritized", "terms"],\n'
                '        "node_label": "Tool"\n'
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
                "    }},\n"
                "\n"
                "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
            ),
            "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
        },
        "Technology": {
            "user": (
                "Given the following Microsoft Security Response Center (MSRC) post, evaluate it to determine if a tertiary technology is referenced in relation to the CVE. If there is, extract it as a technology entity. A Technology refers to any Microsoft product, service, or platform that is relevant to the context. Technologies are separate from the core Microsoft product families. Focus on identifying technologies referenced or implied that could relate to the post. Attempt to extract separate pieces for the `name`, `version`, `architecture`, and `build_number` fields. Generate a complete and thorough description of at least 3 sentences.  This entity extraction is optional therefore do not fabricate or hallucinate a technology, if there is no technology mentioned return an empty json dictionary. The `source_id` and `source_type` fields have been filled in for you. Finally, generate up to 3 tags or keywords that describe the Technology and add them as a list to the output.\n"
                "<start context>\n"
                "{context_str}\n"
                "\n"
                "Build Numbers:\n"
                "{build_numbers}\n"
                "<end context>\n"
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
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
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
                "A Symptom is an observable behavior, error message, or any indication that something is going wrong in the system, especially as experienced by end users or system administrators. Symptoms help people identify issues affecting their environments. Be thorough and consider subtle aspects that may not be explicitly stated. Describe how the post affects a particular system or software product (e.g., How an attacker gains access to a firewall). Do not restate or reference the original post; the Symptom should stand alone and specify the product or technology affected by the post. For each Symptom, generate a six-word CamelCase label, e.g., 'WindowsFailsToBootWithSignedWdacPolicy'. **Avoid including words such as 'Symptom', 'vulnerability', 'exploit', or 'security'** in the label. Generate a complete and thorough description of at least 3 sentences. The `source_id` and `source_type` fields have been filled in for you. Note. When estimating the reliability of the post, factor in to the rating: language usage, the mastery of the topic, the completeness of the answer and supporting links. Finally, generate up to 3 tags or keywords that describe the Symptom and add that as a list to the output.\n"
                "---------------------\n"
                "<start context>\n"
                "{context_str}\n"
                "\n"
                "Build Numbers:\n"
                "{build_numbers}\n"
                "KB Articles:\n"
                "{kb_ids}\n"
                "<end context>\n"
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
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
                '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the symptom describes the core issue in the Patch Management post.",\n'
                '        "severity_type": "one of [critical, important, moderate, low]",\n'
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
                "Be thorough and consider subtle aspects that may not be explicitly stated. Provide a concise description of the Cause that would allow Microsoft experts to trace back the root of the issue. For each Cause, generate a six-word CamelCase label, e.g., 'IncorrectRegistryValueYieldsRebootError'. **Avoid including words such as 'Cause', 'vulnerability', 'exploit', or 'security'** in the label. Generate a complete and thorough description of at least 3 sentences."
                "Patch Management posts are often very limited in their text and contain irrelevant text in the form of email signatures. All your answers must be technically correct in the domain of Microsoft patches and systems.\n"
                "Note. When estimating the reliability of the post, factor in to the rating: language usage, the mastery of the topic, the completeness of the answer and supporting links.\n"
                "The `source_id` and `source_type` fields have been filled in for you."
                "---------------------\n"
                "<start context>\n"
                "{context_str}\n"
                "\n"
                "Build Numbers:\n"
                "{build_numbers}\n"
                "KB Articles:\n"
                "{kb_ids}\n"
                "<end context>\n"
                "---------------------\n"
                "Provide your answer in the following JSON format:\n"
                "\n"
                "    {{\n"
                '        "description": "A concise and precise technical description of the cause.",\n'
                '        "source_id": "{source_id}",\n'
                '        "source_type": "{source_type}",\n'
                '        "cause_label": "The 6 word CamelCase label for the cause",\n'
                '        "tags": ["search", "prioritized", "terms"],\n'
                '        "node_label": "Cause"\n'
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
                '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the cause explains and solves the core issue in the Patch Management post.",\n'
                '        "severity_type": "one of [critical, important, moderate, low]",\n'
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
                "A Fix is a technical action, configuration change, or patch available to address the Cause or mitigate the Symptom. Focus on the specific technical response to the Cause, referencing affected systems or products without repeating the post's full text. This description should stand alone as a precise explanation of the Fix. For each Fix, generate a six-word CamelCase label and provide a concise description, e.g., 'ApplyKbPatchToResolveRebootIssue'. **Avoid including words such as 'Fix', 'vulnerability', 'exploit', or 'security'** in the label. Generate a complete and thorough description of at least 3 sentences. Note. When estimating the reliability of the post, factor in to the rating: language usage, the mastery of the topic, the completeness of the answer and supporting links. Finally, generate up to 3 tags or keywords that describe the Fix and add that as a list to the output.\n"
                "---------------------\n"
                "<start context>\n"
                "{context_str}\n"
                "\n"
                "Build Numbers:\n"
                "{build_numbers}\n"
                "KB Articles:\n"
                "{kb_ids}\n"
                "<end context>\n"
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
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
                '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the fix explains and solves the core issue in the Patch Management post.",\n'
                '        "severity_type": "one of [critical, important, moderate, low]",\n'
                "    }},\n"
                "\n"
                "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
            ),
            "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
        },
        "Tool": {
            "user": (
                "Given the following Patch Management post, attempt to extract any useful tool. Your goals are to identify and, if possible, extract any tool mentioned in the text. Note. the text is messy email text with poor grammar and in many cases irrelavant text like email signatures. Ignore irrelevant text. Proceed in two steps. First, evaluate the context and determine if a tool is mentioned within the text. Inspect the post metadata for a key `post_type`, if it is 'Helpful tool' then a tool has been detected in the text. A Tool is any software, command-line utility, or similar that Microsoft or end users may use to diagnose, mitigate, or resolve the issue. Focus on names, configurations, or commands necessary for addressing the symptoms or causes of the CVE. If you find a tool, generate a six-word CamelCase label, e.g., 'EnableFirewallUsingPowerShellCmd'. **Avoid including words such as 'Tool', 'vulnerability', 'exploit', or 'security'** in the label. Generate a complete description of the tool and how an end-user may find or use the tool. The `source_id`,  `source_type`, and `node_label` fields have been filled in for you. Do not modify the node_label value. This extraction is optional, do not fabricate or hallucinate a tool if it is not stated, return an empty json dictionary if no tool is found.\nNote. When estimating the reliability of the post, factor in to the rating: language usage, the mastery of the topic, the completeness of the answer and supporting links.\n Finally, generate up to 3 tags or keywords that will help end-users search for the Tool and add them as a list to the output.\n"
                "---------------------\n"
                "<start context>\n"
                "{context_str}\n"
                "\n"
                "Build Numbers:\n"
                "{build_numbers}\n"
                "<end context>\n"
                "---------------------\n"
                "Provide your answer in the following JSON format:\n"
                "\n"
                "    {{\n"
                '        "tool_label": "Name of the tool.",\n'
                '        "description": "A concise description of the tool and its purpose.",\n'
                '        "source_id": "{source_id}",\n'
                '        "source_type": "{source_type}",\n'
                '        "source_url": "If a URL is included assign it here.",\n'
                '        "tags": ["search", "prioritized", "terms"],\n'
                '        "node_label": "Tool"\n'
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
                '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the tool contributes to mitigating or supporting the work needed to resolve the core issue in the Patch Management post.",\n'
                '        "severity_type": "one of [critical, important, moderate, low]",\n'
                "    }},\n"
                "\n"
                "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
            ),
            "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
        },
        "Technology": {
            "user": (
                "Given the following Patch Management post, evaluate it and determine if a tertiary technology is mentioned in relation to the central topic of the post. Note. the text is messy email text with poor grammar and in many cases irrelavant text like email signatures. Ignore irrelevant text. Additionally, the only official 'Products' we track in this system filter for Windows 10, Windows 11, and Microsoft Edge, as such technologies tend to be outside of the core Microsoft product families. If you identify a technology, extract it as a technology entity. A Technology refers to any product, service, or platform that is relevant to the context but is not the locus or core issue. Focus on identifying technologies referenced or implied that could relate to the post. Attempt to extract separate variables for the `name`, `version`, `architecture`, and `build_number` fields. The `source_id`,  `source_type`, and `node_label` fields have been filled in for you. Do not modify the node_label value.\nNote. When estimating the reliability of the post, factor in to the rating: language usage, the mastery of the topic, the completeness of the answer and supporting links.\n This extraction is optional, do not fabricate or hallucinate a technology if it is not stated, return an empty json dictionary if no technology is found. Finally, generate up to 3 tags or keywords that will help end-users search for the Technology and add them as a list to the output.\n"
                "---------------------\n"
                "<start context>\n"
                "{context_str}\n"
                "\n"
                "Build Numbers:\n"
                "{build_numbers}\n"
                "KB Articles:\n"
                "{kb_ids}\n"
                "<end context>\n"
                "---------------------\n"
                "Provide your answer in the following JSON format:\n"
                "\n"
                "    {{\n"
                '        "name": "Name of the technology.",\n'
                '        "description": "A concise description of the technology and its relevance to the issue.",\n'
                '        "source_id": "{source_id}",\n'
                '        "source_type": "{source_type}",\n'
                '        "tags": ["search", "prioritized", "terms"],\n'
                '        "node_label": "Technology"\n'
                '        "node_id": "placeholder_node_id_leave_as_is",\n'
                '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the Technology is described in the Patch Management post.",\n'
                '        "severity_type": "one of [critical, important, moderate, low]",\n'
                "    }},\n"
                "\n"
                "Do not output any other dialog or text outside of the json object. The JSON object must be valid."
            ),
            "system": "You are an expert microsoft system administrator, with experience in enterprise scale on-premise and cloud deployments using configuration manager, Azure, Intune, device management and various forms of patch management. You are also an expert in building knowledge graphs and generating Cypher queries to create nodes and edges in graph databases. You are tasked with extracting conceptual entity information from the text below and generating a valid json object to store your extracted entities. You are looking for larger concepts like symptoms, causes and software products. Use a tone that is thoughtful and engaging and helps the audience make sense of the text. The audience is a technical audience of system administrators and security professionals who want to quickly scan high level details related to each post.",
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
                    '        "node_label": "Cause"\n'
                    '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the cause is the true and canonical source of the issue described in the Patch Management post.",\n'
                '        "severity_type": "one of [critical, important, moderate, low]",\n'
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
                    '        "node_label": "Fix"\n'
                    '        "reliability": "a floating score between 0.0 and 1.0 that indicates how reliably the fix mitigates or solves the core issue in the Patch Management post.",\n'
                '        "severity_type": "one of [critical, important, moderate, low]",\n'
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
