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
or can be inferred.

The `post_type` key in the metadata determines whether a Fix should be extracted:
- If `post_type` is 'Information only', do not extract anything and return a null JSON object.
- If `post_type` is 'Problem statement', do not extract anything and return a null JSON object.
- If `post_type` is 'Solution provided', extract the Fix.
- If `post_type` is 'Helpful tool', do not extract anything and return a null JSON object.

### Definition of a Fix:
A Fix is a **technical action, configuration change, or patch** available to address the Cause or mitigate the Symptom. \
The Fix must be actionable and directly relevant to resolving the issue described in the CVE.

### Synthesis Instructions:
If the text context is incomplete but suggests a potential Fix:
1. **Generate a synthesized description or example**:
   - Use domain knowledge to create a plausible Fix.
   - If PowerShell, Bash, or another command-line solution is likely to resolve the issue, generate an appropriate code block.
   - Mark all synthesized content by appending **"(LLM Synthesized)"** to the description or code block.

2. **Add synthesized content tags**:
   - Include the tag **"LLM Synthesized"** in the `tags` list for all synthesized descriptions or code blocks.

3. **Adjust Reliability**:
   - Reduce the `reliability` score based on the degree of synthesis:
     - High (0.7–1.0): Minimal synthesis, largely based on explicit context.
     - Moderate (0.4–0.7): Partial synthesis, significant context inference.
     - Low (0.0–0.4): Mostly synthesized, very limited explicit context.

### Formatting for Human Readability:
To make the Fix description easier to read:
1. Use **headings** to indicate sections (e.g., "Fix Overview").
2. Use **numbered lists** to outline steps or methods, with line breaks between each item.
3. Clearly distinguish **preconditions** or important warnings (e.g., "Backup data before proceeding").
4. Explicitly mention update package URLs, products, build numbers, and KB Article IDs in the description.
5. You must explain the Fix in the description, it must standalone semantically from the parent entity it is extracted from and not implicitly referred to.

### Examples:
#### Valid Fix Extraction with Minimal Synthesis:
Input:
    "The KB5040442 update resolves an issue affecting Windows 11 by addressing kernel vulnerabilities. Apply the update through Windows Update."

Output:
    {{
        "fix_label": "ApplyKb5040442ToAddressKernelIssues",
        "description": "**Fix Overview**:\n\nTo address kernel vulnerabilities affecting Windows 11:\n\n1. Download and apply the KB5040442 update from the official Windows Update catalog.\n2. Restart your system after installation to ensure the changes take effect.\n\n**Details**:\n- This update resolves specific kernel vulnerabilities that could allow privilege escalation.\n- For manual installation, download the update package from: https://www.catalog.update.microsoft.com/Search.aspx?q=KB5040442.\n\n**Preconditions**:\n- Ensure the system is running Windows 11.\n- Backup critical data before proceeding with the update installation.\n",
        "tags": ["KB5040442", "Windows11", "KernelVulnerabilities"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Fix",
        "reliability": "0.9"
    }}

#### Valid Fix Extraction with Full Synthesis:
Input:
    "No further details provided, but a PowerShell script may resolve this vulnerability by reconfiguring system settings."

Output:
    {{
        "fix_label": "ReconfigureSystemSettingsWithPowerShell",
        "description": "**Fix Overview**:\n\nTo address this issue:\n\n1. Open PowerShell with administrative privileges.\n2. Run the following script to reconfigure system settings:\n\n```powershell\n# Reconfigure system settings\nSet-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned\nRestart-Service -Name wuauserv\nRestart-Computer -Force\n```\n\nThis script modifies execution policies and restarts key services to resolve system configuration issues. (LLM Synthesized)\n\n**Preconditions**:\n- Ensure administrative privileges are available.\n- Backup critical data before proceeding.\n",
        "tags": ["PowerShell", "SystemConfiguration", "LLM Synthesized"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Fix",
        "reliability": "0.6"
    }}

#### Invalid Fix Extraction (Problem Statement Misclassified as Fix):
Input:
    "This issue affects kernel security in Windows 11."
Output:
    {{
        "fix_label": null,
        "description": null,
        "tags": [],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": null
    }}

### Steps to Extract the Fix:
1. Identify the **specific technical action** or solution described in the text.
2. If the text is incomplete, synthesize a plausible Fix based on domain knowledge or similar resolutions, and clearly mark synthesized content.
3. Write a concise and standalone explanation of the Fix that:
   - Is at least 5 sentences long.
   - Includes headings (e.g., "**Fix Overview**") and numbered lists for clarity.
   - Mentions any important preconditions or warnings (e.g., "Backup data before proceeding").
4. Extract and include any products, build numbers, KB Article IDs, or update package links provided in the context.
5. Generate a six-word CamelCase label summarizing the Fix (e.g., 'ApplyKb5040442ToAddressKernelIssues').
6. Include up to 3 tags describing the Fix (e.g., ["KB5040442", "Windows11", "KernelVulnerabilities"]).

If no actionable Fix can be inferred or synthesized, return a null JSON object.

### Format for JSON Output:
Provide the following JSON format:
    {{
        "fix_label": "The 6 word CamelCase label for the Fix.",
        "description": "**Fix Overview**:\n\n<Provide an overview of the Fix.>\n\n1. <Step one of the Fix>\n2. <Step two of the Fix>\n\n**Details**:\n<Additional details about the Fix, including preconditions or important considerations.> (LLM Synthesized)\n",
        "tags": ["example_tag1", "example_tag2", "LLM Synthesized"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Fix",
        "reliability": "A score between 0.0 and 1.0 based on how much of the Fix is synthesized."
    }}

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
and determine if a tool is mentioned within the text. A Tool is any **actionable resource, software, script, or \
utility** that Microsoft or end users may use to diagnose, mitigate, or resolve the issue mentioned in the CVE. \
Tools must be actionable and related to resolving the issue. Focus on names, configurations, or commands necessary \
for addressing the symptoms or causes of the CVE.

The `post_type` key in the metadata determines whether a Tool should be attempted to be extracted:

- If `post_type` is 'Information only', do not extract anything and return a null JSON object.
- If `post_type` is 'Problem statement', do not extract anything and return a null JSON object.
- If `post_type` is 'Solution provided', attempt to extract a Tool. If there is no tool mentioned, return a null JSON object.
- If `post_type` is 'Helpful tool', extract the Tool.

### Definition of a Tool:
A Tool must meet the following criteria:
- **Actionable**: Must be a script, software, command-line utility, or clearly described resource that can be directly used to address the issue.
- **Relevant**: Must relate directly to the mitigation, diagnosis, or resolution of the CVE.
- **Isn't a fix**, but rather a resource that can be used to address the problem.
    1. If the text describes a Tool but the full explanation of its functionality belongs to the Fix:
    - Use the phrase: **"The detailed explanation for this Tool is provided in the Fix description."**
    - Avoid unclear terms like "this script" or "that function." Use the fixed phrase above to refer to the Fix clearly and consistently.
    - Briefly summarize the Tool's functionality while referring the reader to the Fix for detailed instructions.

    2. If the Tool is fully described in the text (and is not part of a Fix), provide a complete and standalone description, including:
    - Its functionality, purpose, and how it can be used.
    - Relevant details such as commands, parameters, or usage examples.

### Exclusions (What Is Not a Tool):
Do not extract the following as a Tool:
- Descriptions of vulnerabilities, CVEs, or security issues (e.g., bypasses, exploits, weaknesses).
- Generic concepts, issues, or recommendations without actionable content (e.g., "enable multifactor authentication").
- Any text lacking explicit or implied actions that a user can take.

### Format Conventions:
- tool_label:Tools should be named using CamelCase, with no spaces.
- description: A detailed description of the tool, including its purpose, configuration, and usage. Minimum of 5 sentences. You must explain the tool in the description, it must standalone semantically from the parent entity it is extracted from and not implicitly referred to.
- tool_url: A URL to the tool's documentation or website or download location, if available.
- tags: A set of keywords or tags that describe the tool, such as "PowerShell", "debugging", "logs", "LLM Synthesized".
- source_url: the link to the patch management post where the text was extracted.
- source_id: the id of the patch management post where the text was extracted.

### Instructions for Synthesizing Tools:
If a tool is incomplete or ambiguously described in the text:
1. **Synthesize a Plausible Description**:
   - Use the surrounding context to infer a likely purpose, usage, or configuration of the tool.
   - Mark synthesized content explicitly by appending "(LLM Synthesized)" to the description.

2. **Include Synthesized Examples**:
   - If appropriate, generate a synthesized code block, script, or usage example.
```powershell
# Synthesized example of PowerShell script for extracting download links from Windows Update logs
Get-Content -Path "C:\\Windows\\Logs\\WindowsUpdate\\WU.log" | Select-String -Pattern "http.*\\.appx" | ForEach-Object {{
    $_ -match "(http.*\\.appx)"
    Out-File -FilePath "C:\\DownloadLinks\\links.txt" -Append
}}
   - Add the tag `LLMSynthesized` to the `tags` list.

3. **Adjust Reliability**:
   - Reduce the `reliability` score based on the degree of synthesis:
     - High (0.7-1.0): Minimal synthesis, largely based on explicit context.
     - Moderate (0.4-0.7): Partial synthesis, significant context inference.
     - Low (0.0-0.4): Mostly synthesized, very limited explicit context.

4. **Use URLs to Strengthen Plausibility**:
   - If a URL is mentioned in the text, assume it is relevant to the tool and use it to inform the synthesized description.

### Examples:
#### Valid Tool Extraction:
Input: "I used PowerShell to retrieve logs for debugging this issue."
Output:
    {{
        "tool_label": "RetrieveLogsUsingPowerShell",
        "description": "A PowerShell script used to retrieve logs for debugging Windows Update issues. [Some powershell script block]",
        "tool_url": "https://example.com/powershell-script",
        "source_url": "//groups.google.com/d/msgid/patchmanagement/SA0PR09MB7116CACF268980E52ACE8ED9A5AEA%40SA0PR09MB7116.namprd09.prod.outlook.com",
        "source_id": "{{source_id}}",
        "source_type": "{{source_type}}",
        "tags": ["PowerShell", "debugging", "logs", "LLM Synthesized"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Tool",
        "reliability": "0.8",
        "severity_type": "moderate"
    }}

#### Invalid Tool Extraction:
The following are examples of invalid extractions for each extracted property.
Input: "A vulnerability in Windows Smartscreen that allows attackers to bypass security checks."
Output:
    {{
        "tool_label": "Windows Smartscreen Security Bypass",
        "description": "A vulnerability in the Windows Smartscreen feature that allows attackers to bypass security checks. This vulnerability has a high CVSS rating of 8.8 and requires user interaction to exploit. It affects Microsoft Windows 10 and later versions.",
        "tool_url": "//groups.google.com/d/msgid/patchmanagement/d9d32576-bf61-4e1e-ac57-910bd685a64cn%40googlegroups.com",
        "source_id": "{{source_id}}",
        "source_type": "{{source_type}}",
        "source_url": nan,
        "tags": ["vulnerability","security", "smartscreen"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": null,
        "reliability": "0.0",
        "severity_type": null
    }}

### Handling Ambiguities:
If a tool is incomplete or ambiguously described in the text, you may cautiously synthesize a plausible \
representation of the tool while making it clear that the synthesized portion is an LLM-generated assumption. \
For example, if the author writes, "I use the WU logs with some PowerShell code to extract the download links of Appx applications," respond with: \
"The author mentions using PowerShell to extract download links from Windows Update logs. Below is a synthesized PowerShell script to illustrate this process (LLM Synthesized):

```powershell
# Synthesized example of PowerShell script for extracting download links from Windows Update logs
Get-Content -Path "C:\\Windows\\Logs\\WindowsUpdate\\WU.log" | Select-String -Pattern "http.*\\.appx" | ForEach-Object {{
    $_ -match "(http.*\\.appx)"
    Out-File -FilePath "C:\\DownloadLinks\\links.txt" -Append
}}```
---------------------
<start context>
{context_str}
<end context>
---------------------
Provide your answer in the following JSON format:

    {{
        "tool_label": "The 6 word CamelCase label for the tool, no spaces.",
        "description": "A concise description of the tool and its purpose, minimum of 5 sentences.",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "source_url": "//groups.google.com/d/msgid/patchmanagement/123456789/",
        "tool_url": "https://reddit.com/r/Windows10/comments/123456789/",
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
signatures. The `post_type` key in the metadata determines whether a Fix should be extracted:

- If `post_type` is 'Information only', do not extract anything and return a null JSON object.
- If `post_type` is 'Problem statement', do not extract anything and return a null JSON object.
- If `post_type` is 'Solution provided', extract the Fix.
- If `post_type` is 'Helpful tool', do not extract anything and return a null JSON object.

### Definition of a Fix:
A Fix is a **technical action, configuration change, or patch** available to address the Cause or mitigate the Symptom. \
The Fix must be actionable and directly relevant to the described problem or solution.

### Synthesis Instructions:
If the text context is incomplete but suggests a potential Fix:
1. **Generate a synthesized description or example**:
   - Use domain knowledge to create a plausible Fix.
   - If PowerShell, Bash, or another command-line solution is likely to resolve the issue, generate an appropriate code block.
   - Mark all synthesized content by appending **"(LLM Synthesized)"** to the description or code block.

2. **Add synthesized content tags**:
   - Include the tag **"LLM Synthesized"** in the `tags` list for all synthesized descriptions or code blocks.

3. **Adjust Reliability**:
   - Reduce the `reliability` score based on the degree of synthesis:
     - High (0.7-1.0): Minimal synthesis, largely based on explicit context.
     - Moderate (0.4-0.7): Partial synthesis, significant context inference.
     - Low (0.0-0.4): Mostly synthesized, very limited explicit context.

### Formatting for Human Readability:
To make the Fix description easier to read:
1. Use **headings** to indicate sections (e.g., "Fix Overview").
2. Use **numbered lists** to outline steps or methods, with line breaks between each item.
3. Clearly distinguish **preconditions** or important warnings (e.g., "Backup data before proceeding").
4. You must explain the Fix in the description, it must standalone semantically from the parent entity it is extracted from and not implicitly referred to.

### Examples:
#### Valid Fix Extraction with Minimal Synthesis:
Input:
    "Two 'brute force' things I try next are either upgrade to the next version via ISO (23H2) or do an in-place upgrade via ISO."

Output:
    {{
        "fix_label": "UpgradeWindowsViaISOToResolveIssues",
        "description": "**Fix Overview**:\n\nTo address installation issues with Windows 11:\n\n1. Perform an upgrade to the next version via ISO (23H2).\n2. Alternatively, conduct an in-place upgrade via ISO to install 22H2 over the existing setup.\n\n**Details**:\n- The in-place upgrade retains all user data while refreshing the operating system, akin to a feature update.\n- This method removes and reinstalls Windows, as well as applications under user profiles.\n\n**Preconditions**:\n- Backup important data before proceeding with either method to prevent data loss.\n",
        "tags": ["Windows11", "ISOUpgrade", "InstallationIssues"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Fix",
        "reliability": "0.9"
    }}

#### Valid Fix Extraction with Full Synthesis:
Input:
    "No further details provided, but PowerShell can likely resolve this type of installation issue."

Output:
    {{
        "fix_label": "ResolveInstallationIssueUsingPowerShell",
        "description": "**Fix Overview**:\n\nWhen encountering installation issues:\n\n1. Open PowerShell with administrative privileges.\n2. Use the following command to reset the Windows Update components:\n\n```powershell\n# Reset Windows Update components\nnet stop wuauserv\nnet stop cryptSvc\nnet stop bits\nnet stop msiserver\nren C:\\Windows\\SoftwareDistribution SoftwareDistribution.old\nren C:\\Windows\\System32\\catroot2 catroot2.old\nnet start wuauserv\nnet start cryptSvc\nnet start bits\nnet start msiserver\n```\n\nThis command sequence resets the update components, which may resolve installation issues. (LLM Synthesized)\n\n**Preconditions**:\n- Ensure administrative privileges are available.\n- Backup critical data before proceeding.\n",
        "tags": ["PowerShell", "WindowsUpdate", "LLM Synthesized"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Fix",
        "reliability": "0.6"
    }}

#### Invalid Fix Extraction (Problem Statement Misclassified as Fix):
Input:
    "My system crashes when I apply this update."
Output:
    {{
        "fix_label": null,
        "description": null,
        "tags": [],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": null
    }}

### Steps to Extract the Fix:
1. Identify the **specific technical action** or solution described in the text.
2. If the text is incomplete, synthesize a plausible Fix based on domain knowledge or similar resolutions, and clearly mark synthesized content.
3. Write a concise and standalone explanation of the Fix that:
   - Is at least 5 sentences long.
   - Includes headings (e.g., "**Fix Overview**") and numbered lists for clarity.
   - Mentions any important preconditions or warnings (e.g., "Backup data before proceeding").
4. Extract and include any products, build numbers, KB Article IDs, or update package links provided in the context.
5. Generate a six-word CamelCase label summarizing the Fix (e.g., 'UpgradeWindowsViaISOToResolveIssues').
6. Include up to 3 tags describing the Fix (e.g., ["Windows11", "ISOUpgrade", "InstallationIssues"]).

If no actionable Fix can be inferred or synthesized, return a null JSON object.

### Format for JSON Output:
Provide the following JSON format:
    {{
        "fix_label": "The 6 word CamelCase label for the Fix.",
        "description": "**Fix Overview**:\n\n<Provide an overview of the Fix.>\n\n1. <Step one of the Fix>\n2. <Step two of the Fix>\n\n**Details**:\n<Additional details about the Fix, including preconditions or important considerations.> (LLM Synthesized)\n",
        "tags": ["example_tag1", "example_tag2", "LLM Synthesized"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Fix",
        "reliability": "A score between 0.0 and 1.0 based on how much of the Fix is synthesized."
    }}
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
        "fix_label": "The 6 word CamelCase label for the fix with no spaces.",
        "tags": ["search", "prioritized", "terms", "LLM synthesized"],
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
                """Given the following Patch Management post and its metadata, attempt to extract any Tool described or inferred.
Patch Management posts are often very limited in their text and contain irrelevant text in the form of email
signatures. Tools are not required to extract; it is important you do not fabricate or hallucinate a tool if it
doesn't exist. The `post_type` key in the metadata determines whether a Tool should be attempted to be extracted:

- If `post_type` is 'Information only', do not extract anything and return a null JSON object.
- If `post_type` is 'Problem statement', do not extract anything and return a null JSON object.
- If `post_type` is 'Solution provided', attempt to extract a Tool. If there is no tool mentioned, return a null JSON object.
- If `post_type` is 'Helpful tool', extract the Tool.

### Definition of a Tool:
A Tool must meet the following criteria:
- **Actionable**: Must be a script, software, command-line utility, or clearly described resource that can be directly used to address the issue.
- **Relevant**: Must relate directly to the mitigation, diagnosis, or resolution of the CVE.
- **Isn't a fix**, but rather a resource that can be used to address the problem.
    1. If the text describes a Tool but the full explanation of its functionality belongs to the Fix:
    - Use the phrase: **"The detailed explanation for this Tool is provided in the Fix description."**
    - Avoid unclear terms like "this script" or "that function." Use the fixed phrase above to refer to the Fix clearly and consistently.
    - Briefly summarize the Tool's functionality while referring the reader to the Fix for detailed instructions.

    2. If the Tool is fully described in the text (and is not part of a Fix), provide a complete and standalone description, including:
    - Its functionality, purpose, and how it can be used.
    - Relevant details such as commands, parameters, or usage examples.

### Exclusions (What Is Not a Tool):
Do not extract the following as a Tool:
- Descriptions of vulnerabilities, CVEs, or security issues (e.g., bypasses, exploits, weaknesses).
- Generic concepts, issues, or recommendations without actionable content (e.g., "enable multifactor authentication").
- Any text lacking explicit or implied actions that a user can take.

### Format Conventions:
- tool_label: Tools should be named using CamelCase, with no spaces.
- description: A detailed description of the tool, including its purpose, configuration, and usage. Minimum of 5 sentences. You must explain the tool in the description, it must standalone semantically from the parent entity it is extracted from and not implicitly referred to.
- tool_url: A URL to the tool's documentation or website or download location, if available.
- tags: A set of keywords or tags that describe the tool, such as "PowerShell", "debugging", "logs", "LLM Synthesized".
- source_url: the link to the patch management post where the text was extracted.
- source_id: the id of the patch management post where the text was extracted.

### Instructions for Synthesizing Tools:
If a tool is incomplete or ambiguously described in the text:
1. **Synthesize a Plausible Description**:
   - Use the surrounding context to infer a likely purpose, usage, or configuration of the tool.
   - Mark synthesized content explicitly by appending "(LLM Synthesized)" to the description.

2. **Include Synthesized Examples**:
   - If appropriate, generate a synthesized code block, script, or usage example.
```powershell
# Synthesized example of PowerShell script for extracting download links from Windows Update logs
Get-Content -Path "C:\\Windows\\Logs\\WindowsUpdate\\WU.log" | Select-String -Pattern "http.*\\.appx" | ForEach-Object {{
    $_ -match "(http.*\\.appx)"
    Out-File -FilePath "C:\\DownloadLinks\\links.txt" -Append
}}
   - Add the tag `LLMSynthesized` to the `tags` list.

3. **Adjust Reliability**:
   - Reduce the `reliability` score based on the degree of synthesis:
     - High (0.7-1.0): Minimal synthesis, largely based on explicit context.
     - Moderate (0.4-0.7): Partial synthesis, significant context inference.
     - Low (0.0-0.4): Mostly synthesized, very limited explicit context.

4. **Use URLs to Strengthen Plausibility**:
   - If a URL is mentioned in the text, assume it is relevant to the tool and use it to inform the synthesized description.

### Examples:
#### Valid Tool Extraction:
Input: "I used PowerShell to retrieve logs for debugging this issue."
Output:
    {{
        "tool_label": "RetrieveLogsUsingPowerShell",
        "description": "A PowerShell script used to retrieve logs for debugging Windows Update issues. [Some powershell script block]",
        "tool_url": "https://example.com/powershell-script",
        "source_url": "//groups.google.com/d/msgid/patchmanagement/SA0PR09MB7116CACF268980E52ACE8ED9A5AEA%40SA0PR09MB7116.namprd09.prod.outlook.com",
        "source_id": "{{source_id}}",
        "source_type": "{{source_type}}",
        "tags": ["PowerShell", "debugging", "logs", "LLM Synthesized"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": "Tool",
        "reliability": "0.8",
        "severity_type": "moderate"
    }}

#### Invalid Tool Extraction:
The following are examples of invalid extractions for each extracted property.
Input: "A vulnerability in Windows Smartscreen that allows attackers to bypass security checks."
Output:
    {{
        "tool_label": "Windows Smartscreen Security Bypass",
        "description": "A vulnerability in the Windows Smartscreen feature that allows attackers to bypass security checks. This vulnerability has a high CVSS rating of 8.8 and requires user interaction to exploit. It affects Microsoft Windows 10 and later versions.",
        "tool_url": "//groups.google.com/d/msgid/patchmanagement/d9d32576-bf61-4e1e-ac57-910bd685a64cn%40googlegroups.com",
        "source_id": "{{source_id}}",
        "source_type": "{{source_type}}",
        "source_url": nan,
        "tags": ["vulnerability","security", "smartscreen"],
        "node_id": "placeholder_node_id_leave_as_is",
        "node_label": null,
        "reliability": "0.0",
        "severity_type": null
    }}

### Handling Ambiguities:
If a tool is incomplete or ambiguously described in the text, you may cautiously synthesize a plausible \
representation of the tool while making it clear that the synthesized portion is an LLM-generated assumption. \
For example, if the author writes, "I use the WU logs with some PowerShell code to extract the download links of Appx applications," respond with: \
"The author mentions using PowerShell to extract download links from Windows Update logs. Below is a synthesized PowerShell script to illustrate this process (LLM Synthesized):

```powershell
# Synthesized example of PowerShell script for extracting download links from Windows Update logs
Get-Content -Path "C:\\Windows\\Logs\\WindowsUpdate\\WU.log" | Select-String -Pattern "http.*\\.appx" | ForEach-Object {{
    $_ -match "(http.*\\.appx)"
    Out-File -FilePath "C:\\DownloadLinks\\links.txt" -Append
}}```
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
        "description": "A concise description of the tool, its purpose, and any synthesized details.",
        "synthesized_code": "Include only if code was synthesized, with proper disclaimers.",
        "source_id": "{source_id}",
        "source_type": "{source_type}",
        "source_url": "//groups.google.com/d/msgid/patchmanagement/...",
        "tool_url": "https://www.reddit.com/r/SCCM/comments/...",
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
