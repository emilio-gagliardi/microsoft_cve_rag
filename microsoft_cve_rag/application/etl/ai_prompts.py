class AIMigrationPrompts:
    PROMPTS = {
        "AffectedProduct": """
Given the following data from a CSV row:
Labels: {labels}
Properties: {props}

And the following list of V2 products (node_id, product_name, product_version, product_architecture):
{v2_specs}

Your goal is to evaluate the CSV data and compare it to the existing V2 products to determine if the row references an existing product or not. You proceed in three steps:
1. Cleanup and transform the CSV data into a dictionary based on the rules below.
2. Decide if the row references an existing product or not. If it doesn't, set the action to 'create_new' and set the new_label to 'Technology'.
3. Generate the response JSON object to contain your decisions and transformed row data.

Please analyze this data and return a JSON object with the following structure:
{{
    "action": "Either 'map_to_existing', 'create_new', or 'ignore'",
    "v1_original_id": "The original ID from the input",
    "v2_node_id": "The node_id of the existing V2 product if action is 'map_to_existing', otherwise null",
    "new_label": "Product | Technology | None",
    "new_props": {{
        "name": "The normalized name",
        "version": "The version information, insert 'NV' if not provided",
        "architecture": "The architecture information, insert 'NA' if not provided",
        "build_number": "Product build number as an array of ints, or null if not applicable",
        "description": "Description if available or ambiguous data found elsewhere",
        "node_label":"Product | Technology"
    }},
    "is_product": true or false
}}

Rules:
1. Ignore (set new_label to "Ignore") for FAQ nodes.
2. For AffectedProducts:
   - Try to map to the provided V2 products exactly, using all three attributes (product_name, product_version, product_architecture).
   - Any combination of 'Microsoft Edge' or 'chromium' references the 'edge' V2 product.
   - If multiple products are mentioned in the `name` field of the row data, select the most common or most important product or a V2 matching product. For example, if 'Windows 10 SQL server' are both referenced in the name, choose the product that matches a V2 product, i.e., 'windows 10'.
   - If it can't be mapped to a V2 product, use 'Technology' as the new_label.
   - If the `name` contains version, build, or architecture information, extract and assign it to the correct property.
   - If the `version` field contains a build number, move it to `build_number` as an array of ints. For example, 'version: 8.105.0.208' -> `build_number: [8,105,0,208]`.
   - If `version` contains architecture data, move it to `architecture`. For example, `version: 8.105.0.208 64-bit` -> `architecture: 'x64'`, `build_number: [8,105,0,208]`.
   - If there is ambiguous data in the name, move it to `description`.
   - For multiple products or versions in a single field, pick one (prioritize Windows 10/11 x64, non-ARM systems).
   - Insert 'NV' for missing versions and 'NA' for missing architectures.
   - Set "is_product" to true only for Products, false for all others.
3. Normalize product names to match V2 product specifications exactly when possible.
4. Do smart data cleaning to handle messy or incorrectly formatted data.
5. If there is no useful data in the row, return null.

""",
        "Symptom": """
Given the following data from a CSV row:
Labels: {labels}
Properties: {props}

Your goal is to transform this data into the appropriate format for V2, maintaining as much relevant information as possible. Do not add keys to `new_props`, the only valid keys are listed below.

Please analyze this data and return a JSON object with the following structure:
{{
    "action": "create_new",
    "v1_original_id": "The original ID from the input",
    "new_label": "Symptom",
    "new_props": {{
        "description": "The description or summary",
        "symptom_label": "The original V1 ID"
    }}
}}

Rules:
1. Keep the original label as "Symptom".
2. Map the V1 ID to the V2 property 'symptom_label'.
3. Include any relevant descriptions or summaries.
4. The only properties allowed are listed in the example. Do not add any new properties.
5. If there is no useful data in the row, return null.

""",
        "Cause": """
Given the following data from a CSV row:
Labels: {labels}
Properties: {props}

Your goal is to transform this data into the appropriate format for V2, maintaining as much relevant information as possible. Do not add keys to `new_props`, the only valid keys are listed below.

Please analyze this data and return a JSON object with the following structure:
{{
    "action": "create_new",
    "v1_original_id": "The original ID from the input",
    "new_label": "Cause",
    "new_props": {{
        "description": "The description",
    }}
}}

Rules:
1. Keep the original label as "Cause".
2. Include any relevant descriptions.
3. The only properties allowed are listed in the example. Do not add any new properties.
4. If there is no useful data in the row, return null.
""",
        "Fix": """
Given the following data from a CSV row:
Labels: {labels}
Properties: {props}

Your goal is to transform this data into the appropriate format for V2, maintaining as much relevant information as possible. Do not add keys to `new_props`, the only valid keys are listed below.

Please analyze this data and return a JSON object with the following structure:
{{
    "action": "create_new",
    "v1_original_id": "The original ID from the input",
    "new_label": "Fix",
    "new_props": {{
        "description": "The description",
    }}
}}

Rules:
1. Keep the original label as "Fix".
2. Include any relevant descriptions.
3. The only properties allowed are listed in the example. Do not add any new properties.
4. If there is no useful data in the row, return null.
""",
        "Tool": """
Given the following data from a CSV row:
Labels: {labels}
Properties: {props}

Your goal is to transform this data into the appropriate format for V2, maintaining as much relevant information as possible. Do not add keys to `new_props`, the only valid keys are listed below.

In particular, you should extract or generate a concise name for the tool. Sometimes the name of the tool is contained within the description field. Where possible, generate a value for the `name` property. You must include both a `description` and a `name`. If you have knowledge about a tool, you can elaborate the description to better fill in its function or purpose.

Please analyze this data and return a JSON object with the following structure:
{{
    "action": "create_new",
    "v1_original_id": "The original ID from the input",
    "new_label": "Tool",
    "new_props": {{
        "description": "The description",
        "name": "The extracted or generated name for the tool".
        "source_url": "If there is a URL in the row data, include it here."
    }}
}}

Rules:
1. Keep the original label as "Tool".
2. Include any relevant descriptions.
3. The only properties allowed are listed in the example. Do not add any new properties.
4. If there is no useful data in the row, return null.
""",
        "MSRCSecurityUpdate": """
Given the following data from a CSV row:
Labels: {labels}
Properties: {props}

Your goal is to evaluate the CSV data and create a JSON object based on the following schema.  Do not add keys to `new_props`, the only valid keys are listed below.

You proceed in three steps:
1. Cleanup and transform the CSV data into a dictionary based on the rules below.
2. Set action to 'create_new'.
3. Generate the response JSON object to contain your decisions and transformed row data.

Please analyze this data and return a JSON object with the following structure:
{{
    "action": "create_new",
    "v1_original_id": "The original ID from the input",
    "v2_node_id": None,
    "new_label": "MSRCPost",
    "new_props": {{
        "post_id": "The id from the row data",
        "title": "The title from the row data",
        "revision": "The revision from the row data",
        "description": "Description if available or ambiguous data found elsewhere",
        "node_label":"MSRCPost"
    }},
}}

Rules:
1. The id from the row data does not map to the node_id in the V2 MSRCPosts. It maps to the property 'post_id'.
2. Do smart data cleaning to handle messy or incorrectly formatted data.
3. If there is no useful data in the row, return null.
4. If row data references an existing MSRCPost, set action to 'map_to_existing' and set "new_props" equal to an empty dictionary.
5. The only properties allowed are listed in the example. Do not add any new properties.

""",
        "PatchManagement": """
Given the following data from a CSV row:
Labels: {labels}
Properties: {props}

Your goal is to evaluate the CSV data and carefully create a json object with the following schema. Set action to 'create_new'.

Do not add keys to `new_props`, the only valid keys are listed below.

Please analyze the row data and return a JSON object with the following structure:
{{
    "action": "create_new",
    "v1_original_id": "The original ID from the input",
    "v2_node_id": null,
    "new_label": "PatchManagementPost",
    "new_props": {{
        "summary": "The `summary` value from the row data",
        "receivedDateTime": "The `receivedDateTime` value from the row data",
        "post_type": "The `post_type` from the row data",
        "published": "The `published` date from the row data",
        "conversation_link": "The `conversation_link` from the row data",
        "title": "The `title` from the row data",
    }},
}}

Rules:
1. create a new node with label "PatchManagementPost".
2. Include all relevant properties listed in new_props above.
3. The only properties allowed are listed in the example. Do not add any new properties.

""",
        "FAQ": """
Given the following data from a CSV row:
Labels: {labels}
Properties: {props}

Your goal is to transform this data into the appropriate format for V2, maintaining as much relevant information as possible. Do not add keys to `new_props`, the only valid keys are listed below.

Please analyze this data and return a JSON object with the following structure:
{{
    "action": "create_new",
    "v1_original_id": "The original ID from the input",
    "new_label": "FAQ",
    "new_props": {{
        "description": "The description",
    }}
}}

Rules:
1. Keep the original label as "FAQ".
2. Include any relevant descriptions.
3. The only properties allowed are listed in the example. Do not add any new properties.
4. If there is no useful data in the row, return null.
""",
    }

    @classmethod
    def get_prompt(cls, node_type: str) -> str:
        return cls.PROMPTS.get(node_type, "")
