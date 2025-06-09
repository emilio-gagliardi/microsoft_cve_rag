STAT_CARD_LABELS = {
    # 1. Executive Summary
    "total_cves": "Total CVEs",
    "critical_high_pct": "% Critical & High CVEs",
    "median_cvss": "Median CVSS",
    "highest_cvss": "Highest CVSS",
    "top_impact_type": "Top Vulnerability Category",

    # 2. Volume & Severity Trends
    "total_cves_qoq_delta": "QoQ Δ in Total CVEs",
    "critical_count": "Critical Count",
    "moderate_count": "Medium Count",
    "important_count": "High Count",
    "low_count": "Low Count",
    "high_to_total_pct": "High-to-Total Ratio",
    "most_volatile_month": "Most Volatile Month",

    # 3. CVSS Distribution
    "p90_cvss": "P90 CVSS",
    "lowest_cvss": "Lowest CVSS",
    "highest_cvss": "Highest CVSS",
    "score_iqr": "CVSS IQR",
    "avg_cvss_score": "Average CVSS Score",
    "score_std_dev": "CVSS Score Variability",
    "cvss_iqr_by_category": "CVSS Range by Category",

    # 4. Impact Analysis
    "rce_pct": "% of RCE CVEs",
    "eop_pct": "% of EOP CVEs",
    "dos_pct": "% of DoS CVEs",
    "info_disclosure_pct": "% of Disclosure CVEs",
    "spoofing_pct": "% of Spoofing CVEs",
    "tampering_pct": "% of Tampering CVEs",
    "cve_category_distribution": "CVE Vulnerability Distributions",
    "feature_bypass_pct": "% of Feature Bypass CVEs",
    "top_3_affected_builds": "Top 3 Affected Builds",
    "most_affected_product": "Most Affected Product(s)",

    # 5. Exploitability
    "network_vector_pct": "Network Vector",
    "adjacent_vector_pct": "Adjacent Vector",
    "local_vector_pct": "Local Vector",
    "physical_vector_pct": "Physical Vector",
    "no_user_action_pct": "No User Action",
    "user_action_req_pct": "User Action Required",
    "low_privs_required_pct": "Low Privileges Required",
    "high_privs_required_pct": "High Privileges Required",
    "low_attack_comp_pct": "Low Attack Complexity",
    "high_attack_comp_pct": "High Attack Complexity",
    "worst_case_cves": "High Risk CVEs",

    # 6. CWE Weaknesses
    "top_cwe_id": "Top Common Weakness",
    "top_3_cwe_coverage_pct": "Top 3 CWEs Coverage",
    "memory_safety_vs_logic_bugs_pct": "Memory Safety vs Logic Bugs",
    "new_cwe_types_this_qtr": "Count of newly seen CWEs",
    "spotlight_cwe_CWE-16": "CWE-16 Config Weakness",
    "spotlight_cwe_CWE-20": "CWE-20 Input Validation",
    "spotlight_cwe_CWE-74": "CWE-74 Downstream Injection",
    "spotlight_cwe_CWE-78": "CWE-78 OS Command Injection",
    "spotlight_cwe_CWE-79": "CWE-79 Cross-Site Scripting",
    "spotlight_cwe_CWE-89": "CWE-89 SQL Injection",
    "spotlight_cwe_CWE-94": "CWE-94 Code Injection",
    "spotlight_cwe_CWE-119": "CWE-119 Memory Buffer",
    "spotlight_cwe_CWE-120": "CWE-120 Buffer Overflow",
    "spotlight_cwe_CWE-121": "CWE-121 Stack Overflow",
    "spotlight_cwe_CWE-122": "CWE-122 Heap Overflow",
    "spotlight_cwe_CWE-124": "CWE-124 Buffer Underwrite",
    "spotlight_cwe_CWE-125": "CWE-125 Out-of-Bounds Read",
    "spotlight_cwe_CWE-188": "CWE-188 Memory Layout",
    "spotlight_cwe_CWE-276": "CWE-276 Default Permissions",
    "spotlight_cwe_CWE-284": "CWE-284 Access Control",
    "spotlight_cwe_CWE-285": "CWE-285 Improper Auth",
    "spotlight_cwe_CWE-310": "CWE-310 Crypto Issues",
    "spotlight_cwe_CWE-326": "CWE-326 Encrypt Strength",
    "spotlight_cwe_CWE-337": "CWE-337 Predictable Seed",
    "spotlight_cwe_CWE-338": "CWE-338 Weak PRNG",
    "spotlight_cwe_CWE-362": "CWE-362 Race Condition",
    "spotlight_cwe_CWE-416": "CWE-416 Use After Free",
    "spotlight_cwe_CWE-444": "CWE-444 HTTP Smuggling",
    "spotlight_cwe_CWE-610": "CWE-610 External Resource Ref",
    "spotlight_cwe_CWE-632": "CWE-632 Access Control Wkns",
    "spotlight_cwe_CWE-639": "CWE-639 Auth Bypass",
    "spotlight_cwe_CWE-667": "CWE-667 Improper Locking",
    "spotlight_cwe_CWE-703": "CWE-703 Exception Handling",
    "spotlight_cwe_CWE-707": "CWE-707 Downstream Inject",
    "spotlight_cwe_CWE-732": "CWE-732 Critical Resource Perms",
    "spotlight_cwe_CWE-762": "CWE-762 Mismatched Memory",
    "spotlight_cwe_CWE-783": "CWE-783 Operator Precedence",
    "spotlight_cwe_CWE-787": "CWE-787 Out-of-Bounds Write",
    "spotlight_cwe_CWE-840": "CWE-840 Business Logic Err",
    "spotlight_cwe_CWE-841": "CWE-841 Workflow Enforce",
    "spotlight_cwe_CWE-642": "CWE-642 Critical State Data",

    # 7. Patch Timeliness
    "median_days_to_patch": "Typical Patch Delay",
    "patched_le_7d_pct": "Patched within 7 days",
    "patched_le_30d_pct": "Patched within 30 days",
    "zero_day_count": "Exploited Before Patched",

    # 8. Conclusion Recap
    "quarter_risk_index": "Calculated overall risk score",
    "trend_direction": "Overall risk trend indicator",
    "most_affected_build": "Build With Most CVEs",
    "next_patch_tuesday_date": "Date of Upcoming Patch Release",

    # 9. Appendix / Reference
    "data_freshness": "Timestamp of Last Data Update",
    "rows_analysed": "Total CVE Records Processed",
    "null_field_pct": "Percentage of missing key data",
    "epss_score_gt_threshold_pct": "EPSS > Threshold",
    "exploit_maturity_high_pct": "High Exploit Maturity",
    "known_exploited_pct": "Known Exploited",
}

STAT_CARD_SUBTITLES = {
    # 1. Executive Summary
    "total_cves": "Total vulnerabilities this quarter",
    "critical_high_pct": "Most problematic CVEs this quarter",
    "median_cvss": "Typical CVSS score (out of 10)",
    "highest_cvss": "Highest CVSS score (out of 10)",
    "top_impact_type": "Most frequent vulnerability category",

    # 2. Volume & Severity Trends
    "total_cves_qoq_delta": "Quarter-over-quarter volume change",
    "critical_count": "Number of 'Critical' rated CVEs",
    "high_to_total_pct": "Percentage of 'High' severity CVEs",
    "most_volatile_month": "Month with most CVEs",
    "important_count": "Number of 'Important' rated CVEs",
    "low_count": "Number of 'Low' rated CVEs",
    "moderate_count": "Number of 'Moderate' rated CVEs",

    # 3. CVSS Distribution
    "p90_cvss": "90th percentile CVSS score",
    "lowest_cvss": "Minimum recorded CVSS score",
    "highest_cvss": "Maximum recorded CVSS score",
    "score_iqr": "Middle 50% CVSS score range",
    "avg_cvss_score": "Average CVSS score",
    "score_std_dev": "CVSS score standard deviation",
    "cvss_iqr_by_category": "CVSS Interquartile range by CVE Category",

    # 4. Impact Analysis
    "rce_pct": "% of Remote Code Execution CVEs",
    "eop_pct": "% of Elevation of Privilege CVEs",
    "dos_pct": "% of Denial of Service CVEs",
    "info_disclosure_pct": "% of Information Disclosure CVEs",
    "feature_bypass_pct": "% of Security Feature Bypass CVEs",
    "spoofing_pct": "% of Spoofing CVEs",
    "tampering_pct": "% of Tampering CVEs",
    "cve_category_distribution": "Various metrics of CVEs by vulnerability category",
    "most_affected_product": "Windows Products with Highest CVE Count",

    # 5. Exploitability
    "network_vector_pct": "% of Network Vector CVEs",
    "adjacent_vector_pct": "% of Adjacent Network Vector CVEs",
    "local_vector_pct": "% of Local Vector CVEs",
    "physical_vector_pct": "% of Physical Vector CVEs",
    "user_action_req_pct": "% of User Action Required CVEs",
    "no_user_action_pct": "% of No User Action CVEs",
    "no_privs_required_pct": "% of No Privileges Required CVEs",
    "low_privs_required_pct": "% of Low Privileges Required CVEs",
    "high_privs_required_pct": "% of High Privileges Required CVEs",
    "low_attack_comp_pct": "% of Low Attack Complexity CVEs",
    "high_attack_comp_pct": "% of High Attack Complexity CVEs",
    "worst_case_cves": "Weighted score (network + low complexity + low privileges + no interaction)",

    # 6. CWE Weaknesses
    "top_cwe_id": "CWE with highest % of total CVEs",
    "top_3_cwe_coverage_pct": "Percent of total CVEs covered by top 3 CWEs",
    "memory_safety_vs_logic_bugs_pct": "Ratio of memory safety vs logic bugs",
    "new_cwe_types_this_qtr": "Count of newly seen CWEs",
    "spotlight_cwe_CWE-16": "System Configuration: Weakness, leads to vulnerabilities from insecure ...",
    "spotlight_cwe_CWE-20": "Input: Improperly validated, leads to security vulnerabilities ...",
    "spotlight_cwe_CWE-74": "Downstream Output Special Elements: Not neutralized, leads to injection ...",
    "spotlight_cwe_CWE-78": "OS Command Special Elements: Not neutralized, leads to OS command ...",
    "spotlight_cwe_CWE-79": "Web Page Generation Input: Not neutralized, leads to cross-site ...",
    "spotlight_cwe_CWE-89": "SQL Command Special Elements: Not neutralized, leads to SQL injection ...",
    "spotlight_cwe_CWE-94": "Code Generation: Improperly controlled, leads to code injection ...",
    "spotlight_cwe_CWE-119": "Memory Buffer Operations: Improperly restricted within defined bounds ...",
    "spotlight_cwe_CWE-120": "Buffer Copy Input Size: Not checked ('Classic Buffer Overflow'), leads ...",
    "spotlight_cwe_CWE-121": "Stack-Based Buffer: Overflow, leads to crashes or arbitrary code ...",
    "spotlight_cwe_CWE-122": "Heap-Based Buffer: Overflow, leads to crashes or arbitrary code ...",
    "spotlight_cwe_CWE-124": "Buffer Underwrite: Data written before buffer start ('Buffer Underflow') ...",
    "spotlight_cwe_CWE-125": "Out-of-Bounds Read: Occurs outside allocated memory, leads to info ...",
    "spotlight_cwe_CWE-188": "Data/Memory Layout of Type: Reliance upon, leads to portability or ...",
    "spotlight_cwe_CWE-276": "Default Permissions: Incorrectly set, leads to unauthorized access ...",
    "spotlight_cwe_CWE-284": "Access Control: Improperly implemented, leads to unauthorized actions ...",
    "spotlight_cwe_CWE-285": "Authorization: Improperly performed, leads to privilege escalation ...",
    "spotlight_cwe_CWE-310": "Cryptographic Functions/Issues: Flawed use or implementation, leads to ...",
    "spotlight_cwe_CWE-326": "Encryption Strength: Inadequate (weak algorithm/key), leads to data ...",
    "spotlight_cwe_CWE-337": "PRNG Seed: Predictable, leads to compromised random values for ...",
    "spotlight_cwe_CWE-338": "Weak PRNG: Used in security context, leads to predictable values ...",
    "spotlight_cwe_CWE-362": "Shared Resource (Concurrent): Improperly synchronized ('Race Condition') ...",
    "spotlight_cwe_CWE-416": "Use After Free: Referencing memory after deallocation, leads to ...",
    "spotlight_cwe_CWE-444": "HTTP Request Interpretation: Inconsistent ('HTTP Request Smuggling') ...",
    "spotlight_cwe_CWE-610": "Cross-Sphere Resource Reference: Externally controlled, leads to ...",
    "spotlight_cwe_CWE-632": "User Interface Security: Weaknesses, leads to client-side attacks ...",
    "spotlight_cwe_CWE-639": "Authorization Bypass (User Key): Using user-controlled key, leads to ...",
    "spotlight_cwe_CWE-667": "Resource Locking: Improperly performed, leads to race conditions or ...",
    "spotlight_cwe_CWE-703": "Exceptional Conditions: Improperly checked or handled, leads to ...",
    "spotlight_cwe_CWE-707": "Downstream Output Special Elements: Not neutralized, leads to injection ...",
    "spotlight_cwe_CWE-732": "Critical Resource Permissions: Incorrectly assigned, leads to ...",
    "spotlight_cwe_CWE-762": "Memory Management Routines: Mismatched for alloc/dealloc, leads to ...",
    "spotlight_cwe_CWE-783": "Operator Precedence Logic: Error in evaluation order, leads to ...",
    "spotlight_cwe_CWE-787": "Out-of-Bounds Write: Data written outside allocated memory, leads to ...",
    "spotlight_cwe_CWE-840": "Business Logic: Errors in design or implementation, leads to ...",
    "spotlight_cwe_CWE-841": "Behavioral Workflow: Improperly enforced by application, leads to ...",
    "spotlight_cwe_CWE-642": "Critical State Data: Externally controlled, leads to manipulation ...",

    # 7. Patch Timeliness
    "median_days_to_patch": "Median number of days to patch",
    "patched_le_7d_pct": "% of CVEs patched within 7 days",
    "patched_le_30d_pct": "% of CVEs patched within 30 days",
    "zero_day_count": "",

    # 8. Conclusion Recap
    "quarter_risk_index": "Calculated overall risk score",
    "trend_direction": "Overall risk trend indicator",
    "most_affected_build": "Build with highest CVE count",
    "next_patch_tuesday_date": "Date of upcoming patch release",

    # 9. Appendix / Reference
    "data_freshness": "Timestamp of last data update",
    "rows_analysed": "Total CVE records processed",
    "null_field_pct": "Percentage of missing key data",
}

# Define base text sizes, can be overridden in specific styles
BASE_TEXT_SIZES = {
    "label": "text-sm font-semibold leading-[1.1]",
    "value": "text-3xl md:text-4xl font-bold leading-tight",
    "unit": "text-base font-medium ml-1 opacity-80 pl-0.5",
    "subtitle": "text-xxs leading-[1.1] text-center",
    "trend": "text-xs font-medium",
    "cwe_extra_info": "text-xxs",
}

# style mappings for the stats cards (maps metric_id to card properties)
# Uses semantic color names defined in tailwind.config.js mapped to company-base.css variables
# Inherit from default_standard and override as needed
CARD_STYLES = {
    "default_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-report-text-muted dark:text-report-text-muted-dark", # Text color for the label
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "total_cves_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "critical_count_card": {
        "container": "bg-status-danger-bg dark:bg-status-danger-bg-dark border border-status-danger dark:border-status-danger-border", # Neutral card, danger border
        # Title Bar themed for danger
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-status-danger-text dark:text-status-danger-text-dark", # Text color for danger bar
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-status-danger-text dark:text-status-danger-text-dark", # Icon color for danger bar
        # Content Area
        "content_area_bg": "",
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-status-danger-text dark:text-status-danger-text-dark", # Value text is danger-themed
        "unit": f"{BASE_TEXT_SIZES['unit']} text-status-danger/80 dark:text-status-danger-text-dark/80", # Unit also danger-themed
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-status-danger-text dark:text-status-danger-text-dark", # Subtitle can be neutral
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-success dark:text-status-success-text-dark",
        "trend_down": "text-red-500 dark:text-red-400", # Example, if 'down' is bad for critical count
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "critical_high_pct_card": {
        "container": "bg-status-warning-bg dark:bg-status-warning-bg-dark border border-status-warning dark:border-status-warning-border", # Neutral card, danger border
        # Title Bar themed for danger
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-status-warning-text dark:text-status-warning-text-dark", # Text color for danger bar
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-status-warning-text dark:text-status-warning-text-dark", # Icon color for danger bar
        # Content Area
        "content_area_bg": "",
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-status-warning-text dark:text-status-warning-text-dark", # Value text is danger-themed
        "unit": f"{BASE_TEXT_SIZES['unit']} text-status-warning-text dark:text-status-warning-text-dark", # Unit also danger-themed
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-status-warning-text dark:text-status-warning-text-dark", # Subtitle can be neutral
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-success dark:text-status-success-text-dark",
        "trend_down": "text-red-500 dark:text-red-400", # Example, if 'down' is bad for critical count
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "high_to_total_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "most_volatile_month_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "avg_cvss_score_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "median_cvss_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "p90_cvss_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "score_iqr_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "score_std_dev_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "rce_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "eop_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "dos_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "disclosure_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "spoofing_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "tampering_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "network_vector_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "adjacent_vector_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "no_user_action_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "no_privs_required_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "low_privs_required_pct_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-report-text dark:text-report-text-dark", # Text color for the label
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "high_privs_required_pct_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-report-text dark:text-report-text-dark", # Text color for the label
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "low_attack_comp_pct_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-report-text dark:text-report-text-dark", # Text color for the label
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "high_attack_comp_pct_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-report-text dark:text-report-text-dark", # Text color for the label
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "worst_case_cves_card": {
        "container": "bg-status-danger-bg dark:bg-status-danger-bg-dark border border-status-danger dark:border-status-danger-border", # Neutral card, danger border
        # Title Bar themed for danger
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-status-danger-text dark:text-status-danger-text-dark", # Text color for danger bar
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-status-danger-text dark:text-status-danger-text-dark", # Icon color for danger bar
        # Content Area
        "content_area_bg": "",
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-status-danger-text dark:text-status-danger-text-dark", # Value text is danger-themed
        "unit": f"{BASE_TEXT_SIZES['unit']} text-status-danger/80 dark:text-status-danger-text-dark/80", # Unit also danger-themed
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-status-danger-text dark:text-status-danger-text-dark", # Subtitle can be neutral
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-success dark:text-status-success-text-dark",
        "trend_down": "text-red-500 dark:text-red-400", # Example, if 'down' is bad for critical count
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "epss_score_gt_threshold_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "exploit_maturity_high_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "known_exploited_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "local_vector_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "physical_vector_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "user_action_req_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "lowest_cvss_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "cvss_iqr_by_category_style": {
        "container": "bg-report-surface-light dark:bg-report-surface-dark border border-report-border-light dark:border-report-border-dark text-center",
        "label": f"{BASE_TEXT_SIZES['label']} text-report-text-muted-light dark:text-report-text-muted-dark",
        "icon": "text-brand-blue dark:text-brand-blue-light opacity-75",
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text-light dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['value']} text-report-text-light dark:text-report-text-dark opacity-80 pl-0.5",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted-light dark:text-report-text-muted-dark  text-xxs leading-[1.1]",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "",
        "trend_down": "",
        "trend_neutral": ""
    },
    "feature_bypass_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "patched_le_7d_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "patched_le_30d_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "most_affected_build_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "next_patch_tuesday_date_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "rows_analysed_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "null_field_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "adj_vector_pct_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "total_cves_qoq_delta_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "moderate_count_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "important_count_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "low_count_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "highest_cvss_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "category_comparison_bar_style": {
        "container": "bg-report-surface-light dark:bg-report-surface-dark border border-report-border-light dark:border-report-border-dark text-center",
        "label": f"{BASE_TEXT_SIZES['label']} text-report-text-muted-light dark:text-report-text-muted-dark",
        "icon": "text-brand-blue dark:text-brand-blue-light opacity-75",
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text-light dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['value']} text-report-text-light dark:text-report-text-dark opacity-80 pl-0.5",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted-light dark:text-report-text-muted-dark  text-xxs leading-[1.1]",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "",
        "trend_down": "",
        "trend_neutral": ""
    },
    "top_impact_type_card": {
        "container": "bg-white dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "",
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}",
        "label_text_color": "text-report-text dark:text-report-text-dark",
        "icon_style": "w-4 h-4",
        "icon_color": "text-report-text-muted dark:text-report-text-muted-dark opacity-90",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": "text-1xl md:text-1xl font-bold text-report-text dark:text-report-text-dark leading-[1.1]",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "top_3_affected_builds_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "bg-gray-100 dark:bg-gray-700", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark", # Or your desired up color
        "trend_down": "text-status-success dark:text-status-success-text-dark", # Or your desired down color
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "most_affected_product_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"text-1xl text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark", # Or your desired up color
        "trend_down": "text-status-success dark:text-status-success-text-dark", # Or your desired down color
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "top_cwe_id_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "",
        # Value, Unit, Subtitle, Trend styles
        "value": f"text-2xl md:text-2xl font-bold text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark",
        "trend_down": "text-status-success dark:text-status-success-text-dark",
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "top_3_cwe_coverage_pct_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "",
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark", # Or your desired up color
        "trend_down": "text-status-success dark:text-status-success-text-dark", # Or your desired down color
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "memory_safety_vs_logic_bugs_pct_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark", # Or your desired up color
        "trend_down": "text-status-success dark:text-status-success-text-dark", # Or your desired down color
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "new_cwe_types_this_qtr_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "bg-gray-100 dark:bg-gray-700", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark", # Or your desired up color
        "trend_down": "text-status-success dark:text-status-success-text-dark", # Or your desired down color
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "median_days_to_patch_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark", # Or your desired up color
        "trend_down": "text-status-success dark:text-status-success-text-dark", # Or your desired down color
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "zero_day_count_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "bg-gray-100 dark:bg-gray-700", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark", # Or your desired up color
        "trend_down": "text-status-success dark:text-status-success-text-dark", # Or your desired down color
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "quarter_risk_index_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "bg-gray-100 dark:bg-gray-700", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark", # Or your desired up color
        "trend_down": "text-status-success dark:text-status-success-text-dark", # Or your desired down color
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "data_freshness_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark", # Overall card container
        # Title Bar specific styles
        "title_bar_bg": "bg-gray-100 dark:bg-gray-700", # Default subtle bar
        "title_bar_padding": "py-1 px-1",
        "label_text_style": f"{BASE_TEXT_SIZES['label']}", # Text styling for the label
        "label_text_color": "text-gray-700 dark:text-gray-300", # Text color for the label
        "icon_style": "w-4 h-4 opacity-75",
        "icon_color": "text-gray-500 dark:text-gray-400",
        # Content Area specific styles (if different from overall container)
        "content_area_bg": "bg-report-surface dark:bg-report-surface-dark", # Usually same as container for default
        # Value, Unit, Subtitle, Trend styles
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark",
        "unit": f"{BASE_TEXT_SIZES['unit']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle": f"{BASE_TEXT_SIZES['subtitle']} text-report-text-muted dark:text-report-text-muted-dark",
        "subtitle_bar_padding": "px-2 pb-2 pt-1",
        "trend_base": f"{BASE_TEXT_SIZES['trend']}",
        "trend_up": "text-status-danger dark:text-status-danger-text-dark", # Or your desired up color
        "trend_down": "text-status-success dark:text-status-success-text-dark", # Or your desired down color
        "trend_neutral": "text-report-text-muted dark:text-report-text-muted-dark"
    },
    "spotlight_cwe_CWE-16_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-20_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-74_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-78_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-79_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-89_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-94_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-119_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-120_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-121_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-122_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-124_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-125_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-188_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-276_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-284_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-285_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-310_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-326_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-337_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-338_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-362_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-416_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-444_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-610_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-632_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-639_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-667_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-703_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-707_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-732_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-762_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-783_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-787_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-840_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-841_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_CWE-642_card": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
    "spotlight_cwe_default_style": {
        "container": "bg-report-surface dark:bg-report-surface-dark border border-report-border dark:border-report-border-dark rounded-md p-2 flex flex-col justify-between min-h-[150px]",
        "label": f"text-xs text-report-text dark:text-report-text-dark font-semibold text-left break-words leading-[1.1] overflow-hidden mb-1", # Added mb-1 for a bit of space below label block
        "icon_container": "ml-2 flex-shrink-0", # Classes for the div wrapping the icon
        "icon": f"w-5 h-5 text-brand-blue dark:text-brand-blue-light opacity-75", # Classes for the icon span itself
        "value_container": "my-auto text-center", # For the div wrapping value and its subtexts
        "value": f"{BASE_TEXT_SIZES['value']} text-report-text dark:text-report-text-dark block leading-[1.1]", # Added block & leading-tight
        "predominant_category_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} font-semibold mt-1 leading-[1.1]",
        "avg_cvss_text": f"{BASE_TEXT_SIZES.get('cwe_extra_info', 'text-xs')} text-report-text-muted dark:text-report-text-muted-dark mt-0.5 leading-[1.1]",
        "subtitle": f"{BASE_TEXT_SIZES.get('subtitle', 'text-xxs')} text-report-text-muted dark:text-report-text-muted-dark leading-[1.1]"
    },
}


# Color mapping for CVE Category Distribution Bar Chart
# Maps category 'id_for_color' to specific chart Tailwind classes
CVE_CATEGORY_COLOR_MAP = {
    # Map critical/high-impact categories to warmer chart colors
    'remote_code_execution': 'bg-chart-red-1 dark:bg-chart-red-1-dark',
    'privilege_elevation': 'bg-chart-orange-1 dark:bg-chart-orange-1-dark',
    'feature_bypass': 'bg-chart-orange-2 dark:bg-chart-orange-2-dark',
    # Map informational/less critical categories to cooler chart colors
    'disclosure': 'bg-chart-blue-1 dark:bg-chart-blue-1-dark',
    'denial_of_service': 'bg-chart-teal-1 dark:bg-chart-teal-1-dark',
    'spoofing': 'bg-chart-green-1 dark:bg-chart-green-1-dark',
    'tampering': 'bg-chart-green-1 dark:bg-chart-green-1-dark',
    # Specific product/other categories (add more if needed)
    'edge_specific': 'bg-chart-blue-2 dark:bg-chart-blue-2-dark',
    # Fallback/Default/None categories
    'none': 'bg-chart-gray-1 dark:bg-chart-gray-1-dark',
    'unknown': 'bg-chart-gray-1 dark:bg-chart-gray-1-dark',
    'other': 'bg-chart-gray-2 dark:bg-chart-gray-2-dark',
    'default': 'bg-chart-gray-2 dark:bg-chart-gray-2-dark'  # Fallback for unmapped keys
}
