CWE_ROOT_CAUSE = {
    # 1. Memory‑safety (use‑after‑free, OOB, double‑free, etc.)
    "memory_safety": {
        "ids": [
            "CWE-416", "CWE-787", "CWE-119", "CWE-120", "CWE-121", "CWE-122",
            "CWE-124", "CWE-125", "CWE-188", "CWE-762"
        ],
        "description": "These vulnerabilities relate to how software manages computer memory. Attackers exploit these flaws to read sensitive data, crash systems, or execute malicious code by manipulating memory locations they shouldn't access. Proper memory handling and boundary checks are crucial defenses."
    },

    # 2. Concurrency / Synchronisation
    "race_concurrency": {
        "ids": ["CWE-362", "CWE-667", "CWE-783"],
        "description": "Concurrency issues arise when multiple processes access shared resources without proper coordination. This can lead to unpredictable behavior, data corruption, or deadlocks, known as race conditions. Ensuring proper locking and synchronization mechanisms prevents these system stability problems."
    },

    # 3. Access‑control & Priv‑escalation logic
    "improper_access_control": {
        "ids": ["CWE-284", "CWE-285", "CWE-639", "CWE-732"],
        "description": "Access control flaws occur when software fails to correctly enforce permissions, allowing unauthorized users to view data or perform actions. Privilege escalation means an attacker gains higher-level access than intended. Robust permission checks and principle of least privilege are key mitigations."
    },

    # 4. Input Validation & Injection
    "input_validation_injection": {
        "ids": [
            "CWE-20", "CWE-74", "CWE-78", "CWE-79", "CWE-89", "CWE-94", "CWE-707"
        ],
        "description": "These weaknesses happen when software doesn't properly sanitize or validate user-supplied input. Attackers can inject malicious commands (like SQL injection) or scripts (like cross-site scripting - XSS) to steal data or compromise the system. Strict input validation is essential to block such attacks."
    },

    # 5. Cryptography & Confidentiality flaws
    "cryptographic_issues": {
        "ids": ["CWE-310", "CWE-326", "CWE-337", "CWE-338"],
        "description": "Cryptographic flaws involve weak encryption algorithms, improper key management, or predictable random number generation. These issues can expose sensitive data like passwords or session tokens. Using strong, standard cryptographic practices is vital for protecting data confidentiality."
    },

    # 6. Configuration / Hardening / Defaults
    "configuration_weakness": {
        "ids": ["CWE-16", "CWE-276", "CWE-444", "CWE-610"],
        "description": "Vulnerabilities can arise from insecure default settings, incorrect configurations, or inadequate system hardening. This includes overly permissive file permissions or unnecessary services being enabled. Regularly reviewing configurations and applying security baselines reduces this risk."
    },

    # 7. Business / Application Logic
    "logic_state_errors": {
        "ids": ["CWE-840", "CWE-841", "CWE-642", "CWE-703"],
        "description": "Logic flaws occur when the application's intended workflow or state management can be manipulated in unexpected ways. This might allow bypassing security checks, performing unauthorized actions, or causing denial of service. Thorough testing of application logic helps identify these issues."
    },
}

CWE_DETAILS = {
    "CWE-16": {
        "name": "Configuration Weakness",
        "description": "Vulnerabilities can arise from insecure default settings, incorrect configurations, or inadequate system hardening. This includes overly permissive file permissions or unnecessary services being enabled. Regularly reviewing configurations and applying security baselines reduces this risk.",
        "category": "Configuration / Hardening / Defaults",
        "cwe_url": "https://cwe.mitre.org/data/definitions/16.html"
    },
    "CWE-20": {
        "name": "Improper Input Validation",
        "description": "The product receives input or data, but it does not validate or incorrectly validates that the input has the properties that are required to process the data safely and correctly.",
        "category": "Input Validation & Injection",
        "cwe_url": "https://cwe.mitre.org/data/definitions/20.html"
    },
    "CWE-74": {
        "name": "Improper Neutralization of Special Elements in Output Used by a Downstream Component ('Injection')",
        "description": "The product constructs all or part of a command, data structure, or record using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify how it is parsed or interpreted when it is sent to a downstream component.",
        "category": "Input Validation & Injection",
        "cwe_url": "https://cwe.mitre.org/data/definitions/74.html"
    },
    "CWE-78": {
        "name": "Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')",
        "description": "The product constructs all or part of an OS command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended OS command when it is sent to a downstream component.",
        "category": "Input Validation & Injection",
        "cwe_url": "https://cwe.mitre.org/data/definitions/78.html"
    },
    "CWE-79": {
        "name": "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')",
        "description": "The product does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output that is used as a web page that is served to other users.",
        "category": "Input Validation & Injection",
        "cwe_url": "https://cwe.mitre.org/data/definitions/79.html"
    },
    "CWE-89": {
        "name": "Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')",
        "description": "The product constructs all or part of an SQL command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command when it is sent to a downstream component.",
        "category": "Input Validation & Injection",
        "cwe_url": "https://cwe.mitre.org/data/definitions/89.html"
    },
    "CWE-94": {
        "name": "Improper Control of Generation of Code ('Code Injection')",
        "description": "The product generates code using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended code when it is sent to a downstream component.",
        "category": "Input Validation & Injection",
        "cwe_url": "https://cwe.mitre.org/data/definitions/94.html"
    },
    "CWE-119": {
        "name": "Improper Restriction of Operations within the Bounds of a Memory Buffer",
        "description": "The product does not properly restrict or validate operations within the bounds of a memory buffer, allowing an attacker to access or modify memory outside of the intended bounds.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/119.html"
    },
    "CWE-120": {
        "name": "Buffer Copy without Checking Size of Input ('Classic Buffer Overflow')",
        "description": "The product copies data into a buffer without checking if the input size exceeds the buffer's capacity, leading to an overflow.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/120.html"
    },
    "CWE-121": {
        "name": "Stack-based Buffer Overflow",
        "description": "The product copies data into a buffer without checking if the input size exceeds the buffer's capacity, leading to an overflow.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/121.html"
    },
    "CWE-122": {
        "name": "Heap-based Buffer Overflow",
        "description": "The product copies data into a buffer without checking if the input size exceeds the buffer's capacity, leading to an overflow.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/122.html"
    },
    "CWE-124": {
        "name": "Buffer Underwrite ('Buffer Underflow')",
        "description": "The product copies data into a buffer without checking if the input size exceeds the buffer's capacity, leading to an overflow.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/124.html"
    },
    "CWE-125": {
        "name": "Out-of-bounds Read",
        "description": "The product copies data into a buffer without checking if the input size exceeds the buffer's capacity, leading to an overflow.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/125.html"
    },
    "CWE-188": {
        "name": "Reliance on Data/Memory Layout of a Type",
        "description": "The product relies on the memory layout of a type to perform operations, but this layout is not guaranteed to be consistent across different platforms or versions.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/188.html"
    },
    "CWE-276": {
        "name": "Incorrect Default Permissions",
        "description": "Vulnerabilities can arise from insecure default settings, incorrect configurations, or inadequate system hardening. This includes overly permissive file permissions or unnecessary services being enabled. Regularly reviewing configurations and applying security baselines reduces this risk.",
        "category": "Configuration / Hardening / Defaults",
        "cwe_url": "https://cwe.mitre.org/data/definitions/276.html"
    },
    "CWE-284": {
        "name": "Improper Access Control",
        "description": "The product does not enforce proper access control, allowing unauthorized access to resources or operations.",
        "category": "Access-Control & Priv-Escalation Logic",
        "cwe_url": "https://cwe.mitre.org/data/definitions/284.html"
    },
    "CWE-285": {
        "name": "Improper Authorization",
        "description": "The product does not enforce proper authorization checks, allowing unauthorized access to resources or operations.",
        "category": "Access-Control & Priv-Escalation Logic",
        "cwe_url": "https://cwe.mitre.org/data/definitions/285.html"
    },
    "CWE-310": {
        "name": "Cryptographic Issues",
        "description": "The product uses weak or inadequate encryption algorithms, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Cryptography & Confidentiality Flaws",
        "cwe_url": "https://cwe.mitre.org/data/definitions/310.html"
    },
    "CWE-326": {
        "name": "Inadequate Encryption Strength",
        "description": "The product uses weak or inadequate encryption algorithms, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Cryptography & Confidentiality Flaws",
        "cwe_url": "https://cwe.mitre.org/data/definitions/326.html"
    },
    "CWE-337": {
        "name": "Predictable Seed in Pseudo-Random Number Generator (PRNG)",
        "description": "The product uses a predictable seed value for a pseudo-random number generator (PRNG), leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Cryptography & Confidentiality Flaws",
        "cwe_url": "https://cwe.mitre.org/data/definitions/337.html"
    },
    "CWE-338": {
        "name": "Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG)",
        "description": "The product uses a cryptographically weak pseudo-random number generator (PRNG), leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Cryptography & Confidentiality Flaws",
        "cwe_url": "https://cwe.mitre.org/data/definitions/338.html"
    },
    "CWE-362": {
        "name": "Concurrent Execution using Shared Resource with Improper Synchronization ('Race Condition')",
        "description": "The product uses concurrent execution using shared resources without proper synchronization, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Concurrency / Synchronisation",
        "cwe_url": "https://cwe.mitre.org/data/definitions/362.html"
    },
    "CWE-416": {
        "name": "Use After Free",
        "description": "The product uses a use-after-free vulnerability, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/416.html"
    },
    "CWE-444": {
        "name": "Inconsistent Interpretation of HTTP Requests ('HTTP Request Smuggling')",
        "description": "Vulnerabilities can arise from insecure default settings, incorrect configurations, or inadequate system hardening. This includes overly permissive file permissions or unnecessary services being enabled. Regularly reviewing configurations and applying security baselines reduces this risk.",
        "category": "Configuration / Hardening / Defaults",
        "cwe_url": "https://cwe.mitre.org/data/definitions/444.html"
    },
    "CWE-610": {
        "name": "Externally Controlled Reference to a Resource in Another Sphere",
        "description": "Vulnerabilities can arise from insecure default settings, incorrect configurations, or inadequate system hardening. This includes overly permissive file permissions or unnecessary services being enabled. Regularly reviewing configurations and applying security baselines reduces this risk.",
        "category": "Configuration / Hardening / Defaults",
        "cwe_url": "https://cwe.mitre.org/data/definitions/610.html"
    },
    "CWE-632": {
        "name": "Weakness in Access Control",
        "description": "The product does not enforce proper access control, allowing unauthorized access to resources or operations.",
        "category": "Access-Control & Priv-Escalation Logic",
        "cwe_url": "https://cwe.mitre.org/data/definitions/632.html"
    },
    "CWE-639": {
        "name": "Authorization Bypass Through User-Controlled Key",
        "description": "The product does not enforce proper authorization checks, allowing unauthorized access to resources or operations.",
        "category": "Access-Control & Priv-Escalation Logic",
        "cwe_url": "https://cwe.mitre.org/data/definitions/639.html"
    },
    "CWE-667": {
        "name": "Improper Locking",
        "description": "The product does not enforce proper access control, allowing unauthorized access to resources or operations.",
        "category": "Concurrency / Synchronisation",
        "cwe_url": "https://cwe.mitre.org/data/definitions/667.html"
    },
    "CWE-703": {
        "name": "Improper Check or Handling of Exceptional Conditions",
        "description": "The product does not enforce proper access control, allowing unauthorized access to resources or operations.",
        "category": "Business / Application Logic",
        "cwe_url": "https://cwe.mitre.org/data/definitions/703.html"
    },
    "CWE-707": {
        "name": "Improper Neutralization of Special Elements in Output Used by a Downstream Component ('Injection')",
        "description": "The product does not enforce proper access control, allowing unauthorized access to resources or operations.",
        "category": "Input Validation & Injection",
        "cwe_url": "https://cwe.mitre.org/data/definitions/707.html"
    },
    "CWE-732": {
        "name": "Incorrect Permission Assignment for Critical Resource",
        "description": "The product does not enforce proper access control, allowing unauthorized access to resources or operations.",
        "category": "Access-Control & Priv-Escalation Logic",
        "cwe_url": "https://cwe.mitre.org/data/definitions/732.html"
    },
    "CWE-762": {
        "name": "Mismatched Memory Management Routines",
        "description": "The product uses mismatched memory management routines, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/762.html"
    },
    "CWE-783": {
        "name": "Operator Precedence Logic Error",
        "description": "The product uses operator precedence logic error, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Concurrency / Synchronisation",
        "cwe_url": "https://cwe.mitre.org/data/definitions/783.html"
    },
    "CWE-787": {
        "name": "Out-of-bounds Write",
        "description": "The product uses out-of-bounds write, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Memory-Safety",
        "cwe_url": "https://cwe.mitre.org/data/definitions/787.html"
    },
    "CWE-840": {
        "name": "Business Logic Errors",
        "description": "The product uses business logic errors, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Business / Application Logic",
        "cwe_url": "https://cwe.mitre.org/data/definitions/840.html"
    },
    "CWE-841": {
        "name": "Improper Enforcement of Behavioral Workflow",
        "description": "The product uses improper enforcement of behavioral workflow, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Business / Application Logic",
        "cwe_url": "https://cwe.mitre.org/data/definitions/841.html"
    },
    "CWE-642": {
        "name": "External Control of Critical State Data",
        "description": "The product uses external control of critical state data, leading to vulnerabilities in data confidentiality or integrity.",
        "category": "Business / Application Logic",
        "cwe_url": "https://cwe.mitre.org/data/definitions/642.html"
    }
}
