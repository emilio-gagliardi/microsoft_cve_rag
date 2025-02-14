"""Example script demonstrating KB report generation."""
from datetime import datetime
from pathlib import Path

from ..services.report_generator import (
    ReportGenerator,
    ReportContext,
    KBArticle,
    CVEInfo
)

def main():
    """Generate a sample KB report."""
    # Sample data
    cves = [
        CVEInfo(
            id="CVE-2025-12345",
            description="Remote code execution vulnerability in USB handling allowing arbitrary code execution with system privileges",
            severity="HIGH"
        ),
        CVEInfo(
            id="CVE-2025-12346",
            description="Elevation of privileges in system service that could allow local users to gain administrative access",
            severity="HIGH"
        ),
        CVEInfo(
            id="CVE-2025-12347",
            description="Information disclosure in network stack that could expose sensitive system information",
            severity="MEDIUM"
        ),
        CVEInfo(
            id="CVE-2025-12348",
            description="Cross-site scripting vulnerability in web interface",
            severity="MEDIUM"
        ),
        CVEInfo(
            id="CVE-2025-12349",
            description="Local denial of service in print spooler",
            severity="LOW"
        ),
        CVEInfo(
            id="CVE-2025-12350",
            description="Race condition in file handling that may cause temporary application instability",
            severity="LOW"
        )
    ]
    
    kb_articles = [
        KBArticle(
            id="KB5036893",
            title="Windows 10 (21H2, 22H2)",
            os_version="Windows 10",
            published_date="2024-04-09",
            url="#",
            os_builds=[
                "22621.3447",
                "22631.3447",
                "22641.3447"
            ],
            cves=cves[:2],  # First two HIGH severity CVEs
            new_features=[
                "USB BOOT Sock - Enhanced security features for USB boot process",
                "USB DOUBLE BOOT - Improved dual boot handling"
            ],
            bug_fixes=[
                "Fixed USB boot security vulnerability",
                "Resolved dual boot configuration issues"
            ],
            known_issues=[
                "Some USB devices may require re-authentication after update"
            ],
            summary="This update includes important security fixes for USB-related vulnerabilities and improvements to the boot process."
        ),
        KBArticle(
            id="KB5036894",
            title="Windows 11 (23H2)",
            os_version="Windows 11",
            published_date="2024-04-09",
            url="#",
            os_builds=[
                "23621.3447",
                "23631.3447"
            ],
            cves=cves[2:4],  # Two MEDIUM severity CVEs
            new_features=[
                "Enhanced Network Security - Improved packet filtering",
                "Web Interface Updates - Modern design improvements"
            ],
            bug_fixes=[
                "Fixed network stack information disclosure",
                "Addressed XSS vulnerability in web interface"
            ],
            known_issues=[
                "Some network settings may reset after update"
            ],
            summary="This update enhances network security and modernizes the web interface."
        ),
        KBArticle(
            id="KB5036895",
            title="Multi-OS Security Update",
            os_version="Multi-OS",
            published_date="2024-04-09",
            url="#",
            os_builds=[
                "Windows 10: 22621.3447",
                "Windows 11: 23621.3447",
                "Server 2022: 20348.2159"
            ],
            cves=cves[4:],  # Two LOW severity CVEs
            new_features=[],
            bug_fixes=[
                "Fixed print spooler denial of service vulnerability",
                "Resolved file handling race condition"
            ],
            known_issues=[
                "Print jobs may be delayed on some systems",
                "Temporary file operations may be slower"
            ],
            summary="This multi-OS update addresses common vulnerabilities affecting print services and file handling."
        )
    ]
    
    context = ReportContext(
        title="PortalFuse Weekly KB Report",
        report_date="January 29 - February 4, 2025",
        total_kb_articles=12,
        multi_os_updates=3,
        known_issues=5,
        windows_10_count=2,
        windows_11_count=3,
        windows_server_count=1,
        kb_articles=kb_articles
    )
    
    # Generate report
    template_dir = Path(__file__).parent.parent / "data" / "templates" / "weekly_kb_report"
    output_dir = Path(__file__).parent.parent / "data" / "reports"
    output_file = output_dir / f"kb_report_{datetime.now().strftime('%Y%m%d')}.html"
    
    generator = ReportGenerator(str(template_dir))
    generator.generate_report(context, str(output_file))
    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    main()
