"""
Test dataset for company deduplication (Approach 1: Classical String Matching).

This dataset contains synthetic company records with known duplicates for testing
the AllPairsBlocker, RapidfuzzModule, and end-to-end pipeline.

Duplicate groups:
- Group 1 (exact duplicates): companies[0] and companies[1] (IDs: "c1", "c1_dup1")
- Group 2 (typo duplicates): companies[2] and companies[3] (IDs: "c2", "c2_typo")
- Group 3 (abbreviation): companies[4] and companies[5] (IDs: "c3", "c3_abbrev")
- Group 4 (missing fields): companies[6] and companies[7] (IDs: "c4", "c4_partial")
- Group 5 (address variation): companies[8] and companies[9] (IDs: "c5", "c5_addr_var")
- Non-duplicates: companies[10-14] (IDs: "c6", "c7", "c8", "c9", "c10")
"""

COMPANY_RECORDS = [
    # Group 1: Exact duplicates (should match with high score)
    {
        "id": "c1",
        "name": "Acme Corporation",
        "address": "123 Main St, San Francisco, CA 94102",
        "phone": "+1-415-555-0100",
        "website": "https://acme.com",
    },
    {
        "id": "c1_dup1",
        "name": "Acme Corporation",
        "address": "123 Main St, San Francisco, CA 94102",
        "phone": "+1-415-555-0100",
        "website": "https://acme.com",
    },
    # Group 2: Typo in name (should match with medium-high score)
    {
        "id": "c2",
        "name": "TechStart Industries",
        "address": "456 Innovation Dr, Palo Alto, CA 94301",
        "phone": "+1-650-555-0200",
        "website": "https://techstart.io",
    },
    {
        "id": "c2_typo",
        "name": "TechStrat Industries",  # Typo: TechStart -> TechStrat
        "address": "456 Innovation Dr, Palo Alto, CA 94301",
        "phone": "+1-650-555-0200",
        "website": "https://techstart.io",
    },
    # Group 3: Abbreviation (should match with medium score)
    {
        "id": "c3",
        "name": "Global Systems Incorporated",
        "address": "789 Corporate Blvd, San Jose, CA 95110",
        "phone": "+1-408-555-0300",
        "website": "https://globalsys.com",
    },
    {
        "id": "c3_abbrev",
        "name": "Global Systems Inc.",  # Abbreviated: Incorporated -> Inc.
        "address": "789 Corporate Blvd, San Jose, CA 95110",
        "phone": "+1-408-555-0300",
        "website": "https://globalsys.com",
    },
    # Group 4: Missing optional fields (should match based on name)
    {
        "id": "c4",
        "name": "DataFlow Solutions",
        "address": "321 Tech Way, Mountain View, CA 94040",
        "phone": "+1-650-555-0400",
        "website": "https://dataflow.net",
    },
    {
        "id": "c4_partial",
        "name": "DataFlow Solutions",
        # Missing address, phone, website
    },
    # Group 5: Address variation (same company, different address format)
    {
        "id": "c5",
        "name": "CloudNet Services",
        "address": "555 Market Street, Suite 1200, San Francisco, CA 94105",
        "phone": "+1-415-555-0500",
        "website": "https://cloudnet.com",
    },
    {
        "id": "c5_addr_var",
        "name": "CloudNet Services",
        "address": "555 Market St Ste 1200, SF, CA 94105",  # Abbreviated address
        "phone": "+1-415-555-0500",
        "website": "https://cloudnet.com",
    },
    # Non-duplicates (should NOT match with each other or above)
    {
        "id": "c6",
        "name": "Quantum Dynamics",
        "address": "100 Research Park, Berkeley, CA 94720",
        "phone": "+1-510-555-0600",
        "website": "https://quantumdynamics.com",
    },
    {
        "id": "c7",
        "name": "NexGen Robotics",
        "address": "200 Innovation Circle, Sunnyvale, CA 94086",
        "phone": "+1-408-555-0700",
        "website": "https://nexgenrobotics.com",
    },
    {
        "id": "c8",
        "name": "BioTech Research Labs",
        "address": "300 Science Ave, South San Francisco, CA 94080",
        "phone": "+1-650-555-0800",
        "website": "https://biotechresearch.com",
    },
    {
        "id": "c9",
        "name": "Pacific Logistics Group",
        "address": "400 Harbor Blvd, Oakland, CA 94607",
        "phone": "+1-510-555-0900",
        "website": "https://pacificlogistics.com",
    },
    {
        "id": "c10",
        "name": "Green Energy Partners",
        "address": "500 Sustainability Way, San Rafael, CA 94901",
        "phone": "+1-415-555-1000",
        "website": "https://greenenergypartners.com",
    },
]

# Expected duplicate groups (ground truth for evaluation)
EXPECTED_DUPLICATE_GROUPS = [
    {"c1", "c1_dup1"},  # Exact duplicates
    {"c2", "c2_typo"},  # Typo duplicates
    {"c3", "c3_abbrev"},  # Abbreviation
    {"c4", "c4_partial"},  # Missing fields
    {"c5", "c5_addr_var"},  # Address variation
    # c6-c10 are singletons (no duplicates)
]
