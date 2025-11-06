"""
Actor profile templates for simulation.

This module provides actor profile structures used as templates
for generating realistic actor profiles in conversation simulation.
"""

DEFAULT_USER_PROFILE_SCHEMA = {
    "traits": {
        "personal_profile": {
            "identity": {
                "first_name": "User",
                "last_name": "Default",
                "preferred_name": "User",
                "gender": "other",
                "birthdate": "1990-01-01",
                "email": "user@example.com",
            },
            "location": {"address1": "123 Main St", "city": "Default City", "province": "CA", "country": "USA"},
            "languages": [{"language": "English", "proficiency": "Advanced"}],
        },
        "persona": "Friendly and helpful user seeking assistance with general topics.",
        "supplementary_profile": "Default user profile for simulation.",
    },
    "context": "some context",
}
