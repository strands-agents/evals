from . import root_cause_merge_v0, root_cause_v0

VERSIONS = {
    "v0": root_cause_v0,
}

MERGE_VERSIONS = {
    "v0": root_cause_merge_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    if version not in VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {list(VERSIONS.keys())}")
    return VERSIONS[version]


def get_merge_template(version: str = DEFAULT_VERSION):
    if version not in MERGE_VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {list(MERGE_VERSIONS.keys())}")
    return MERGE_VERSIONS[version]
