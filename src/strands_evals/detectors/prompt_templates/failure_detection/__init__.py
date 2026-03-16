from . import failure_detection_v0

VERSIONS = {
    "v0": failure_detection_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    if version not in VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {list(VERSIONS.keys())}")
    return VERSIONS[version]
