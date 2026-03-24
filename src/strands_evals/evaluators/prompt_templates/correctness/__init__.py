from . import correctness_v0, correctness_with_reference_v0

VERSIONS = {
    "v0": correctness_v0,
}

REFERENCE_VERSIONS = {
    "v0": correctness_with_reference_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]


def get_reference_template(version: str = DEFAULT_VERSION):
    return REFERENCE_VERSIONS[version]
