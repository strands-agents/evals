from . import helpfulness_v0

VERSIONS = {
    "v0": helpfulness_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
