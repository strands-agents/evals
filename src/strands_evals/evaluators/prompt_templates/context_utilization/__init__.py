from . import context_utilization_v0

VERSIONS = {
    "v0": context_utilization_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
