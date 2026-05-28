from . import failure_communication_v0

VERSIONS = {
    "v0": failure_communication_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
