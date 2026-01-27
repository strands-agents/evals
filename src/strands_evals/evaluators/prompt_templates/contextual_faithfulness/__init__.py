from . import contextual_faithfulness_v0

VERSIONS = {
    "v0": contextual_faithfulness_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
