from . import partial_completion_v0, partial_completion_v1

VERSIONS = {
    "v0": partial_completion_v0,
    "v1": partial_completion_v1,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
