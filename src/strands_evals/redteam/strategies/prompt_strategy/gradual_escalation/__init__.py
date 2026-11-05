from . import gradual_escalation_v0

VERSIONS = {
    "v0": gradual_escalation_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
