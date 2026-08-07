from . import skill_selection_accuracy_v0

VERSIONS = {
    "v0": skill_selection_accuracy_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
