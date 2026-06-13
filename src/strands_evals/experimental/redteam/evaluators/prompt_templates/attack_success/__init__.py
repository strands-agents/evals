from . import attack_success_v0, attack_success_v1

VERSIONS = {
    "v0": attack_success_v0,
    "v1": attack_success_v1,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
