from . import attack_success_v0, attack_success_v1

VERSIONS = {
    "v0": attack_success_v0,
    "v1": attack_success_v1,
}

# v1 is the default: it defines what success means per risk category (v0 left it generic,
# which produced systematic judge errors -- see attack_success_v1 docstring). v0 is retained
# only for reproducing pre-v1 benchmark numbers.
DEFAULT_VERSION = "v1"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
