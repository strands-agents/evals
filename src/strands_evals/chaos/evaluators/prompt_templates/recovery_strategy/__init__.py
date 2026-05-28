from . import recovery_strategy_v0, recovery_strategy_v1

VERSIONS = {
    "v0": recovery_strategy_v0,
    "v1": recovery_strategy_v1,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]
