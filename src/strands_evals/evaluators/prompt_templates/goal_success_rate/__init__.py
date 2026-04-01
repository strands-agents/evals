from . import goal_success_rate_v0, goal_success_rate_with_assertions_v0

VERSIONS = {
    "v0": goal_success_rate_v0,
}

ASSERTION_VERSIONS = {
    "v0": goal_success_rate_with_assertions_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]


def get_assertion_template(version: str = DEFAULT_VERSION):
    return ASSERTION_VERSIONS[version]
