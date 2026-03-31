from . import red_team_judge_v0

VERSIONS = {
    "v0": red_team_judge_v0,
}

DEFAULT_VERSION = "v0"


def get_template(version: str = DEFAULT_VERSION):
    return VERSIONS[version]


def build_system_prompt(metrics: set[str], version: str = DEFAULT_VERSION) -> str:
    return VERSIONS[version].build_system_prompt(metrics)
