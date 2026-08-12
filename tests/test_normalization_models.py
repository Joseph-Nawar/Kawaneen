from __future__ import annotations

from kawaneen.normalization import NormalizationPolicy, all_policies


def test_policy_config_is_immutable_and_typed() -> None:
    policy = all_policies()[0]
    assert isinstance(policy, NormalizationPolicy)
    assert policy.transforms[0] == "nfc"
    assert policy.config["schema_version"] == 1
