from rsl_rl.utils.wandb_environment import resolve_wandb_entity


def test_wandb_entity_is_canonical():
    assert (
        resolve_wandb_entity(
            {"WANDB_ENTITY": "team", "WANDB_USERNAME": "legacy-user"}
        )
        == "team"
    )


def test_wandb_username_remains_a_compatibility_fallback():
    assert resolve_wandb_entity({"WANDB_USERNAME": "legacy-user"}) == "legacy-user"


def test_wandb_entity_is_optional():
    assert resolve_wandb_entity({}) is None
