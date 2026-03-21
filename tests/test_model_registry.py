import pytest  # type: ignore

from src.model_registry import build_model_artifact_reference
from src.model_registry import load_wandb_public_api
from src.model_registry import promote_model_artifact


def _config() -> dict:
    return {
        "wandb": {
            "entity": "group8",
            "project": "spotify-sound-archetypes",
            "model_artifact_name": "spotify-popularity-pipeline",
            "production_alias": "prod",
        }
    }


def test_build_model_artifact_reference_uses_explicit_alias():
    assert build_model_artifact_reference(_config(), alias="latest") == (
        "group8/spotify-sound-archetypes/spotify-popularity-pipeline:latest"
    )


def test_build_model_artifact_reference_defaults_to_production_alias():
    assert build_model_artifact_reference(_config()) == (
        "group8/spotify-sound-archetypes/spotify-popularity-pipeline:prod"
    )


def test_build_model_artifact_reference_requires_project_and_artifact_name():
    with pytest.raises(ValueError, match="wandb.project and wandb.model_artifact_name"):
        build_model_artifact_reference({"wandb": {"entity": "group8"}})


def test_load_wandb_public_api_raises_when_missing(monkeypatch):
    def boom(name):
        raise ImportError("missing")

    monkeypatch.setattr("src.model_registry.importlib.import_module", boom)

    with pytest.raises(ImportError, match="wandb is required"):
        load_wandb_public_api()


def test_load_wandb_public_api_raises_when_api_missing(monkeypatch):
    class BrokenWandb:
        pass

    monkeypatch.setattr(
        "src.model_registry.importlib.import_module",
        lambda name: BrokenWandb(),
    )

    with pytest.raises(ImportError, match="does not expose the Api client"):
        load_wandb_public_api()


def test_load_wandb_public_api_returns_module(monkeypatch):
    class ValidWandb:
        class Api:
            pass

    monkeypatch.setattr(
        "src.model_registry.importlib.import_module",
        lambda name: ValidWandb(),
    )

    assert load_wandb_public_api() is not None


def test_build_model_artifact_reference_requires_dict_section():
    with pytest.raises(ValueError, match="wandb section is required"):
        build_model_artifact_reference({"wandb": "bad"})  # type: ignore[arg-type]


def test_build_model_artifact_reference_supports_missing_entity():
    assert build_model_artifact_reference(
        {
            "wandb": {
                "project": "spotify-sound-archetypes",
                "model_artifact_name": "spotify-popularity-pipeline",
                "production_alias": "prod",
            }
        }
    ) == "spotify-sound-archetypes/spotify-popularity-pipeline:prod"


def test_promote_model_artifact_adds_target_alias():
    class FakeArtifact:
        def __init__(self):
            self.aliases = ["latest"]
            self.version = "v12"
            self.saved = False

        def save(self):
            self.saved = True

    class FakeApi:
        def __init__(self):
            self.artifact_ref = None
            self.artifact_obj = FakeArtifact()

        def artifact(self, reference):
            self.artifact_ref = reference
            return self.artifact_obj

    class FakeWandb:
        def __init__(self):
            self.api = FakeApi()

        def Api(self):
            return self.api

    fake_wandb = FakeWandb()
    result = promote_model_artifact(
        _config(),
        source_alias="latest",
        target_alias="prod",
        wandb_module=fake_wandb,
    )

    assert fake_wandb.api.artifact_ref == (
        "group8/spotify-sound-archetypes/spotify-popularity-pipeline:latest"
    )
    assert result["promoted_reference"] == (
        "group8/spotify-sound-archetypes/spotify-popularity-pipeline:prod"
    )
    assert result["artifact_version"] == "v12"
    assert "prod" in result["aliases"]
    assert fake_wandb.api.artifact_obj.saved is True


def test_promote_model_artifact_requires_save_method():
    class BrokenArtifact:
        aliases = ["latest"]

    class FakeApi:
        def artifact(self, reference):
            return BrokenArtifact()

    class FakeWandb:
        def Api(self):
            return FakeApi()

    with pytest.raises(AttributeError, match="save"):
        promote_model_artifact(_config(), wandb_module=FakeWandb())


def test_promote_model_artifact_uses_default_aliases():
    class FakeArtifact:
        def __init__(self):
            self.aliases = ["latest", "prod"]
            self.version = "v12"
            self.saved = False

        def save(self):
            self.saved = True

    class FakeApi:
        def __init__(self):
            self.artifact_ref = None
            self.artifact_obj = FakeArtifact()

        def artifact(self, reference):
            self.artifact_ref = reference
            return self.artifact_obj

    class FakeWandb:
        def __init__(self):
            self.api = FakeApi()

        def Api(self):
            return self.api

    fake_wandb = FakeWandb()
    result = promote_model_artifact(
        _config(),
        source_alias="",
        target_alias=None,
        wandb_module=fake_wandb,
    )

    assert fake_wandb.api.artifact_ref.endswith(":latest")
    assert result["promoted_reference"].endswith(":prod")
    assert result["aliases"].count("prod") == 1
