import logging

from molscore import resources
from molscore.scoring_functions.base import BaseServerSF

logger = logging.getLogger("legacy_qsar")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class LegacyQSAR(BaseServerSF):
    """
    Run published QSAR models in a stand alone environment to avoid conflict dependencies.
    """

    return_metrics = ["pred_proba"]
    # Directory of env-name, and resource path to relevant dir, model name & server name
    model_dictionary = {
        "libinvent_DRD2": (
            "lib-invent", # env-name
            resources.files("molscore.data.models.libinvent"), # resource path
            "drd2.pkl", # model name
            "legacy_qsar_server.py", # server name
            "ECFP6", # fp
            2048, # nBits
        ),
        "molopt_DRD2": (
            "ms_molopt",
            resources.files("molscore.data.models.molopt"),
            "drd2.pkl",
            "legacy_qsar_server.py",
            "FCFP6c",
            2048,
        ),
        "molopt_DRD2_current": (
            "ms_molopt_current",
            resources.files("molscore.data.models.molopt"),
            "drd2_current.pkl",
            "legacy_qsar_server.py",
            "FCFP6c",
            2048,
        ),
        "molopt_GSK3B": (
            "ms_molopt",
            resources.files("molscore.data.models.molopt"),
            "gsk3b.pkl",
            "legacy_qsar_server.py",
            "ECFP4",
            2048,
        ),
        "molopt_GSK3B_current": (
            "ms_molopt_current",
            resources.files("molscore.data.models.molopt"),
            "gsk3b_current.pkl",
            "legacy_qsar_server.py",
            "ECFP4",
            2048,
        ),
        "molopt_JNK3": (
            "ms_molopt",
            resources.files("molscore.data.models.molopt"),
            "jnk3.pkl",
            "legacy_qsar_server.py",
            "ECFP4",
            2048,
        ),
        "molopt_JNK3_current": (
            "ms_molopt_current",
            resources.files("molscore.data.models.molopt"),
            "jnk3_current.pkl",
            "legacy_qsar_server.py",
            "ECFP4",
            2048,
        ),
        "diversehits_DRD2": (
            "diversehits",
            resources.files("molscore.data.models.diversehits"),
            "drd2.pkl",
            "legacy_qsar_server.py",
            "ECFP4",
            2048,
        ),
        "diversehits_GSK3": (
            "diversehits",
            resources.files("molscore.data.models.diversehits"),
            "gsk3.pkl",
            "legacy_qsar_server.py",
            "ECFP4",
            2048,
        ),
        "diversehits_JNK3": (
            "diversehits",
            resources.files("molscore.data.models.diversehits"),
            "jnk3.pkl",
            "legacy_qsar_server.py",
            "ECFP4",
            2048,
        )
    }

    def __init__(self, prefix: str, model: str, env_engine: str = "mamba", server_grace: int = 60, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param env_envine: Environment engine [conda, mamba]
        :param model: Which legacy model to implement [libinvent_DRD2, molopt_DRD2, molopt_DRD2_current, molopt_GSK3B, molopt_GSK3B_current, molopt_JNK3, molopt_JNK3_current, diversehits_DRD2, diversehits_GSK3, diversehits_JNK3]
        :pararm server_grace: Grace period for server to start before timing out
        """
        # Get model resources
        env_name, res, model_name, server_name, fp, nBits = self.model_dictionary[model]
        env_path = res.joinpath(f"{env_name}.yml")
        model_path = res.joinpath(model_name)
        server_path = resources.files("molscore.scoring_functions.servers").joinpath(
            server_name
        )
        assert env_path.exists(), f"Environment file not found at {env_path}"
        assert model_path.exists(), f"Model file not found at {model_path}"
        assert server_path.exists(), f"Server file not found at {server_path}"

        super().__init__(
            prefix=prefix,
            env_engine=env_engine,
            env_name=env_name,
            env_path=env_path,
            server_path=server_path,
            server_grace=server_grace,
            server_kwargs={"model_path": model_path, "fp": fp, "nBits": nBits},
        )
