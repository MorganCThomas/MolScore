import logging

from molscore import resources
from molscore.scoring_functions.base import BaseServerSF

logger = logging.getLogger("rascore_xgb")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class RAScore_XGB(BaseServerSF):
    """
    Predicted synthetic feasibility according to solveability by AiZynthFinder https://doi.org/10.1039/d0sc05401a
    """

    return_metrics = ["pred_proba"]
    model_dictionary = {
        "ChEMBL": resources.files("molscore.data.models.RAScore").joinpath(
            "XGB_chembl_ecfp_counts/model.pkl"
        ),
        "GDB": resources.files("molscore.data.models.RAScore").joinpath(
            "XGB_gdbchembl_ecfp_counts/model.pkl"
        ),
        "GDBMedChem": resources.files("molscore.data.models.RAScore").joinpath(
            "XGB_gdbmedechem_ecfp_counts/model.pkl"
        ),
    }

    def __init__(
        self,
        prefix: str = "RAScore",
        env_engine: str = "mamba",
        model: str = "ChEMBL",
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance
        :param env_engine: Environment engine [conda, mamba]
        :param model: Either ChEMBL, GDB, GDBMedChem [ChEMBL, GDB, GDBMedChem]
        """
        assert (
            model in self.model_dictionary
        ), f"Model not found in {self.model_dictionary}"
        model_path = self.model_dictionary[model]
        assert model_path.exists(), f"Model file not found at {model_path}"

        super().__init__(
            prefix=prefix,
            env_engine=env_engine,
            env_name="rascore-env",
            env_path=resources.files("molscore.data.models.RAScore").joinpath(
                "environment.yml"
            ),
            server_path=resources.files("molscore.scoring_functions.servers").joinpath(
                "rascore_server.py"
            ),
            server_kwargs={"model_path": model_path},
        )
