import os
from typing import Union

from molscore import resources
from molscore.scoring_functions.base import BaseServerSF


class ChemPropModel(BaseServerSF):
    """
    Score structures by loading a pre-trained chemprop model and return the aggregated predicted values
    """

    # How to handle return metrics
    return_metrics = [
        "mean_pred",
        "max_pred",
        "min_pred",
        "mean_unc",
        "max_unc",
        "min_unc",
    ]

    def __init__(
        self,
        prefix: str,
        model_dir: Union[os.PathLike, str],
        env_engine: str = "mamba",
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param model_dir: Path to pre-trained model directory
        :param env_engine: Environment engine [conda, mamba]
        """

        super().__init__(
            prefix=prefix,
            env_engine=env_engine,
            env_name="chemprop",
            env_path=resources.files("molscore.data.models.chemprop").joinpath(
                "chemprop.yml"
            ),
            server_path=resources.files("molscore.scoring_functions.servers").joinpath(
                "chemprop_server.py"
            ),
            server_grace=30,
            server_kwargs={"model_dir": os.path.abspath(model_dir)},
        )
