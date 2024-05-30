import logging

from molscore import resources
from molscore.scoring_functions.base import BaseServerSF

logger = logging.getLogger("molskill")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class MolSkill(BaseServerSF):
    """
    Medicinal chemistry ranking by MolSkill https://doi.org/10.1038/s41467-023-42242-1
    """

    return_metrics = ["score"]

    def __init__(
        self,
        prefix: str = "MolSkill",
        env_engine: str = "mamba",
        server_grace: int = 60,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance
        :param env_engine: Environment engine [conda, mamba]
        """

        super().__init__(
            prefix=prefix,
            env_engine=env_engine,
            env_name="molskill",
            env_path=resources.files("molscore.data.models.molskill").joinpath(
                "environment.yml"
            ),
            server_path=resources.files("molscore.scoring_functions.servers").joinpath(
                "molskill_server.py"
            ),
            server_grace=server_grace,
        )
