import json
import logging
import os

from molscore import resources
from molscore.scoring_functions import utils
from molscore.scoring_functions.base import BaseServerSF

logger = logging.getLogger("pidgin")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class PIDGIN(BaseServerSF):
    """
    Download and run PIDGIN classification models (~11GB) via Zenodo to return the positive predictions, atleast one uniprot must be specified.
    """

    return_metrics = ["pred_proba"]

    @classmethod
    def get_uniprot_list(cls):
        """Load list of uniprots from file in MolScore, this is a copy taken from Zenodo"""
        uniprots_path = resources.files("molscore.data.models.PIDGINv5").joinpath(
            "uniprots.json"
        )
        with open(uniprots_path, "rt") as f:
            uniprots = json.load(f)
        uniprots = ["None"] + uniprots
        return uniprots

    @classmethod
    def get_uniprot_groups(cls):
        """Load a list of uniprot groups from file in MolScore, this is a copy taken from Zenodo"""
        groups = {"None": None}
        groups_path = resources.files("molscore.data.models.PIDGINv5").joinpath(
            "uniprots_groups.json"
        )
        with open(groups_path, "rt") as f:
            groups.update(json.load(f))
        return groups

    @classmethod
    def set_docstring(cls):
        """Set init docstring here as it's not a string literal"""
        init_docstring = f"""
            :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
            :param env_engine: Environment engine [conda, mamba]
            :param uniprot: Uniprot accession for classifier to use [{', '.join(cls.get_uniprot_list())}]
            :param uniprots: List of uniprot accessions for classifier to use
            :param uniprot_set: Set of uniprots based on protein class (level - name - size) [{', '.join(cls.get_uniprot_groups().keys())}]
            :param exclude_uniprot: Uniprot to exclude (useful to remove from a uniprot set) [{', '.join(cls.get_uniprot_list())}]
            :param exclude_uniprots: Uniprot list to exclude (useful to remove from a uniprot set)
            :param thresh: Concentration threshold of classifier [100 uM, 10 uM, 1 uM, 0.1 uM]
            :param method: How to aggregate the positive prediction probabilities accross classifiers [mean, median, max, min]
            :param binarise: Binarise predicted probability and return ratio of actives based on optimal predictive thresholds (GHOST)
            :param kwargs:
            """
        setattr(cls.__init__, "__doc__", init_docstring)

    def __init__(
        self,
        prefix: str,
        env_engine: str = "mamba",
        uniprot: str = None,
        uniprots: list = None,
        uniprot_set: str = None,
        thresh: str = "100 uM",
        exclude_uniprot: str = None,
        exclude_uniprots: list = None,
        n_jobs: int = 1,
        method: str = "mean",
        binarise=False,
        **kwargs,
    ):
        """This docstring is must be populated by calling PIDGIN.set_docstring() first."""
        # Check if .pidgin_data exists
        if not utils.check_path(os.path.join(os.environ["HOME"], ".pidgin_data")):
            logger.warning(
                f"{os.path.join(os.environ['HOME'], '.pidgin_data')} not found, PIDGINv5 (11GB) will be download which may take several minutes"
            )
        # Make sure something is selected
        self.uniprot = uniprot if uniprot != "None" else None
        self.uniprots = uniprots if uniprots is not None else []
        self.uniprot_set = uniprot_set if uniprot_set != "None" else None
        self.exclude_uniprot = exclude_uniprot if exclude_uniprot != "None" else None
        self.exclude_uniprots = exclude_uniprots if exclude_uniprots is not None else []
        assert (
            (self.uniprot is not None)
            or (len(self.uniprots) > 0)
            or (self.uniprot_set is not None)
        ), "Either uniprot, uniprots or uniprot set must be specified"
        # Set other attributes
        self.thresh = thresh.replace(" ", "").replace(".", "")
        self.n_jobs = n_jobs
        self.method = method
        self.binarise = binarise
        # Curate uniprot set
        self.groups = self.get_uniprot_groups()
        if self.uniprot:
            self.uniprots += [self.uniprot]
        if self.uniprot_set:
            self.uniprots += self.groups[self.uniprot_set]
        if self.exclude_uniprot:
            self.exclude_uniprots += [self.exclude_uniprot]
        for uni in self.exclude_uniprots:
            if uni in self.uniprots:
                self.uniprots.remove(uni)
        # De-duplicate
        self.uniprots = list(set(self.uniprots))

        # Set server kwargs
        server_kwargs = {
            "thresh": self.thresh,
            "method": self.method,
            "n_jobs": self.n_jobs,
            "uniprots": " ".join(self.uniprots),
        }
        if self.binarise:
            server_kwargs["binarise"] = ""

        super().__init__(
            prefix=prefix,
            env_engine=env_engine,
            env_name="pidgin",
            env_path=resources.files("molscore.data.models.PIDGINv5").joinpath(
                "environment.yml"
            ),
            server_path=resources.files("molscore.scoring_functions.servers").joinpath(
                "pidgin_server.py"
            ),
            server_grace=600,
            server_kwargs=server_kwargs,
        )
