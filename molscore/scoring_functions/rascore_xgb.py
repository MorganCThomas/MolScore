import ast
import atexit
import logging
import os
import signal
import socket
import subprocess
import time

import requests

from molscore import resources
from molscore.scoring_functions.utils import timedSubprocess

logger = logging.getLogger("rascore_xgb")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class RAScore_XGB:
    """
    Predicted synthetic feasibility according to solveability by AiZynthFinder https://doi.org/10.1039/d0sc05401a
    """

    return_metrics = ["pred_proba"]

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
        :param kwargs:
        """
        self.prefix = prefix.replace(" ", "_")
        self.engine = env_engine
        self.server_subprocess = None

        self.subprocess = timedSubprocess()
        self.env_name = "rascore-env"
        self.env_path = resources.files("molscore.data.models.RAScore").joinpath(
            "environment.yml"
        )
        self.model_name = model
        self.server_path = resources.files(
            "molscore.scoring_functions.servers"
        ).joinpath("rascore_server.py")

        # Check/create RAscore Environment
        if not self._check_env():
            logger.warning(
                f"Failed to identify {self.env_name}, attempting to create it automatically (this may take several minutes)"
            )
            self._create_env()
            logger.info(f"{self.env_name} successfully created")
        else:
            logger.info(f"Found existing {self.env_name}")

        if self.model_name == "ChEMBL":
            self.model_path = resources.files("molscore.data.models.RAScore").joinpath(
                "XGB_chembl_ecfp_counts/model.pkl"
            )
        elif self.model_name == "GDB":
            self.model_path = resources.files("molscore.data.models.RAScore").joinpath(
                "XGB_gdbchembl_ecfp_counts/model.pkl"
            )
        elif self.model_name == "GDBMedChem":
            self.model_path = resources.files("molscore.data.models.RAScore").joinpath(
                "XGB_gdbmedechem_ecfp_counts/model.pkl"
            )
        else:
            raise "Please select from ChEMBL, GDB or GDBMedChem"

        # Launch server
        self._launch_server()
        atexit.register(self._kill_server)

    def _check_env(self):
        cmd = f"{self.engine} info --envs"
        out = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        envs = [line.split(" ")[0] for line in out.stdout.decode().splitlines()[2:]]
        return self.env_name in envs

    def _create_env(self):
        cmd = f"{self.engine} env create -f {self.env_path}"
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to create {self.env_name} automatically please install as per instructions on respective GitHub with correct name"
            )
            raise e

    @staticmethod
    def _check_port(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _launch_server(self):
        port = 8000
        while self._check_port(port):
            port += 1
        self.server_cmd = f"{self.engine} run -n {self.env_name} python {self.server_path} --port {port} --prefix {self.prefix} --model_path {self.model_path}"
        self.server_url = f"http://localhost:{port}"
        logger.info(f"Launching server: {self.server_cmd}")
        try:
            logger.info("Leaving a grace period of 20s for server to launch")
            self.server_subprocess = subprocess.Popen(
                self.server_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            time.sleep(20)  # Ugly way to wait for server to launch
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to launch server, please check {self.server_path} is correct"
            )
            raise e

    def _kill_server(self):
        if self.server_subprocess is not None:
            os.killpg(os.getpgid(self.server_subprocess.pid), signal.SIGTERM)
            self.server_subprocess = None
            logger.info("Server killed")

    def send_smiles_to_server(self, smiles):
        payload = {"smiles": smiles}
        logger.debug(f"Sending payload to server: {payload}")
        try:
            response = requests.post(self.server_url + "/", json=payload)
        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"{e}: "
                f"\n\tAre sure the server was running at {self.server_url}?"
                f"\n\tAre you sure the right environment engine was used (I'm using {self.engine})?"
                f"\n\tAre you sure the following command runs? (Also try by loading the environment first)"
                f"\n\t{self.server_cmd}"
                f"\n\tAre you sure it loaded within 20 seconds?\n\n"
            )
            raise e
        if response.status_code == 200:
            results = response.json()
            logger.debug(f"Result from server: {results}")
        else:
            results = [{"smiles": smi} for smi in smiles]
            _ = [r.update({m: 0.0 for m in self.return_metrics}) for r in results]
            logger.error(f"Error {response.status_code}: {response.text}")
        return results

    def __call__(self, smiles: list, **kwargs):
        results = self.send_smiles_to_server(smiles)
        # Convert strings back to interpreted type
        results = [
            {k: ast.literal_eval(v) if k != "smiles" else v for k, v in r.items()}
            for r in results
        ]
        return results
