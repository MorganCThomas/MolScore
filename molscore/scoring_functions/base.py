# This file contains templates for

import atexit
import logging
import os
import signal
import socket
import subprocess
import time
from typing import Dict, List

import requests

from molscore.scoring_functions.utils import check_exe, timedSubprocess

logger = logging.getLogger("base")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class BaseSF:
    """Description"""  # Description should be at the class level, this is passed to the config GUI

    return_metrics = []  # Name of metrics returned so that they can be selected in the config GUI

    def __init__(self, prefix: str, **kwargs):
        # Typing and default values are passed to the config GUI
        # PyCharm style docstring should be used for parameters only (example below), these are passed to the config GUI
        # Additional choices can be specified in square brackets, for example, [Choice 1, Choice 2, Choice 3]. This will result in a dropdown list in the config GUI. Hence, avoid the use of square brackets otherwise
        """
        :param prefix: Description
        """
        self.prefix = prefix.strip().replace(
            " ", "_"
        )  # Prefix to seperate multiple uses of the same class
        raise NotImplementedError("This is an abstract method")

    def __call__(self, smiles: list, file_names, directory, **kwargs) -> List[Dict]:
        raise NotImplementedError("This is an abstract method")
        # Results should be a list of dictionaries. Each dictionary corresponds to an input SMILES and should 'smiles' key. Every other key should be '<prefix>_<return_metric>'
        # For example,
        # [{'smiles': 'c1ccccc1', 'prefix_docking_score': -7.8},
        #  {'smiles': 'c1ccccc1C(=O)O', 'prefix_docking_score': -9.0]


class BaseServerSF:
    """Description"""

    return_metrics = []  # Name of metrics returned so that they can be selected in the config GUI

    def __init__(
        self,
        prefix: str,
        env_engine: str,
        env_name: str,
        server_path: str,
        env_path: str = None,
        server_grace: int = 60,
        server_kwargs: dict = {},
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance
        :param env_engine: Environment engine [conda, mamba]
        """
        self.prefix = prefix.replace(" ", "_")
        self.server_subprocess = None
        self.subprocess = timedSubprocess()
        self.env_name = env_name  # Name of python environment
        self.env_path = env_path  # Resource path to environment
        self.server_grace = server_grace  # Time to wait for server to launch
        self.server_path = server_path  # Resource path to server executable

        # Check engine
        if (env_engine == "mamba") and check_exe("mamba"):
            self.engine = "mamba"
        elif check_exe("conda"):
            self.engine = "conda"
        else:
            raise ValueError(
                "Could not find mamba or conda executables needed to create and run this environment"
            )

        # Setup any parameters that need to be passed to the server
        self.server_kwargs = server_kwargs

        # Check/create Environment
        if not self._check_env():
            logger.warning(
                f"Failed to identify {self.env_name}, attempting to create it automatically (this may take several minutes)"
            )
            if self.env_path:
                self._create_env()
                logger.info(f"{self.env_name} successfully created")
            else:
                raise RuntimeError(
                    f"Environment {self.env_name} not found and no environment path provided for installation. Please install the environment manually or provide the path to an environment file."
                )
        else:
            logger.info(f"Found existing {self.env_name}")

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
        kwargs = " ".join([f"--{k} {v}" for k, v in self.server_kwargs.items()])
        self.server_cmd = f"{self.engine} run -n {self.env_name} python {self.server_path} --port {port} {kwargs}"
        self.server_url = f"http://localhost:{port}"
        logger.info(f"Launching server: {self.server_cmd}")
        try:
            self.server_subprocess = subprocess.Popen(
                self.server_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            self._wait_for_server()
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to launch server, please check {self.server_path} is correct"
            )
            raise e

    def _wait_for_server(self):
        start_time = time.time()
        while True:
            try:
                response = requests.options(self.server_url)
                if response.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            if time.time() - start_time > self.server_grace:
                raise TimeoutError(
                    f"Server did not launch within grace period of {self.server_grace} seconds"
                )
            time.sleep(1)

    def _kill_server(self):
        if self.server_subprocess is not None:
            os.killpg(os.getpgid(self.server_subprocess.pid), signal.SIGTERM)
            self.server_subprocess = None
            logger.info("Server killed")

    def send_smiles_to_server(self, smiles):
        payload = {"smiles": smiles}
        #logger.debug(f"Sending payload to server: {payload}")
        try:
            response = requests.post(self.server_url + "/", json=payload)
        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"{e}: "
                f"\n\tAre sure the server was running at {self.server_url}?"
                f"\n\tAre you sure the right environment engine was used (I'm using {self.engine})?"
                f"\n\tAre you sure the following command runs? (Also try by loading the environment first)"
                f"\n\t{self.server_cmd}"
                f"\n\tAre you sure it loaded within {self.server_grace} seconds?\n\n"
            )
            raise e
        if response.status_code == 200:
            results = response.json()
            # Add prefix to metrics
            results = [
                {f"{self.prefix}_{k}" if k != "smiles" else k: v for k, v in r.items()}
                for r in results
            ]
            #logger.debug(f"Result from server: {results}")
        else:
            results = [{"smiles": smi} for smi in smiles]
            _ = [
                r.update({f"{self.prefix}_{m}": 0.0 for m in self.return_metrics})
                for r in results
            ]
            logger.error(f"Error {response.status_code}: {response.text}")
        return results

    def __call__(self, smiles: list, **kwargs):
        results = self.send_smiles_to_server(smiles)
        return results
