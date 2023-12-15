import requests
import logging
import subprocess
import atexit
import socket
import os
import signal
from importlib import resources

logger = logging.getLogger('legacy_qsar')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

class LegacyQSAR:
    """
    Run published QSAR models in a stand alone environment to avoid conflict dependencies.
    """
    return_metrics = ['pred_proba']
    # Directory of env-name, and resource path to relevant dir, model name & server name
    model_dictionary = {
        'libinvent_DRD2': ('lib-invent', resources.files('molscore.data.models.libinvent'), 'drd2.pkl', 'libinvent_DRD2_server.py'),
    }

    def __init__(self, prefix: str, env_engine: str, model: str, **kwargs):
        """
        :param prefix: Prefix to identify scoring function instance (e.g., DRD2)
        :param env_engine: Environment engine [conda, mamba]
        :param model: Which legacy model to implement [libinvent_DRD2]
        :param kwargs:
        """
        self.prefix = prefix.replace(" ", "_")
        self.engine = env_engine
        self.server_process = None
        
        # Get model resources
        self.env_name, res, model_name, server_name = self.model_dictionary[model]
        self.env_path = res.joinpath('environment.yml')
        self.model_path = res.joinpath(model_name)
        self.server_path = resources.files(f'molscore.scoring_functions.servers').joinpath(server_name)
        assert self.env_path.exists(), f"Environment file not found at {self.env_path}"
        assert self.model_path.exists(), f"Model file not found at {self.model_path}"
        assert self.server_path.exists(), f"Server file not found at {self.server_path}"

        # Check/create environment
        if not self._check_env():
            logger.warning(f"Failed to identify {self.env_name}, attempting to create it automatically (this may take several minutes)")
            self._create_env()
            logger.info(f"{self.env_name} successfully created")
        else:
            logger.info(f"Existing {self.env_name} found")

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
            logger.error(f"Failed to create {self.env_name} automatically please install as per instructions on respective GitHub with correct name")
            raise e

    @staticmethod
    def _check_port(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def _launch_server(self):
        port = 8000
        while self._check_port(port):
            port += 1
        cmd = f"{self.engine} run -n {self.env_name} python {self.server_path} --port {port} --prefix {self.prefix} --model_path {self.model_path}"
        self.server_url = f"http://localhost:{port}"
        logger.info(f"Launching server: {cmd}")
        try:
            self.server_subprocess = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to launch server, please check {self.server_path} is correct")
            raise e
    
    def _kill_server(self):
        if self.server_subprocess is not None:
            os.killpg(os.getpgid(self.server_subprocess.pid), signal.SIGTERM)
            self.server_subprocess = None
            logger.info("Server killed")

    def send_smiles_to_server(self, smiles):
        payload = {'smiles': smiles}
        logger.debug(f"Sending payload to server: {payload}")
        response = requests.post(self.server_url+"/", json=payload)
        if response.status_code == 200:
            results = response.json()
            logger.debug(f"Result from server: {results}")
        else:
            results = [{'smiles': smi} for smi in smiles]
            _ = [r.update({m: 0.0 for m in self.return_metrics}) for r in results]
            logger.error(f"Error {response.status_code}: {response.text}")
        return results

    def __call__(self, smiles, directory, file_names, **kwargs):
        results = self.send_smiles_to_server(smiles) 
        return results