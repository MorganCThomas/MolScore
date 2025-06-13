import logging
import os

from molscore import resources
from molscore.scoring_functions.base import BaseServerSF

logger = logging.getLogger("boltz")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class Boltz(BaseServerSF):
    """
    Predicted complex structure using Boltz co-folding https://doi.org/10.1101/2024.11.19.624167
    """

    return_metrics = [
        "confidence_score",
        "pair_chain_iptm_protein-ligand",
        "affinity_pred_value",
        "affinity_probability_binary"
        ]

    def __init__(
        self,
        input_path: os.PathLike,
        prefix: str = "Boltz",
        env_engine: str = "mamba",
        cache: str = "~/.boltz",
        checkpoint: os.PathLike = None,
        devices: list = "0",
        accelerator: str = "gpu",
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        step_scale: float = 1.638,
        output_format: str = "mmcif",
        num_workers: int = 2,
        override: bool = False,
        use_msa_server: bool = True,
        msa_server_url: str = "https://api.colabfold.com",
        msa_pairing_strategy: str = "greedy",
        write_full_pae: bool = False,
        write_full_pde: bool = False,
        **kwargs,
    ):
        """
        :param prefix: Prefix to identify scoring function instance
        :param env_engine: Environment engine [conda, mamba]
        :param input_path: Path to input YAML file (only YAML format is accepted)
        :param cache: The directory where to download the data and model.
        :param checkpoint: An optional checkpoint. Uses the provided Boltz-1 model by default.
        :param devices: The INDEX of devices to use for prediction. Note this is different to Boltz CLI.
        :param accelerator: The accelerator to use for prediction. [gpu, cpu, tpu]
        :param recycling_steps: The number of recycling steps to use for prediction.
        :param sampling_steps: The number of sampling steps to use for prediction.
        :param diffusion_samples: The number of diffusion samples to use for prediction.
        :param step_scale: The step size is related to the temperature at which the diffusion process samples the distribution. The lower the higher the diversity among samples (recommended between 1 and 2).
        :param output_format: The output format to use for the predictions. [mmcif, pdb]
        :param num_workers: The number of dataloader workers to use for prediction.
        :param override: Whether to override existing predictions if found.
        :param use_msa_server: Whether to use the msa server to generate msa's.
        :param msa_server_url: MSA server url. Used only if --use_msa_server is set.
        :param msa_pairing_strategy: Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'.
        :param write_full_pae: Whether to save the full PAE matrix as a file.
        :param write_full_pde: Whether to save the full PDE matrix as a file.
        """

        server_kwargs = {
            "prefix": prefix,
            "input_path": input_path,
            "cache": cache,
            "checkpoint": checkpoint,
            "devices": " ".join([str(d) for d in devices]),
            "accelerator": accelerator,
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "step_scale": step_scale,
            "output_format": output_format,
            "num_workers": num_workers,
            "msa_server_url": msa_server_url,
            "msa_pairing_strategy": msa_pairing_strategy,
        }
        server_kwargs = {k: v for k, v in server_kwargs.items() if v is not None}
        server_args = [
            k
            for k, v in {
                "override": override,
                "use_msa_server": use_msa_server,
                "write_full_pae": write_full_pae,
                "write_full_pde": write_full_pde,
            }.items()
            if v
        ]

        super().__init__(
            prefix=prefix,
            env_engine=env_engine,
            env_name="boltz",
            env_path=resources.files("molscore.data.models.boltz").joinpath(
                "environment.yml"
            ),
            server_path=resources.files("molscore.scoring_functions.servers").joinpath(
                "boltz_server.py"
            ),
            server_kwargs=server_kwargs,
            server_args=server_args,
        )
