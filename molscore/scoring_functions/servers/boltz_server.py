import argparse
import json
import logging
import os
import signal
import subprocess
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np
import yaml
from flask import Flask, jsonify, request

from boltz.main import compute_msa

logger = logging.getLogger("boltz_server")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

app = Flask(__name__)


class timedSubprocess:
    def __init__(self, timeout=None, shell=False):
        self.cmd = None
        self.cwd = None
        self.timeout = timeout
        self.shell = shell
        self.process = None

    def run(self, cmd, cwd=None):
        if not self.shell:
            self.cmd = cmd.split()
            self.process = subprocess.Popen(
                self.cmd,
                preexec_fn=os.setsid,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            self.cmd = cmd
            self.process = subprocess.Popen(
                self.cmd,
                shell=self.shell,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        try:
            out, err = self.process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            print("Process timed out...")
            out, err = (
                "".encode(),
                f"Timed out at {self.timeout}".encode(),
            )  # Encode for consistency
            if not self.shell:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.process.kill()
        return out, err


class Model:
    """
    Create a Model object that loads models once, for example.
    """

    def __init__(self, **kwargs):
        self.prefix = kwargs["prefix"]
        self.msa_server_url = kwargs["msa_server_url"]
        self.msa_pairing_strategy = kwargs["msa_pairing_strategy"]
        del kwargs["prefix"]
        del kwargs["port"]

        # Format n_devices
        devices = kwargs["devices"]
        self.cuda_devices = ",".join([str(d) for d in devices])
        kwargs["devices"] = len(devices)

        # Load yaml file
        self.input = Path(kwargs["input_path"])
        with open(self.input, "r") as f:
            self.config = yaml.safe_load(f)
        del kwargs["input_path"]

        # Load kwargs
        self.kwargs = {k: v for k, v in kwargs.items() if v}

        # Load args
        self.args = []
        for k, v in kwargs.items():
            if isinstance(v, bool) and v:
                self.args.append(k)
        for a in self.args:
            del self.kwargs[a]
        
        self.add_msa()
        self.add_affinity()


    def add_msa(self):
        for i, seq_data in enumerate(self.config["sequences"]):
            if "protein" in seq_data:
                if not "msa" in seq_data["protein"]:
                    # Compute msa
                    chain_id = seq_data["protein"]["id"][0]
                    seq = seq_data["protein"]["sequence"]
                    msa_id = f"{self.input.stem}_{chain_id}_msa"
                    msa_dir = self.input.parent
                    compute_msa(
                        data={msa_id: seq}, # dict {input_id_chain_id: seq}
                        target_id=self.input.stem, # str target_id (record.id)
                        msa_dir=msa_dir,
                        msa_server_url=self.msa_server_url,
                        msa_pairing_strategy=self.msa_pairing_strategy,
                    )
                    # Update config
                    self.config["sequences"][i]["protein"]["msa"] = str(msa_dir / f"{msa_id}.csv")

         
    def add_affinity(self):
        # For now we just replace
        self.config["properties"] = [{
                "affinity": {
                    "binder": "Z"
                }
            }]
        

    def add_smiles(self, smiles: str, out_path: str):
        config = deepcopy(self.config)
        config["sequences"].append({"ligand": {"id": ["Z"], "smiles": smiles}})
        with open(out_path, "w") as f:
            yaml.dump(config, f)


@app.route("/", methods=["POST"])
def compute():
    # Get JSON from request
    smiles = request.get_json().get("smiles", [])
    directory = request.get_json().get("directory", ".")
    file_names = request.get_json().get("file_names", [])

    # Create directories
    step_dir = file_names[0].split("_")[0]
    in_dir = Path(f"{directory}/{model.prefix}_BoltzFold/{step_dir}/inputs/")
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir = in_dir.parent
    
    # Write configs
    for smi, fname in zip(smiles, file_names):
        in_path = in_dir / f"{fname}.yaml"
        model.add_smiles(smi, in_path)

    # Submit Boltz jobs
    p = timedSubprocess(timeout=None, shell=True).run
    # Check we're not running more devices thant jobs
    if model.kwargs["devices"] > len(smiles):
        model.kwargs["devices"] = len(smiles)
    # Format kwargs and args
    kwargs = " ".join([f"--{k} {v}" for k, v in model.kwargs.items()])
    args = " ".join([f"--{arg}" for arg in model.args])
    # Run command
    command = f"export CUDA_VISIBLE_DEVICES={model.cuda_devices} ; boltz predict {in_dir} --out_dir {out_dir} {kwargs} {args}"
    with open(out_dir / "command.txt", "wt") as f:
        f.write(command)
    out, err = p(command)

    # Update results
    results = []
    for smi, fname in zip(smiles, file_names):
        # Get prediction directory
        pred_dir = Path(f"{out_dir}/boltz_results_inputs/predictions/{fname}/")
        if not pred_dir.exists():
            results.append(dict(smiles=smi, confidence_score=0.0))
            continue
        else:
            result = {"smiles": smi}
            # Load confidence data
            pred_data = []
            for pred_path in sorted(glob(str(pred_dir / "confidence*.json"))):
                with open(pred_path, "r") as f:
                    pred_data.append(json.load(f))
            
            if pred_data:
                # Get max score
                max_idx = np.argmax([data["confidence_score"] for data in pred_data])
                pred_data = pred_data[max_idx]
                pred_data.update({"batch_variant": f"{fname}-{int(max_idx)}"})  # JSON serializable int
                # Update results, fix itm and iptm
                chains_ptm = pred_data.pop("chains_ptm")
                pred_data.update({f"chain_ptm_{i}": v for i, v in chains_ptm.items()})
                pair_chains_iptm = pred_data.pop("pair_chains_iptm")
                pred_data.update({f"pair_chain_iptm_{i}-{j}": pair_chains_iptm[i][j] for i in pair_chains_iptm for j in pair_chains_iptm[i]})
                # Assume the protein is the first and ligand is the last chain
                pred_data["pair_chain_iptm_protein-ligand"] = pair_chains_iptm["0"][str(len(pair_chains_iptm["0"]) - 1)]
                result.update(**pred_data)
                
            affin_data = []
            for affin_path in sorted(glob(str(pred_dir / "affinity*.json"))):
                with open(affin_path, "r") as f:
                    affin_data.append(json.load(f))
                    
            if affin_data:
                # Get max score based on probability of scoring
                max_idx = np.argmax([data["affinity_probability_binary"] for data in affin_data])
                affin_data = affin_data[max_idx]
                affin_data.update({"batch_variant": f"{fname}-{int(max_idx)}"})  # JSON serializable int
                result.update(**affin_data)
            
            results.append(result)
            
    return jsonify(results)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run a Boltz server to handle Boltz environment"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument(
        "--prefix",
        type=str,
        default="Boltz",
        help="Prefix to identify scoring function instance",
    )
    # Additional Boltz parameters
    parser.add_argument(
        "--input_path",
        type=str,
        default="./boltz_input.yaml",
        help="Path to input YAML file",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="~/.boltz",
        help="Directory for downloading data and model",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Optional checkpoint file"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=[0],
        nargs="+",
        help="Index of devices to use for prediction",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Accelerator to use",
        choices=["gpu", "cpu", "tpu"],
    )
    parser.add_argument(
        "--recycling_steps", type=int, default=3, help="Number of recycling steps"
    )
    parser.add_argument(
        "--sampling_steps", type=int, default=200, help="Number of sampling steps"
    )
    parser.add_argument(
        "--diffusion_samples", type=int, default=1, help="Number of diffusion samples"
    )
    parser.add_argument(
        "--step_scale",
        type=float,
        default=1.638,
        help="Step scale factor for diffusion sampling",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="mmcif",
        help="Output format",
        choices=["mmcif", "pdb"],
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Override existing predictions if found",
    )
    parser.add_argument(
        "--use_msa_server", action="store_true", default=False, help="Use msa server"
    )
    parser.add_argument(
        "--msa_server_url",
        type=str,
        default="https://api.colabfold.com",
        help="MSA server URL",
    )
    parser.add_argument(
        "--msa_pairing_strategy",
        type=str,
        default="greedy",
        help="MSA pairing strategy",
        choices=["greedy", "complete"],
    )
    parser.add_argument(
        "--write_full_pae",
        action="store_true",
        default=False,
        help="Save the full PAE matrix",
    )
    parser.add_argument(
        "--write_full_pde",
        action="store_true",
        default=False,
        help="Save the full PDE matrix",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = Model(**args.__dict__)
    app.run(port=args.port)
