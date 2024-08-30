# Using MolScore to run multi-parameter objectives

Once MolScore has been integrated into a generative model (see [here](implementing_molscore.md), or some ready-to-use examples [here](https://github.com/MorganCThomas/MolScore_examples)), all we need to do is write a JSON configuration file that defines objective parameters. I.e., configuration file that is passed via `MolScore(task_config=<path_to_configuration>)`.

## The configuration GUI

As MolScore now contains several options and parameters, the number of configurations is large and dependent on the scoring function being run. To help with the process of writing the config, a GUI is available that reads the docstrings and provides documentation and defaults. This can be run with via the following command (which is just a wrapper of `streamlit run molscore/gui/config.py`).

    molscore_config

This launches a GUI in your browser that walksthrough the definition of these configuration files. The underlying file structure is described [below](#the-configuration-file).

![alt text](https://github.com/MorganCThomas/MolScore/blob/v1.0/molscore/data/images/config_v1_albuterol.png?raw=True)

## Using MolScore GUI remotely
If you use MolScore on a remote server you can use SSH to listen to the remote port that the GUI is running on, and view it locally.

On your local machine, run the following command to setup your local port to listen to your remote port.

    ssh -L <local_port>:<local_host>:<remote_port> <remote_host>

For example, Streamlit GUIs typically run on port 8501, therefore I use use the following command. Remember to replace your remote host name.

    ssh -4 -NL 8501:localhost:8501 remote_host

-4 Forces connections using IPv4 addresses only  
-N Redirects stdin from /dev/null, which prevents reading from stdin  
-L Redirects data from the specified local port  

## The configuration file

The overall structure of the configuration file is,
- Logging
- Scoring functions
- Scoring endpoints (derived from the scoring functions above)
- Transformation of the scoring endpoints to \[0, 1\]
- Aggregation into the final score/reward
- Application of a diversity file

For example, see the following config file to calculate the QED.

```JSON
{
    # Logging
    "task": "BACE1_rDock-scaff_Occ",  # Name
    "output_dir": "./",  # Path to save results
    "load_from_previous": false,  # Optionally continue from previous run
    "monitor_app": true,  # Automatically start GUI monitor

    # Scoring functions 
    "scoring_functions": [
    {
      "name": "RDKitDescriptors", # Name
      "run": true, # Turn on/off
      "parameters": {
        "prefix": "desc"  # Specification scoring function parameters
      }
    }
    ],

     # Scoring endpoints 
    "scoring": {
        "metrics": [
        {
            "name": "desc_QED",  # Name of a metric returned from the scoring functions
            "weight": 1,  # Used for some aggregation methods like weighted sum or weighted product
            "modifier": "raw",  # Method for transformation
            "parameters": {}  # Parameters for transformation 
        },
        ],
        # Method to aggregate multiple endpoints
        "method": "single",
    },

    # Diversity filter
    "diversity_filter": {
        "run": true,
        "name": "Unique",  # I.e., penalise non-unique molecules by providing a reward of 0
        "parameters": {}
    }
}
```

## A note on transformation functions

MolScore provides transformation functions to transform the measured metrics $x$ to an output $x'\in [0,1]$. If you use the GUI, a list of the available metrics from the scoring functions will be automatically populated. If a metric is already between 0 and 1, or you don't wish to transform it, you can simply set `"modified": "raw"`. Otherwise, the following functions are available and parameters annotated in the image below (apologies for the handwriting).

Note there should be *no capitilization* of modifier values or parameter keys, and the following modifier names should be used in the JSON:
- Normalize -> norm
- Step threshold -> step
- Linear threshold -> lin_thresh
- Gaussian -> gauss

![alt text](https://github.com/MorganCThomas/MolScore/blob/main/molscore/data/images/transformation_functions.png?raw=True)

## Osimertinib MPO example

Now lets move to another example that include multiple parameters and transformation functions, so we will re-implement Osimertinib_MPO from the GuacaMol benchmark.

```JSON
{
    "task": "Osimertinib_MPO",
    "output_dir": "./",
    "load_from_previous": false,
    "monitor_app": false,
    "scoring_functions": [
    {
        "name": "TanimotoSimilarity",
        "run": true,
        "parameters": {
        "prefix": "Osimertinib_FCFC4",
        "ref_smiles": [
            "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C"
        ],
        "radius": 2,
        "bits": 1024,
        "features": true,
        "counts": true,
        "method": "max",
        "n_jobs": 1
        }
    },
    {
        "name": "TanimotoSimilarity",
        "run": true,
        "parameters": {
        "prefix": "Osimertinib_ECFC6",
        "ref_smiles": [
            "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C"
        ],
        "radius": 3,
        "bits": 1024,
        "features": false,
        "counts": true,
        "method": "max",
        "n_jobs": 1
        }
    },
    {
        "name": "RDKitDescriptors",
        "run": true,
        "parameters": {
        "prefix": "desc",
        "n_jobs": 1
        }
    }
    ],
    "scoring": {
    "method": "gmean",
    "metrics": [
        {
        "name": "Osimertinib_FCFC4_Sim",
        "weight": 1.0,
        "modifier": "lin_thresh",
        "parameters": {
            "objective": "maximize",
            "upper": 0.8,
            "lower": 0.0,
            "buffer": 0.8
        }
        },
        {
        "name": "Osimertinib_ECFC6_Sim",
        "weight": 1.0,
        "modifier": "gauss",
        "parameters": {
            "objective": "minimize",
            "mu": 0.85,
            "sigma": 2.0
        }
        },
        {
        "name": "desc_TPSA",
        "weight": 1.0,
        "modifier": "gauss",
        "parameters": {
            "objective": "maximize",
            "mu": 100.0,
            "sigma": 2.0
        }
        },
        {
        "name": "desc_CLogP",
        "weight": 1.0,
        "modifier": "gauss",
        "parameters": {
            "objective": "minimize",
            "mu": 1.0,
            "sigma": 2.0
        }
        }
    ]
    },
    "diversity_filter": {
    "run": false
    }
}
```