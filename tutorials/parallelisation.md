# Parallelisation of scoring functions

Most scoring functions are implemented with parallelisation over multiple CPUs using pythons multiprocessing by specifying the `n_jobs` parameter.

## Parallelisation of Docking/Ligand preparation via Dask

Some more computationally expensive scoring functions such as molecular docking are parallelised using [Dask](https://www.dask.org/) to allow distributed parallelisation accross compute nodes (`cluster` parameter). Either supply the number of CPUs to utilize on a single compute node or provide the scheduler address that is found by setting up via the [Dask CLI](https://docs.dask.org/en/latest/deploying-cli.html).

### Using a local cluster
Over a single compute node that you are running MolScore, this can be specified by simply providing the number of cores/workers to use (similar to `n_jobs`) to the `cluster` parameter.
For example, if we want to run rDock over 10 CPUs.

In the GUI this would look like:

![alt text](https://github.com/MorganCThomas/MolScore/blob/main/molscore/data/images/dask_example1.png?raw=True)

Or in the configuration JSON:

```JSON
  "parameters": {
    ...
    "cluster": 12
    ...
  }
```

### Using a distributed cluster
Over more than one compute node, we need to setup a dask cluster by starting a scheduler.

    mamba activate molscore
    dask scheduler

After running this command, a scheduler address will be printed to the terminal that will look like this `tcp://0.0.0.0:8000`. 

Now we have the scheduler address, we need to start workers (think CPUs) that are connected to the scheduler.

    mamba activate molscore
    dask worker <scheduler_address> --nworkers <n_jobs> --nthreads 1

You can `ssh` into different nodes on your network and repeat the `dask worker` command for each node and the number of workers you wish to add to the cluster from each node (ensure the environment and any other dependencies are loaded as you would normally). For example, if we have 6 compute nodes, each with 6 workers, docking will be parallelised over all 36 workers, simple! **Note**: It is recommended to not use more than the number of logical cores available on a particular machine, for example, on a 12-core machine (6 logical cores hyperthreaded) I would not recommend more than 6 workers as it may overload CPU. 

Once we have a dask cluster running, then we just supply the scheduler address to MolScore via a configuraiton file. 

In the GUI this would look like:

![alt text](https://github.com/MorganCThomas/MolScore/blob/main/molscore/data/images/dask_example2.png?raw=True)

Or in the configuration JSON:

```JSON
  "parameters": {
    ...
    "cluster": "tcp://0.0.0.0:8000"
    ...
  }
```

### Specifying parallelisation at runtime
Sometimes it's annoying to keep editing this parameter in the configuration file and so environment variables can be set which will override anything provided in the configuration file. To do this, before running MolScore export either of the following variables respectively. 

    export MOLSCORE_NJOBS=<n_jobs>  # Local parallelisation on your compute node e.g., 12
    export MOLSCORE_CLUSTER=<scheduler_address>  # Distributed parallellisation accross multiple compute nodes if set up as above e.g., tcp://0.0.0.0:8000

