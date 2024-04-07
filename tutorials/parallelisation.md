# Parallelisation of scoring functions
Most scoring functions are implemented with parallelisation over multiple CPUs using pythons multiprocessing by specifying the `n_jobs` parameter. Some more computationally expensive scoring functions such as molecular docking are parallelised using a [Dask](https://www.dask.org/) to allow distributed parallelisation accross compute nodes (`cluster` parameter). Either supply the number of CPUs to utilize on a single compute node to the scheduler address setup via the [Dask CLI](https://docs.dask.org/en/latest/deploying-cli.html). 

To setup a dask cluster first start a scheduler by running (the scheduler address will be printed to the terminal)

    mamba activate molscore
    dask scheduler

Now to start workers accross multiple nodes, simply SSH to a connected node and run

    mamba activate molscore
    dask worker <scheduler_address> --nworkers <n_jobs> --nthreads 1

You can `ssh` into different nodes on your network and repeat this for each node and workers you wish to add to the cluster (ensure the environment and any other dependencies are loaded as you would normally). Then supply in the JSON molscore config files you can add `cluster: <scheduler_address>`.

**Optional**: Sometimes it's annoying to keep editing this parameter in the config file and so environment variables can be set which will override anything provided in the config. To do this, before running MolScore export either of the following variables respectively. 

    export MOLSCORE_NJOBS=<n_jobs>  # Local parallelisation on your compute node
    export MOLSCORE_CLUSTER=<scheduler_address>  # Distributed parallellisation accross multiple compute nodes if set up as above

**Note**: It is recommended to not use more than the number of logical cores available on a particular machine, for example, on a 12-core machine (6 logical cores hyperthreaded) I would not recommend more than 6 workers as it may overload CPU. 