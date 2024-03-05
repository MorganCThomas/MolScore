These are a re-implementation of the Experiment 1 objectives used in [LibINVENT](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00469).

Note that currently the output file will be in your current working directory. To modify this change the following line,
```JSON
"output_dir": "./"
```
N.B. Can either be a relative or absolute path.

Note these scoring functions automatically try to extract a substructure from the provided molecule, if this substructure does not exist (and there for reactions cannot be mapped), a score of 0.0 is returned.