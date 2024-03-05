These are a re-implementation of the Experiment 3 objectives used in [LinkINVENT](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00115b).

**In order for the linker descriptor scoring functions to work, the linker SMILES must be specified. This is done by passing as an additional format during the score call (which should be the same size and order as SMILES of course)**

```python
ms.score(SMILES, additional_formats={"linker": [str]})
```

Note that currently the output file will be in your current working directory. To modify this change the following line,
```JSON
"output_dir": "./"
```
N.B. Can either be a relative or absolute path.