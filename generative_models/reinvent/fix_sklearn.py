import sklearn
import os

root = os.path.dirname(sklearn.__file__)
print(f"sklearn source {root}")

for source_file in [os.path.join(root, 'model_selection/_split.py'),
                    os.path.join(root, 'metrics/cluster/supervised.py')]:
    print(f"Reading {source_file}")
    with open(source_file, 'r') as f:
        text = f.read()
    print(f"Replacing scipy.misc with scipy.special")
    with open(source_file, 'w') as f:
        f.write(text.replace("from scipy.misc", "from scipy.special"))


