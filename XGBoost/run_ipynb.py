import nbformat
from nbconvert.preprocessors import ExecutePreprocessor



notebook_filename = '/root/autodl-tmp/xiaolong/WorkSpace/Amazon-KDDCUP-23/XGBoost/pynb_test.ipynb'
with open(notebook_filename) as f:
    nb = nbformat.read(f, as_version=4)


ep = ExecutePreprocessor(kernel_name='torch12')

ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})

with open('executed_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

