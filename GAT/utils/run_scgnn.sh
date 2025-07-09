
def run_scGNN(file_name):

"""creates graph representation of features

Parameters
----------
file_name : str
    The filename of the features that need to be processed

Returns
-------
none
    a graph.csv is created in GAT/data/features
"""


python3 -W ignore ../scGNN/PreprocessingscGNN.py \
    --datasetName train_features.csv \
    --datasetDir ../data/features/ \
    --LTMGDir ../data/features/ \
    --filetype CSV \
    --geneSelectnum 2000 \
    --inferLTMGTag

python3 -W ignore ../scGNN/scGNN.py \
    --datasetName features \
    --datasetDir ../data/ \
    --LTMGDir ../data/ \
    --outputDir ../data/features/ \
    --EM-iteration 2 \
    --Regu-epochs 50 \
    --EM-epochs 20 \
    --regulized-type LTMG \
    --args.useBothembedding \
    --args.converge_type "graph"

