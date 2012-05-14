This is a C++1X-Implementation of a Graph-Edit-Algorithm, namely the Kullback-Leibler-Divergence between two graphs via their Hidden Markov Model.

Requirements: 

Armadillo: An Open Source C++ Linear Algebra Library for Fast Prototyping and Computationally Intensive Experiments. 

and its requirements, 

ATLAS or BLAS


Features:

It can be used as a FeatureTemplate in GraphDB via the normal installation procedure, offering two templates:
- Hidden Markov Model, which creates a model to insert into feature_hmm_lib_values.
- Delta HMM, which calculates the similarity between the gid the feature should be tied to and a set of gids the user enters.

Notes: 
-While Delta_HMM normally uses the saved HMMs from the aforementioned table, when they don't exist already they will get calculated on the fly with a clusternumber of "3", though not saved to the table.
-There's an experimental feature to use ChineseWhisper to cluster the graphs (therefore using edge data), it is activated by using a clusternumber below zero when creating them, the entered number will be interpreted as the amount of passes the algorithm does.
Note that it may lead to single-node clusters which breaks the GMM building algorithm.


