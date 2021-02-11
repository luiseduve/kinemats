import pandas as pd
import numpy as np


def cluster_aggregation(clustering_results):
    """
    Applies clustering aggregation to the results of many
    clustering algorithms over a set of observations

    :param clustering_results: DataFrame with cluster results in columns, 
                        with observations as rows
    :rtype clustering_results: Pandas DataFrame
    :return: Pandas Series with the result of clustering aggregation
    """
    data = clustering_results
    rows,cols = data.shape

    # Stores the disagreement distance between
    # clustering algorithms, and pair-wise objects
    I = np.zeros((cols,cols, rows, rows))
    D = np.zeros((cols,cols))

    # Iterate each pair of clustering algorithm
    for C in range(cols):
        for P in range(C, cols):
            # Comparing the same two clusters,
            #  disagreement will be zero
            if (C == P):
                I[C,P,:,:] = 0
                I[P,C,:,:] = 0
                continue
            else:

                # Iterate over pairs of rows
                for i in range(rows):
                    for j in range(i, rows):
                        #print(C,P,i,j)
                        if  ( (data.iloc[i,C] == data.iloc[j,C]) 
                                and (data.iloc[i,P] != data.iloc[j,P]) ) or \
                            ( (data.iloc[i,C] != data.iloc[j,C]) 
                                and (data.iloc[i,P] == data.iloc[j,P]) ):
                            I[C,P,i,j] = 1
                        else:
                            I[C,P,i,j] = 0

                        # Symmetric values 
                        I[C,P,j,i] = I[C,P,i,j] # Opposite rows are the same distance
                        I[P,C,i,j] = I[C,P,i,j] # Opposite clusters are the same
                        I[P,C,j,i] = I[C,P,i,j] # Opposite clusters and opposite row

            ## Disagreement distance
            D[C,P] = np.sum(I[C,P,:,:])
            D[P,C] = D[C,P]
    
    ## Given m clustering C1,C2,...,Cm; find C such that:
    #       D(C) = sum_m[ D(C,Ci) ] is minimized
    result = pd.DataFrame(data = D, index = data.columns, columns=data.columns)
    print(result)

    print(result.sum(axis=0))

    idx_best_cluster = result.sum(axis=0).argmin()
    print("\nCluster Min Distance:", \
            result.columns[idx_best_cluster],\
            "with distance", result.sum(axis=0).min()
            )
    
    print("\nFinal array")
    best_cluster = pd.Series(data = data.iloc[:,idx_best_cluster], name = str("BEST_CLUST="+str(idx_best_cluster)))
    print(data.join(best_cluster))

    return result


if __name__ == "__main__":    
    clusters = 3
    observations = 10
    algorithms = 4

    clust_labels = [str("ClustAlg_"+str(i+1)) for i in range(algorithms)]
    clustering_results = clusters * np.random.rand(observations, algorithms)

    data = pd.DataFrame(data=clustering_results, columns=clust_labels).astype(int)
    print(data)
    print("\n\n")
    cluster_aggregation(data)