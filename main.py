#### ==========================================================================
#### Orthogonal Procrustes and Machine Learning: Predicting errors on time.
#### Code repository submitted to Elsevier.
#### --------------------------------------------------------------------------
"""
This repository includes the complete code for the paper "Orthogonal Procrustes
and Machine Learning: Predicting errors on time". It is handed in to Elsevier
for publishing. The authors are prepared to publish their code on GitHub, too.
"""
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
import time
import sys
from constants import FP_DATA
from orthogonal_procrustes import solve_orthogonal_procrustes
from chi_squared_test import chi_squared_test
from clustering import cluster
from isolation_forest import isolation_forest
from multi_output_mlp import multi_output_mlp
# blockPrint()
start_time = time.time()
print('================================================================= \n' +
      'Pipeline \n' + sys.argv[0] + '\n' + 'started at: ' +
      time.strftime('%H:%M:%S', time.gmtime(start_time)) + '. \n' +
      '-----------------------------------------------------------------')
### --------------------------------------------------------------------------
### Main.
### --------------------------------------------------------------------------
# The main pipeline as described in figure 6 of the paper.
def main():
    '''
    The first step is to solve the Orthogonal Procrustes problem for multi-level
    Bills of Material.
    '''
    solve_orthogonal_procrustes()
    '''
    Some feature engineering, which is actually not required using the sample
    dataset but mentioned for completeness. Note that benchmark outlier detection
    algorithms are implemented in the file "benchmark_outlier_detection".
    '''
    chi_squared_test()
    '''
    The contextualization using k-means is the next step. In this variant, the
    cluster attribute is given to the multi-output mlp as another feature. An 
    alternative would be to deploy clustering later and train a mlp on each
    cluster seperately. Note that benchmark clustering
    algorithms are implemented in the file "benchmark_clustering".
    '''
    cluster()
    '''
    Next, the anomaliy detection based on the isolation forest algorithm is
    implemented. Note that there are benchmark algorithms implemented in the 
    file "benchmark_anomaly_detection.py".
    '''
    isolation_forest()
    '''
    Eventually, the multi-output mlp is called, predicting error instances
    as well as the respective timestamp for a potential BOM error. Note that 
    there are benchmark algorithms implemented in the file 
    "benchmark_error_prediction.py".
    '''
    multi_output_mlp()

# Call main.
if __name__ == '__main__':
    main()
### --------------------------------------------------------------------------
### Postscript.
### --------------------------------------------------------------------------
# enablePrint()
elapsed_time = time.time() - start_time
print('---------------------------------------------------------------- \n' +
      'Elapsed time: ' +
      time.strftime('%H:%M:%S', time.gmtime(elapsed_time)) +
      '. \n' +
      '================================================================')
### ---------------------------------------------------------------------------
### End
#### ==========================================================================