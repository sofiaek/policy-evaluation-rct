# Off-Policy Evaluation with Out-of-Sample Guarantees
This repository contains code to replicate the experimental results in:

Sofia Ek, Dave Zachariah. "Externally Valid Policy Evaluation from Randomized Trials Using Additional Observational Data". 2024.

## Abstract 
Randomized trials are widely considered as the gold standard for evaluating the effects of decision policies. Trial data 
is, however, drawn from a population which may differ from the intended target population and this raises a problem of 
external validity (aka. generalizability). In this paper we seek to use trial data to draw valid inferences about the 
outcome of a policy on the target population. Additional covariate data from the target population is used to model the 
sampling of individuals in the trial study. We develop a method that yields certifiably valid trial-based policy 
evaluations under any specified range of model miscalibrations. The method is nonparametric and the validity is assured 
even with finite samples. The certified policy evaluations are illustrated using both simulated and real data.

## Results
The files named run_simulations and run_nhanes replicates all the results in the paper.

### NHANES data
The National Health and Nutrition Examination Survey (NHANES) dataset nhanes.fish.rda used in some of the numerical experiments is available here:

Qingyuan Zhao, Dylan S Small, and Paul R Rosenbaum. ["Cross-screening in observational studies that test many hypotheses".](https://github.com/qingyuanzhao/CrossScreening?tab=readme-ov-file) 2018.

The file should be placed in the data folder.

More details of the study can be found [here](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013).

