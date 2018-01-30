# Title: Hitting the target: Mathematical attainment in children is related to interceptive timing ability

## Authors: Oscar T. Giles, Katy A. Shire, Liam J.B. Hill, Faisal Mushtaq, Amanda Waterman, Raymond J. Holt, Peter R. Culmer, Justin H. G. Williams, Richard M. Wilkie, Mark Mon-Williams

### Email: o.t.giles@leeds.ac.uk


#### Overview:

This directory contains all the files and data required to reproduce the analysis in "Hitting the Target: Mathematical attainment in children is related to interceptive timing ability".

#### Raw_data:

The raw data is provided in the '//Raw_data//' subdirectory in the file "master_concat_data.csv". Each row is a participant and each column is a variable. A brief decriptiuon of each variable is given below

participant: A participant number
interception: The interceptive score metric (IntT)
Open: Balance task with eyes open 
Closed: Balance task with eyes closed
age: The participant's age
Ckat_tracking: The Tracking task score
Ckat_aiming: The Aiming score
Ckat_tracing: The Steering task (names tracing here, renamed "Steering" in the manuscript)
Attainment_Maths: Mathematics attainment
Attainment_Reading: Reading attainment
Attainment_Writing: Writing attainment

#### Fit the Stan models:

Four R files are containted in the parent directory as well as two .stan files which provide the code for the statistical models. Running each of these R files will fit the statistical models and populate the "//MCMC_samples//"" subdirectory with the results of the Bayesian model fitting. These samples are then used by the scripts in the "//Plotting//" subdirectory to produce the figures found in the manuscript. Running the files will require R and RStan (http://mc-stan.org/users/interfaces/rstan).

#### Analyses and figure plotting:

All plotting and further analysis fo the MCMC samples was conducted using Python 3. The easiest way to ensure you have all the python libraries used by the analysis is to install the Anaconda distribution of python (https://www.anaconda.com/).

The "//Plotting//" subdirectory contains python dscvripts which will recreate the plots in the manuscript. There is a seperate python file for each figure. You need to make sure that all the .R files in the parent directory have already been run (this may take several hours) in order to generate the MCMC sample files used for the figures. 
