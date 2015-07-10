# Instance Specific Scheduling with FlexFolio

## Authors

  * David Bergdoll <bergdolr@tf.uni-freiburg.de>, Univerity of Freiburg
  * Marius Lindauer <lindauer@cs.uni-freiburg.de>, University of Freiburg
  * Frank Hutter <fh@cs.uni-freiburg.de>, University of Freiburg

## System Description

### Overall Workflow

  * setup a predefined set of configurations of the scheduling approaches SUNNY-like, aspeed and instance-specifc aspeed,
   which are described later on
  * evaluate the performance of those approaches on training data (using cross-validation)
  * choose the candidate with the best par10 score to be run on the current scenario and train it if necessary
  * employ the chosen portfolio on the current test set 

### SUNNY-like

SUNNY-like is a adaption of the solver SUNNY[1] for FlexFolio, the successor of claspfolio2 [2]. The algorithm does not rely on offline 
training but computes a schedule of solvers online by the following steps:

1. A set of k nearest neighbors of benchmark instances is selected, whereby the distance of instances is defined as the
 euclidian distance of their feature vectors.
2. Time slots of equal size are assigned to candidate solvers proportional to the number of neighborhood instances they
 solve.
3. For each neighborhood instance that is not solved by any solver a time slot is assigned to a backup solver, in this
 case the single best solver for this neighborhood.
4. The solvers are sorted, in decreasing order, by the number of neighborhood instances they solve.

### aspeed

aspeed [3] is a scheduling portfolio which constructs a static solver schedule offline using benchmark data. The scheduling
mechanism is encoded as an Answer Set Programing optimization problem, the schedule is therefore obtained by solving 
this problem as follows:

1. The benchmark data is rewritten as a set of ASP facts.
2. The ASP solver clasp is applied to those facts together with aspeed's encoding, assigning the available time to the
 candidate solvers in a way that minimizes the total number of timeouts on the training data.
3. As a second optimization criterion, this time assignment is aligned, minimizing the total runtime.

### instance-specific aspeed

The approach of instance-specific aspeed is inspired by a scheduling enhancement proposed for the solver 3S by ... et al ...[4]
Unlike the original aspeed, this approach builds an individual schedule for each instance:
 
1. A set of k nearest neighbors of benchmark instances is selected, whereby the distance of instances is defined as the
 euclidian distance of their feature vectors.
2. A solver schedule is computed by aspeed using the benchmark data of the those neighborhood instances as input.

As for SUNNY, k is set to the square root of the number of training instances.


## Literature

[1] R. Amadini and M. Gabbrielli and J. Mauro
SUNNY: a Lazy Portfolio Approach for Constraint Solving
In: Theory and Practice of Logic Programming 14 (2014). pages 509-524

[2] H. Hoos and M. Lindauer and T. Schaub
claspfolio 2: Advances in Algorithm Selection for Answer Set Programming
In: Theory and Practice of Logic Programming 14 (2014). pages 569-585

[3] H. Hoos and R. Kaminski and M. Lindauer and T. Schaub
aspeed: Solver Scheduling via Answer Set Programming
In: Theory and Practice of Logic Programming 15 (2015). pages 117-142

[4] S. Kadioglu and Y. Malitsky and A. Sabharwal and H. Samulowitz and M. Sellmann
Algorithm Selection and Scheduling. 
In: CP. Lecture Notes in Computer Science, vol. 6876. Springer.

## Pre-Solvers

No pre-solvers are used for this approach.

## Feature Groups

...

## Installation

### FlexFolio

  * run ```selectors/flexfolio/python_env.sh``` to install a virtualenv with the required packages of flexfolio
  * set an alias of Python to selectors/flexfolio/virtualenv/bin/python (```alias python=selectors/flexfolio/virtualenv/bin/python```)
  * test the binaries in selectors/flexfolio/binaries; recompile them if they are broken
  * test runsolver binary in runsolver/; recompile it if it is broken

## Call 

### Training

`cd <root-folder>`  
`mkdir <training-output-directory>`  
`python ./flexfolio-train.py --aslib-dir <ASlib-scenario> --model <training-output-directory --approach SchedulerTrainer --train`

### Testing

`cd <root-folder>`
`python ./flexfolio-run-aschallenge.py --aslib-dir <ASlib-scenario> --configfile <training-output-directory>/config.json --output <testing-output-file>`
