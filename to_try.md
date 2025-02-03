## ESGVI Debugging Notes
#### Current Problems
- Slow backtracking, and backtracking never finds suitable step size. 
- Conditioning of new_information sucks.
- 
#### Things to try
- Using scipy.sparse matrices
- Split up factored_state_list into intervals of 1 sec.
- Better regularization.
- Initialize from MAP solution. 
- Accept lowest backtracking step.