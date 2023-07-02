# BayesNet

Bayesian Network Classifier with libtorch from scratch

## Variable Elimination

To decide the first variable to eliminate wel use the MinFill criterion, that is
the variable that minimizes the number of edges that need to be added to the
graph to make it triangulated.
This is done by counting the number of edges that need to be added to the graph
if the variable is eliminated. The variable with the minimum number of edges is
chosen.
In pgmpy this is done computing then the length of the combinations of the
neighbors taken 2 by 2.

Once the variable to eliminate is chosen, we need to compute the factors that
need to be multiplied to get the new factor.
This is done by multiplying all the factors that contain the variable to
eliminate and then marginalizing the variable out.

The new factor is then added to the list of factors and the variable to
eliminate is removed from the list of variables.

The process is repeated until there are no more variables to eliminate.

## Code for combination

// Combinations of length 2
vector<string> combinations(vector<string> source)
{
    vector<string> result;
    for (int i = 0; i < source.size(); ++i) {
        string temp = source[i];
        for (int j = i + 1; j < source.size(); ++j) {
            result.push_back(temp + source[j]);
        }
    }
    return result;
}
