# Algorithm

- // notation

- $n$ features ${\cal{X}} = \{X_1, \dots, X_n\}$ and the class $Y$

- $m$ instances.

- $D = \{ (x_1^i, \dots, x_n^i, y^i) \}_{i=1}^{m}$

- $W$ a weights vector. $W_0$ are the initial weights.

- $D[W]$ dataset with weights $W$ for the instances.

1. // initialization

2. $W_0 \leftarrow (w_1, \dots, w_m) \leftarrow 1/m$

3. $W \leftarrow W_0$

4. $Vars \leftarrow {\cal{X}}$

5. $\delta \leftarrow 10^{-4}$

6. $convergence \leftarrow True$ // hyperparameter

7. $maxTolerancia \leftarrow 3$ // hyperparameter

8. $bisection \leftarrow False$ // hyperparameter

9. $finished \leftarrow False$

10. $AODE \leftarrow \emptyset$ // the ensemble

11. $tolerance \leftarrow 0$

12. $numModelsInPack \leftarrow 0$

13. $maxAccuracy \leftarrow -1$

14.

15. // main loop

16. While $(\lnot finished)$

    1. $\pi \leftarrow SortFeatures(Vars, criterio, D[W])$

    2. $k \leftarrow 2^{tolerance}$

    3. if ($tolerance == 0$) $numItemsPack \leftarrow0$

    4. $P \leftarrow Head(\pi,k)$ // first k features in order

    5. $spodes \leftarrow \emptyset$

    6. $i \leftarrow 0$

    7. While ($i < size(P)$)

        1. $X \leftarrow P[i]$

        2. $i \leftarrow i + 1$

        3. $numItemsPack \leftarrow numItemsPack + 1$

        4. $Vars.remove(X)$

        5. $spode \leftarrow BuildSpode(X, {\cal{X}}, D[W])$

        6. $\hat{y}[] \leftarrow spode.Predict(D)$

        7. $\epsilon \leftarrow error(\hat{y}[], y[])$

        8. $\alpha \leftarrow \frac{1}{2} ln \left ( \frac{1-\epsilon}{\epsilon} \right )$

        9. if ($\epsilon > 0.5$)

            1. $finished \leftarrow True$

            2. break

        10. $spodes.add( (spode,\alpha_t) )$

        11. $W \leftarrow UpdateWeights(W,\alpha,y[],\hat{y}[])$

    8. $AODE.add( spodes )$

    9. if ($convergence \land \lnot finished$)

        1. $\hat{y}[] \leftarrow AODE.Predict(D)$

        2. $actualAccuracy \leftarrow accuracy(\hat{y}[], y[])$

        3. $if (maxAccuracy == -1)\; maxAccuracy \leftarrow actualAccuracy$

        4. if $((accuracy - maxAccuracy) < \delta)$ // result doesn't
            improve enough

            1. $tolerance \leftarrow tolerance + 1$

        5. else

            1. $tolerance \leftarrow 0$

            2. $numItemsPack \leftarrow 0$

    10. If
        $(Vars == \emptyset \lor tolerance>maxTolerance) \; finished \leftarrow True$

    11. $lastAccuracy \leftarrow max(lastAccuracy, actualAccuracy)$

17. if ($tolerance > maxTolerance$) // algorithm finished because of
    lack of convergence

    1. $removeModels(AODE, numItemsPack)$

18. Return $AODE$
