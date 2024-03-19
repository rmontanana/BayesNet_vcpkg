1. // initialization

2. $W_0 \leftarrow (w_1, \dots, w_m) \leftarrow 1/m$

3. $W \leftarrow W_0$

4. $Vars \leftarrow {\cal{X}}$

5. $\delta \leftarrow 10^{-4}$

6. $convergence \leftarrow True$

7. $maxTolerancia \leftarrow 3$

8. $bisection \leftarrow False$

9. $error \leftarrow \inf$

10. $finished \leftarrow False$

11. $AODE \leftarrow \emptyset$ // the ensemble

12. $tolerance \leftarrow 0$

13. $numModelsInPack \leftarrow 0$

15. // main loop

16. While (!finished)

    1. $\pi \leftarrow SortFeatures(Vars, criterio, D[W])$

    2. $k \leftarrow 2^{tolerance}$

    3. if ($tolerance == 0$)
        $numItemsPack \leftarrow0$

    4. $P \leftarrow Head(\pi,k)$ // first k features in order

    6. $i \leftarrow 0$

    7. While ($i < size(P)$)

        1. $X \leftarrow P[i]$

        2. $i \leftarrow i + 1$

        3. $numItemsPack \leftarrow numItemsPack + 1$

        4. $Vars.remove(X)$

        5. $spode \leftarrow BuildSpode(X, {\cal{X}}, D[W])$

        6. $\hat{y}[] \leftarrow spode.Predict(D[W])$

        7. $\epsilon \leftarrow error(\hat{y}[], y[])$

        8. $\alpha \leftarrow \frac{1}{2} ln \left ( \frac{1-\epsilon}{\epsilon} \right )$

        9. if ($\epsilon > 0.5$)

            1. $finished \leftarrow True$

            2. break

        10. $AODE.add( (spode,\alpha_t) )$

        11. $W \leftarrow UpdateWeights(D[W],\alpha,y[],\hat{y}[])$

    8. if ($convergence$ $\And$ $! finished$)

        1. $\hat{y}[] \leftarrow AODE.Predict(D[W])$

        2. $e \leftarrow error(\hat{y}[], y[])$

        3. if $(e > (error+\delta))$ // result doesn't improve

            1. if $(tolerance == maxTolerance)\; finished\leftarrow True$

            2. else $tolerance \leftarrow tolerance+1$

        4. else

            1. $tolerance \leftarrow 0$

            2. $error \leftarrow min(error,e)$

    9. if $(Vars == \emptyset) \; finished \leftarrow True$

17. if ($tolerance == maxTolerance$) // algorithm finished because of
    lack of convergence

    1. $removeModels(AODE, numItemsPack)$

    2. $W \leftarrow W_B$

18. Return $AODE$
