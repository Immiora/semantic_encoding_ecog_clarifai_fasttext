import numpy as np

class Crossvalidator:
    # currently works only for arrays, along 0 axis
    # save indices to self

    def __init__(self, k=5, make_val=False, shuffle=False):
        self.nfolds = k
        self.make_val = make_val
        self.shuffle = shuffle
        self.nblocks = 13

    def _reshape_folds(self, x):
        x = np.concatenate([x, np.zeros((1, x.shape[1]))])
        return list(x.reshape(self.nblocks, 750, x.shape[1]))

    def _make_indices(self):
        a = np.tile(np.arange(self.nblocks), [self.nfolds, 1])
        b = np.tile(np.arange(self.nfolds), [self.nblocks, 1]).T
        b = np.apply_along_axis(np.random.permutation, 0, b)
        print('Getting a, b')

        while not np.all(np.array([len(np.unique(a[:, i])) for i in range(self.nblocks)]) == self.nfolds):
            for i in range(self.nfolds):
                perm = np.random.permutation(range(self.nblocks))
                a[i, :] = a[i, perm]
                b[i, :] = b[i, perm]
        print('Done with a, b')
        return a, b

    def __call__(self, x, t):
        ix, it = self._reshape_folds(np.arange(x.shape[0])[:, None]), self._reshape_folds(np.arange(t.shape[0])[:, None])
        x, t = self._reshape_folds(x), self._reshape_folds(t)
        a, b = self._make_indices()

        time_in_fold = 750 / self.nfolds
        Train, Test = [], []
        indTrain, indTest = [], []

        for ifold in range(self.nfolds):

            Test.append([[], []])
            Train.append([x[:], t[:]])
            indTest.append([[], []])
            indTrain.append([ix[:], it[:]])

            for iblock in range(self.nblocks):
                indTest[ifold][0].append(ix[a[ifold, iblock]]
                                [time_in_fold * b[ifold, iblock]:time_in_fold * b[ifold, iblock] + time_in_fold])
                indTest[ifold][1].append(it[a[ifold, iblock]]
                                [time_in_fold * b[ifold, iblock]:time_in_fold * b[ifold, iblock] + time_in_fold])

                Test[ifold][0].append(x[a[ifold, iblock]]
                                [time_in_fold * b[ifold, iblock]:time_in_fold * b[ifold, iblock] + time_in_fold, :])
                Test[ifold][1].append(t[a[ifold, iblock]]
                                [time_in_fold * b[ifold, iblock]:time_in_fold * b[ifold, iblock] + time_in_fold, :])

                indTrain[ifold][0][a[ifold, iblock]] = np.delete(indTrain[ifold][0][a[ifold, iblock]],
                          [range(time_in_fold * b[ifold, iblock],
                                 time_in_fold * b[ifold, iblock] + time_in_fold)], 0)
                indTrain[ifold][1][a[ifold, iblock]] = np.delete(indTrain[ifold][1][a[ifold, iblock]],
                          [range(time_in_fold * b[ifold, iblock],
                                 time_in_fold * b[ifold, iblock] + time_in_fold)], 0)

                Train[ifold][0][a[ifold, iblock]] = np.delete(Train[ifold][0][a[ifold, iblock]],
                          [range(time_in_fold * b[ifold, iblock],
                                 time_in_fold * b[ifold, iblock] + time_in_fold)], 0)
                Train[ifold][1][a[ifold, iblock]] = np.delete(Train[ifold][1][a[ifold, iblock]],
                          [range(time_in_fold * b[ifold, iblock],
                                time_in_fold * b[ifold, iblock] + time_in_fold)], 0)

            indTest[ifold] = [np.concatenate(indTest[ifold][i], 0) for i in range(2)]
            indTrain[ifold] = [np.concatenate(indTrain[ifold][i], 0) for i in range(2)]

            Test[ifold] = [np.concatenate(Test[ifold][i], 0).astype(np.float32) for i in range(2)]
            Train[ifold] = [np.concatenate(Train[ifold][i], 0).astype(np.float32) for i in range(2)]

            #indTest[ifold] = [indTest[ifold][0][~np.all(Test[ifold][0] == 0, axis=1)],
            #                indTest[ifold][1][~np.all(Test[ifold][1] == 0, axis=1)]]
            #indTrain[ifold] = [indTrain[ifold][0][~np.all(Train[ifold][0] == 0, axis=1)],
            #                indTrain[ifold][1][~np.all(Train[ifold][1] == 0, axis=1)]]
            #Test[ifold] = [Test[ifold][0][~np.all(Test[ifold][0] == 0, axis=1)],
            #                Test[ifold][1][~np.all(Test[ifold][1] == 0, axis=1)]]
            #Train[ifold] = [Train[ifold][0][~np.all(Train[ifold][0] == 0, axis=1)],
            #                Train[ifold][1][~np.all(Train[ifold][1] == 0, axis=1)]]

        return Train, Test, indTrain, indTest



