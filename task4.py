import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy


def make_plot():
    '''
    Function to plot the average FVAF values for each training set size
    Lines for training, validation, and test sets shown
    '''
    # Values used for task 1
    rotations = range(0, 20, 2)
    Ntraining = [1, 2, 4, 6, 9, 13, 18]
    dropout = [0.05, 0.2, 0.35, 0.5, 0.65]
    l2 = [0.001, 0.01, 0.1, 1, 10]

    # Create numpy arrays for each set
    fvafs_validation_dropout = np.empty((len(rotations), len(Ntraining), len(dropout)))
    fvafs_validation_l2 = np.empty((len(rotations), len(Ntraining), len(l2)))

    # Loop over each experiment
    for i, r in enumerate(rotations):
        for j, n in enumerate(Ntraining):
            for k, d in enumerate(dropout):
                # Open experiment results and add them to arrays
                with open(f'results/dropout__ddtheta_1_hidden_500_250_125_75_36_17_JI_rotation_{r}_Ntraining_{n}_dropout_{d}_results.pkl', "rb") as fp:
                    results = pickle.load(fp)
                    fvafs_validation_dropout[i][j][k] = results['predict_validation_fvaf']

            for k, l in enumerate(l2):
                # Open experiment results and add them to arrays
                with open(f'results/l2__ddtheta_1_L2_{l:.6f}_hidden_500_250_125_75_36_17_JI_rotation_{r}_Ntraining_{n}_l2_{l}_results.pkl', "rb") as fp:
                    results = pickle.load(fp)
                    fvafs_validation_l2[i][j][k] = results['predict_validation_fvaf']

    # Compute average FVAF for each training set size for each set
    avg_fvafs_validation_dropout = np.average(fvafs_validation_dropout, axis=0)
    avg_fvafs_validation_l2 = np.average(fvafs_validation_l2, axis=0)

    # Find maximum validation FVAF for each training fold size
    best_avg_fvafs_validation_dropout = np.argmax(avg_fvafs_validation_dropout, axis=1)
    best_avg_fvafs_validation_l2 = np.argmax(avg_fvafs_validation_l2, axis=1)

    # Create numpy arrays for each set
    fvafs_testing_base = np.empty((len(rotations), len(Ntraining)))
    fvafs_testing_dropout = np.empty((len(rotations), len(Ntraining)))
    fvafs_testing_l2 = np.empty((len(rotations), len(Ntraining)))

    # Loop over each experiment
    for i, r in enumerate(rotations):
        for j, n in enumerate(Ntraining):
            with open(f'results/base__ddtheta_1_hidden_500_250_125_75_36_17_JI_rotation_{r}_Ntraining_{n}_results.pkl', "rb") as fp:
                results = pickle.load(fp)
                fvafs_testing_base[i][j] = results['predict_testing_fvaf']

            with open(f'results/dropout__ddtheta_1_hidden_500_250_125_75_36_17_JI_rotation_{r}_Ntraining_{n}_'
                      f'dropout_{dropout[best_avg_fvafs_validation_dropout[j]]}_results.pkl', "rb") as fp:
                results = pickle.load(fp)
                fvafs_testing_dropout[i][j] = results['predict_testing_fvaf']

            with open(f'results/l2__ddtheta_1_L2_{l2[best_avg_fvafs_validation_l2[j]]:.6f}_hidden_500_250_125_75_36_17_JI_'
                      f'rotation_{r}_Ntraining_{n}_l2_{l2[best_avg_fvafs_validation_l2[j]]}_results.pkl', "rb") as fp:
                results = pickle.load(fp)
                fvafs_testing_l2[i][j] = results['predict_testing_fvaf']

    # Compute average FVAF for each training set size for each set
    avg_fvafs_testing_base = np.average(fvafs_testing_base, axis=0)
    avg_fvafs_testing_dropout = np.average(fvafs_testing_dropout, axis=0)
    avg_fvafs_testing_l2 = np.average(fvafs_testing_l2, axis=0)

    # Create line plot
    fig = plt.figure()
    plt.plot(Ntraining, avg_fvafs_testing_base, label='Base')
    plt.plot(Ntraining, avg_fvafs_testing_dropout, label='Dropout')
    plt.plot(Ntraining, avg_fvafs_testing_l2, label='L2')
    plt.ylabel('Testing FVAF')
    plt.xlabel('Training Folds')
    plt.title('Testing FVAF vs Training Folds')
    plt.legend()
    fig.savefig('task4.png')

    # Run t-tests
    print('t-tests for 1 training fold')
    t_base_dropout_1 = scipy.stats.ttest_rel(fvafs_testing_base[:, 0], fvafs_testing_dropout[:, 0])
    print(f'For base and dropout pair\npvalue = {t_base_dropout_1.pvalue}, '
          f'diff = {np.abs(np.mean(fvafs_testing_base[:, 0]) - np.mean(fvafs_testing_dropout[:, 0]))}')
    t_base_l2_1 = scipy.stats.ttest_rel(fvafs_testing_base[:, 0], fvafs_testing_l2[:, 0])
    print(f'For base and L2 pair\npvalue = {t_base_l2_1.pvalue}, '
          f'diff = {np.abs(np.mean(fvafs_testing_base[:, 0]) - np.mean(fvafs_testing_l2[:, 0]))}')
    t_dropout_l2_1 = scipy.stats.ttest_rel(fvafs_testing_dropout[:, 0], fvafs_testing_l2[:, 0])
    print(f'For dropout and L2 pair\npvalue = {t_dropout_l2_1.pvalue}, '
          f'diff = {np.abs(np.mean(fvafs_testing_dropout[:, 0]) - np.mean(fvafs_testing_l2[:, 0]))}')

    print(f'\nt-tests for 18 training folds')
    t_base_dropout_18 = scipy.stats.ttest_rel(fvafs_testing_base[:, 6], fvafs_testing_dropout[:, 6])
    print(f'For base and dropout pair\npvalue = {t_base_dropout_18.pvalue}, '
          f'diff = {np.abs(np.mean(fvafs_testing_base[:, 6]) - np.mean(fvafs_testing_dropout[:, 6]))}')
    t_base_l2_18 = scipy.stats.ttest_rel(fvafs_testing_base[:, 6], fvafs_testing_l2[:, 6])
    print(f'For base and L2 pair\npvalue = {t_base_l2_18.pvalue}, '
          f'diff = {np.abs(np.mean(fvafs_testing_base[:, 6]) - np.mean(fvafs_testing_l2[:, 6]))}')
    t_dropout_l2_18 = scipy.stats.ttest_rel(fvafs_testing_dropout[:, 6], fvafs_testing_l2[:, 6])
    print(f'For dropout and L2 pair\npvalue = {t_dropout_l2_18.pvalue}, '
          f'diff = {np.abs(np.mean(fvafs_testing_dropout[:, 6]) - np.mean(fvafs_testing_l2[:, 6]))}')


if __name__ == '__main__':
    make_plot()
