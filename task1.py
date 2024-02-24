import matplotlib.pyplot as plt
import numpy as np
import pickle


def make_plot():
    '''
    Function to plot the average FVAF values for each training set size
    Lines for training, validation, and test sets shown
    '''
    # Values used for task 1
    rotations = list(range(0, 20, 2))
    Ntraining = [1, 2, 4, 6, 9, 13, 18]

    # Create numpy arrays for each set
    fvafs_training = np.empty((len(rotations), len(Ntraining)))
    fvafs_validation = np.empty((len(rotations), len(Ntraining)))

    # Loop over each experiment
    for i, r in rotations:
        for j, n in enumerate(Ntraining):
            # Open experiment results and add them to arrays
            with open(f'base__ddtheta_1_hidden_500_250_125_75_36_17_JI_rotation_{r}_Ntraining_{n}_results.pkl', "rb") as fp:
                results = pickle.load(fp)
                fvafs_training[i][j] = results['predict_training_fvaf']
                fvafs_validation[i][j] = results['predict_validation_fvaf']

    # Compute average FVAF for each training set size for each set
    avg_fvafs_training = np.average(fvafs_training, axis=0)
    avg_fvafs_validation = np.average(fvafs_validation, axis=0)

    # Create line plot
    fig = plt.figure()
    plt.plot(Ntraining, avg_fvafs_training, label='training')
    plt.plot(Ntraining, avg_fvafs_validation, label='validation')
    plt.ylabel('FVAF')
    plt.xlabel('Training Folds')
    plt.title('FVAF vs Training Folds')
    plt.legend()
    fig.savefig('task1.png')


if __name__ == '__main__':
    make_plot()
