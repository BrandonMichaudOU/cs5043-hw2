import matplotlib.pyplot as plt
import numpy as np
import pickle


def make_plot():
    '''
    Function to plot the average FVAF values for each training set size
    Lines for training, validation, and test sets shown
    '''
    # Values used for task 1
    rotations = range(0, 20, 2)
    Ntraining = [1, 2, 4, 6, 9, 13, 18]
    dropout = [0.05, 0.2, 0.35, 0.5, 0.65]

    # Create numpy arrays for each set
    fvafs_validation = np.empty((len(rotations), len(Ntraining), len(dropout)))

    # Loop over each experiment
    for i, r in enumerate(rotations):
        for j, n in enumerate(Ntraining):
            for k, d in enumerate(dropout):
                # Open experiment results and add them to arrays
                with open(f'results/dropout__ddtheta_1_hidden_500_250_125_75_36_17_JI_rotation_{r}_Ntraining_{n}_dropout_{d}_results.pkl', "rb") as fp:
                    results = pickle.load(fp)
                    fvafs_validation[i][j][k] = results['predict_validation_fvaf']

    # Compute average FVAF for each training set size for each set
    avg_fvafs_validation = np.average(fvafs_validation, axis=0)
    print(np.shape(avg_fvafs_validation))

    # Create line plot
    fig = plt.figure()
    for i, d in enumerate(dropout):
        plt.plot(Ntraining, avg_fvafs_validation[:, i], label=f'{d}')
    plt.ylabel('Validation FVAF')
    plt.xlabel('Training Folds')
    plt.title('Validation FVAF vs Training Folds')
    plt.legend(title='Dropout')
    fig.savefig('task2.png')


if __name__ == '__main__':
    make_plot()
