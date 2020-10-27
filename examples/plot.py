import numpy as np
import matplotlib.pyplot as plt
import json

def read_inti(filename: str) -> np.array:
    with open(filename, "r") as f:
        data = json.load(f)

    m = data['matrix']

    return np.reshape([x[0]+x[1]*1j for x in m['data']], m['dim'])

def plot_log_abs(inti: np.array, title: str):
    matrix = 20.0 * np.log10(np.abs(inti))
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    _ = plt.imshow(matrix, aspect='auto')
    ax.set_title("{}, {}".format(title, matrix.shape))
    ax.set_ylabel('Range Sample')
    ax.set_xlabel('Pulse/Channel')
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('[dB]')
    ax = fig.add_subplot(2, 2, 2)
    plt.plot(np.amax(matrix, axis=0))
    ax.grid()
    ax = fig.add_subplot(2, 2, 4)
    plt.plot(np.amax(matrix, axis=1))
    ax.grid()

if __name__ == "__main__":

    FILES = {
        'Input Samples': "input.json",
        'Pulse Compressed': "pulse_compressed.json",
        'Doppler Domain': "range_doppler.json",
    }

    for desc, filename in FILES.items():
        plot_log_abs(read_inti(filename), desc)

    plt.show()