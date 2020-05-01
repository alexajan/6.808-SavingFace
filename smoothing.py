import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import *

filename = sys.argv[1]


def plot_data(accel, mag, gyro, filename):
    # yellow = x
    # green = y
    # blue = z

    x = [i[0] for i in accel]
    y = [j[1] for j in accel]
    z = [k[2] for k in accel]

    plt.plot(x)
    plt.plot(y)
    plt.plot(z)
    plt.savefig(filename + "_accel.png")
    plt.show()

    x_m = [i[0] for i in mag]
    y_m = [j[1] for j in mag]
    z_m = [k[2] for k in mag]

    plt.plot(x_m)
    plt.plot(y_m)
    plt.plot(z_m)
    plt.savefig(filename + "_mag.png")
    plt.show()

    x_g = [i[0] for i in gyro]
    y_g = [j[1] for j in gyro]
    z_g = [k[2] for k in gyro]

    plt.plot(x_g)
    plt.plot(y_g)
    plt.plot(z_g)
    plt.savefig(filename + "_gyro.png")
    plt.show()


def parse_data():

    rssi = []
    accel = []
    mag = []
    gyro = []
    temp = []

    with open(filename + ".csv", newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:

            if row[0][0] == "D": # RSSI
                rssi_i = row[0].find("RSSI = ")
                rssi.append(float(row[0][rssi_i+7:]))

            elif row[0][0] == "A": # Acceleration
                acc_i = row[0].find(":")
                accel.append([float(row[0][acc_i+3:]), float(row[1]), float(row[2][:-1])])

            elif row[0][0] == "M": # Magnetometer
                mag_i = row[0].find(":")
                mag.append([float(row[0][mag_i+3:]), float(row[1]), float(row[2][:-1])])

            elif row[0][0] == "G": # Gyroscope
                gyro_i = row[0].find(":")
                gyro.append([float(row[0][gyro_i+3:]), float(row[1]), float(row[2][:-1])])

            elif row[0][0] == "T": # Temperature
                temp_i = row[0].find(":")
                temp.append(float(row[0][temp_i+3:-1]))

        return rssi, accel, mag, gyro


def f(mu, sigma2, x):
    """Gaussian function, mu = mean, sigma2 = squared variance, x = input, returns gaussian value"""
    coefficient = 1.0 / sqrt(2.0 * pi * sigma2)
    exponential = exp(-0.5 * (x - mu) ** 2 / sigma2)
    return coefficient * exponential


def update(mean1, var1, mean2, var2):
    """Updates Gaussian parameters"""
    new_mean = (var2 * mean1 + var1 * mean2) / (var2 + var1)
    new_var = 1 / (1 / var2 + 1 / var1)
    return [new_mean, new_var]


def predict(mean1, var1, mean2, var2):
    """Returns updated Gaussian parameters after motion"""
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]


def kalman(rssi, motions, rssi_sig, motion_sig):
    mu = 0.
    sig = 10000.
    x = []

    for i in range(len(rssi)):
        mu, sig = update(mu, sig, rssi[i], rssi_sig)
        x.append(mu)
        print("update: [{}, {}]".format(mu, sig))
        mu, sig = predict(mu, sig, motions[i], motion_sig)
        print("predict: [{}, {}]".format(mu, sig))

    print("final: [{}, {}]".format(mu, sig))
    plt.plot(x)
    plt.savefig(filename + "_kalman.png")
    plt.show()
    return mu, sig


def main():
    rssi, accel, mag, gyro = parse_data()
    motions = [1.5 for i in range(len(rssi))]
    rssi_sig = 90 # opt+clk wifi for rssi noise
    motion_sig = 0.001
    kalman(rssi, motions, rssi_sig, motion_sig)

if __name__ == "__main__":
    main()