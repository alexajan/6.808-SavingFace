import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import *
import statistics as stat


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
    filename = sys.argv[1]

    rssi = []
    accel = []
    mag = []
    gyro = []
    temp = []

    with open(filename + ".csv", newline='') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:

            if len(row) == 1 or len(row) == 2:  # 1 column format
                if row[0][0] == "R":  # RSSI
                    rssi.append(float(row[1]))

                elif row[0][0] == "A":  # Acceleration
                    acc_i = row[0].find(":")
                    values = row[0].split(",")
                    accel.append([float(values[0][acc_i+3:]), float(values[1]), float(values[2][:-1])])

                elif row[0][0] == "M":  # Magnetometer
                    mag_i = row[0].find(":")
                    values = row[0].split(",")
                    mag.append([float(values[0][mag_i+3:]), float(values[1]), float(values[2][:-1])])

                elif row[0][0] == "G":  # Gyroscope
                    gyro_i = row[0].find(":")
                    values = row[0].split(",")
                    gyro.append([float(values[0][gyro_i+3:]), float(values[1]), float(values[2][:-1])])

                elif row[0][0] == "T":  # Temperature
                    temp_i = row[0].find(":")
                    temp.append(float(row[0][temp_i + 3:-1]))

            else:

                if row[0][0] == "D":  # RSSI
                    rssi_i = row[0].find("RSSI = ")
                    rssi.append(float(row[0][rssi_i+7:]))

                elif row[0][0] == "A":  # Acceleration
                    acc_i = row[0].find(":")
                    accel.append([float(row[0][acc_i+3:]), float(row[1]), float(row[2][:-1])])

                elif row[0][0] == "M":  # Magnetometer
                    mag_i = row[0].find(":")
                    mag.append([float(row[0][mag_i+3:]), float(row[1]), float(row[2][:-1])])

                elif row[0][0] == "G":  # Gyroscope
                    gyro_i = row[0].find(":")
                    gyro.append([float(row[0][gyro_i+3:]), float(row[1]), float(row[2][:-1])])

                elif row[0][0] == "T":  # Temperature
                    temp_i = row[0].find(":")
                    temp.append(float(row[0][temp_i+3:-1]))

        return rssi, accel, mag, gyro


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


def kalman(rssi, dist, rssi_sig, dist_sig):
    mu = 0.6
    sig = 10.
    x = []

    ratio = len(dist)//len(rssi)
    for i in range(len(dist)):
        if i % ratio == 0:
            mu, sig = update(mu, sig, rssi[i % ratio], rssi_sig)
            x.append(mu)
            # print("update: [{}, {}]".format(mu, sig))
        mu, sig = predict(mu, sig, dist[i], dist_sig)
        # print("predict: [{}, {}]".format(mu, sig))
        if mu < 0:
            mu = 0

    print("final: [{}, {}]".format(mu, sig))
    return mu, sig

# 1 accumulate the accel until next sensor measurement v=v+a*t, x=x+v*t
# 2 given accel x,y,z, find vector that points at your face (i.e. the x-axis)
# 3 project accel vector onto face plane
# 4 plug accel as motion into kalman filter
# 5 calculate dist from predicted RSSI


def rssiToDist(rssi):
    n = 2  # propagation in free space
    tx = -55  # kontakt BLE beacon RSSI @ 1m
    return pow(10, (tx - rssi)/(10 * n))


def distToRSSI(dist):
    #dist = 10^[(tx-rssi)/10n]
    n = 2
    tx = -55
    return log10(dist) * -10*n + tx


def project(plane, vector):
    # c is the percentage of the plane vector that you have traversed so far
    c = np.dot(vector, plane)/(magnitude(plane)**2)
    return c


def magnitude(vector):
    return np.linalg.norm(vector)


def check_direction(projection, plane):
    if np.dot(projection, plane) > 0:
        return 1.0
    else:
        return -1.0


def define_plane(vectors):
    # end vector - start vector
    # start vector = avg accel 5-10
    # end vector = avg accel last 5

    start_x = sum([i[0] for i in vectors[:5]])/5
    start_y = sum([i[1] for i in vectors[:5]])/5
    start_z = sum([i[2] for i in vectors[:5]])/5

    end_x = sum([i[0] for i in vectors[-5:]])/5
    end_y = sum([i[1] for i in vectors[-5:]])/5
    end_z = sum([i[2] for i in vectors[-5:]])/5

    return [end_x-start_x, end_y-start_y, end_z-start_z]


def accelToDist(accel, away):
    dist = []

    dt = 0.0025  # 1/400Hz

    v_x = 0
    v_y = 0
    v_z = 0

    x_x = 0
    x_y = 0
    x_z = 0

    for a in accel[5:]:
        v_x += a[0] * dt
        x_x += v_x * dt

        v_y += a[1] * dt
        x_y += v_y * dt

        v_z += a[2] * dt
        x_z += v_z * dt

        dist.append([x_x, x_y, x_z])

    if away:
        plane = [-i for i in define_plane(dist)]
    else:
        plane = define_plane(dist)

    projections = []
    previous_percentage = 0
    for d in dist:
        percentage_moved_total = project(plane, d)
        if percentage_moved_total < 0:
            new_movement_percent = abs(percentage_moved_total) - abs(previous_percentage)  # negative
            projections.append(abs(new_movement_percent) * 0.6096)  # +0.1

        else:
            new_movement_percent = percentage_moved_total-previous_percentage  # positive
            projections.append(new_movement_percent * -0.6096)  # -0.1

        previous_percentage = percentage_moved_total
    print(sum(projections))
    return projections


def main():
    rssi, accel, mag, gyro = parse_data()
    print(rssi, stat.median(rssi[-3:]))
    if stat.median(rssi[-3:]) <= rssi[0]:
        dist = accelToDist(accel, True)
    else:
        dist = accelToDist(accel, False)

    rssi_dist = [rssiToDist(i) for i in rssi]
    rssi_sig = rssiToDist(-90)  # opt+clk wifi for rssi noise
    dist_sig = .00003  # 126ug/sqrt(Hz)^2 * 200Hz > m/s^2 FXOS8700CQ noise

    kalman(rssi_dist, dist, rssi_sig, dist_sig)


if __name__ == "__main__":
    main()
