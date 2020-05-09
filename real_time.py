from bluepy.btle import Scanner
import csv
import sys
import matplotlib.pyplot as plt
from math import *
import statistics as stat
import RPi.GPIO as GPIO
import time
import board
import busio
import adafruit_lsm9ds1
import numpy as np
from threading import Thread, Lock
import queue

rssi_available = False

rssi_size = 15
data_per_rssi = 75

num_face_touches = 0
count = 0

# DATA BUFFERS, SHOULD BE PROTECTED BY LOCKS
rssi_vals = queue.Queue(maxsize=rssi_size)
accel_data = queue.Queue(maxsize=rssi_size*data_per_rssi)

accel_data_size = rssi_size * data_per_rssi

# DATA LOCKS
RSSI_LOCK = Lock()
OTHER_LOCK = Lock()


def setup_i2c():
    # I2C connection:
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_lsm9ds1.LSM9DS1_I2C(i2c)


def get_accel_mag_gyro_temp_vals():
    # Main loop will read the acceleration

    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_lsm9ds1.LSM9DS1_I2C(i2c)
    # connection = True
    while True:
        # Read acceleration
        accel_x, accel_y, accel_z = sensor.acceleration

        OTHER_LOCK.acquire()
        readings = [accel_x, accel_y, accel_z]

        if accel_data.full():
            accel_data.get()
        accel_data.put(readings)

        OTHER_LOCK.release()


def scan_rssi_vals():
    # i2c = busio.I2C(board.SCL, board.SDA)
    # sensor = adafruit_lsm9ds1.LSM9DS1_I2C(i2c)
    global count
    for i in range(1000):
        scanner = Scanner()
        devices = scanner.scan(0.5)

        for device in devices:
            if device.addr == "88:c6:26:8e:c9:cd":
                rssi_available = True
                print("DEV = {} RSSI = {}".format(device.addr, device.rssi))
                RSSI_LOCK.acquire()
                OTHER_LOCK.acquire()
                global rssi_vals
                global num_face_touches
                #rssi_vals = rssi_vals[1:] + [device.rssi]
                if rssi_vals.full():
                    rssi_vals.get()
                rssi_vals.put(device.rssi)

                # Wait for other thread to fill in values
                while not accel_data.full():
                    OTHER_LOCK.release()
                    time.sleep(0.1)
                    OTHER_LOCK.acquire()

                if rssi_vals.full():
                    touch = main_kalman()
                    count += 1
                    if touch:
                        num_face_touches +=1
                        print("FACE TOUCH - Num touches: " + str(num_face_touches) + "; Total runs: " + str(count))
                    rssi_vals.get()
                    accel_data.get()

                OTHER_LOCK.release()
                RSSI_LOCK.release()

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

    ratio = len(dist) // len(rssi)
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
    return pow(10, (tx - rssi) / (10 * n))


def distToRSSI(dist):
    # dist = 10^[(tx-rssi)/10n]
    n = 2
    tx = -55
    return log10(dist) * -10 * n + tx


def project(plane, vector):
    # c is the percentage of the plane vector that you have traversed so far
    c = np.dot(vector, plane) / (magnitude(plane) ** 2)
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

    start_x = sum([i[0] for i in vectors[:5]]) / 5
    start_y = sum([i[1] for i in vectors[:5]]) / 5
    start_z = sum([i[2] for i in vectors[:5]]) / 5

    end_x = sum([i[0] for i in vectors[-5:]]) / 5
    end_y = sum([i[1] for i in vectors[-5:]]) / 5
    end_z = sum([i[2] for i in vectors[-5:]]) / 5

    return [end_x - start_x, end_y - start_y, end_z - start_z]


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

    print(plane)

    projections = []
    previous_percentage = 0
    for d in dist:
        percentage_moved_total = project(plane, d)
        if percentage_moved_total < 0:
            new_movement_percent = abs(percentage_moved_total) - abs(previous_percentage)  # negative
            projections.append(abs(new_movement_percent) * 0.6096)  # +0.1

        else:
            new_movement_percent = percentage_moved_total - previous_percentage  # positive
            projections.append(new_movement_percent * -0.6096)  # -0.1

        previous_percentage = percentage_moved_total
    return projections


def main_kalman():
    rssi = list(rssi_vals)
    accel = list(accel_data)
    w = max(15, len(rssi))  # metric for optimized RSSI list length
    r = len(accel) // len(rssi)
    accel = accel[-w * r:]
    rssi = rssi_vals[-w:]
    if stat.median(rssi[-3:]) <= rssi[0]:
        dist = accelToDist(accel, True)
    else:
        dist = accelToDist(accel, False)

    rssi_dist = [rssiToDist(i) for i in rssi]
    rssi_sig = rssiToDist(-90)  # opt+clk wifi for rssi noise
    dist_sig = .00003  # 126ug/sqrt(Hz)^2 * 200Hz > m/s^2 FXOS8700CQ noise

    mu, sig = kalman(rssi_dist, dist, rssi_sig, dist_sig)
    if mu < 0.2:
        return True
    return False


if __name__ == '__main__':
    p1 = Thread(target = scan_rssi_vals)
    p1.start()
    p2 = Thread(target = get_accel_mag_gyro_temp_vals)
    p2.start()
