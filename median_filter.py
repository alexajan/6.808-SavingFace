import numpy as np
from scipy.signal import medfilt
import sys
import pylab as p
import csv


def parse_data(filename):
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
                    accel.append([float(values[0][acc_i + 3:]), float(values[1]), float(values[2][:-1])])

                elif row[0][0] == "M":  # Magnetometer
                    mag_i = row[0].find(":")
                    values = row[0].split(",")
                    mag.append([float(values[0][mag_i + 3:]), float(values[1]), float(values[2][:-1])])

                elif row[0][0] == "G":  # Gyroscope
                    gyro_i = row[0].find(":")
                    values = row[0].split(",")
                    gyro.append([float(values[0][gyro_i + 3:]), float(values[1]), float(values[2][:-1])])

                elif row[0][0] == "T":  # Temperature
                    temp_i = row[0].find(":")
                    temp.append(float(row[0][temp_i + 3:-1]))

            else:

                if row[0][0] == "D":  # RSSI
                    rssi_i = row[0].find("RSSI = ")
                    rssi.append(float(row[0][rssi_i + 7:]))

                elif row[0][0] == "A":  # Acceleration
                    acc_i = row[0].find(":")
                    accel.append([float(row[0][acc_i + 3:]), float(row[1]), float(row[2][:-1])])

                elif row[0][0] == "M":  # Magnetometer
                    mag_i = row[0].find(":")
                    mag.append([float(row[0][mag_i + 3:]), float(row[1]), float(row[2][:-1])])

                elif row[0][0] == "G":  # Gyroscope
                    gyro_i = row[0].find(":")
                    gyro.append([float(row[0][gyro_i + 3:]), float(row[1]), float(row[2][:-1])])

                elif row[0][0] == "T":  # Temperature
                    temp_i = row[0].find(":")
                    temp.append(float(row[0][temp_i + 3:-1]))

        return rssi, accel, mag, gyro

#given a file, returns all the medians with a window of l
def median_filter(data_points, k):
    end_padding_length = k // 2 + 1
    beg_padding_length = k // 2

    beginning_pad = np.full(shape=beg_padding_length, fill_value=data_points[0], dtype=np.int)
    end_pad = np.full(shape=end_padding_length, fill_value=data_points[-1], dtype=np.int)

    data_points = np.concatenate([beginning_pad, data_points])
    data_points = np.concatenate([data_points, end_pad])

    y = np.zeros((len(data_points) - k, k), dtype=data_points.dtype)
    for i in range(len(data_points) - k):
        k_len_array = np.array(data_points[i:i + k])
        y[i] = k_len_array

    return np.median(y, axis=1)

#detemine if a touch was detected using median filter
def test_median(w, threshold):
    total = 0
    total_num_files = 10
    for i in range(1, total_num_files+1):
        touch = False
        filename = "data_motions/" + str(i) + "_single_may5"
        #filename = "6.808-SavingFace/data_motions/notouch_face_single_" + str(i)

        rssi, accel, mag, gyro = parse_data(filename)
        median_points = median_filter(rssi, w)

        #for each median, see if is surpasses the threshold
        print("---------START---------- " + str(i))
        for i in range(len(median_points)):
            if median_points[i] > threshold:
                print("TOUCH")
                touch = True
            else:
                print("---------")
        if touch:
            total += 1
        print("----------DONE----------")
    print(str(total) + "/" + str(total_num_files))

#detemine if a touch was detected using threshold filter
#to detect a touch, all data within a window w must be above threshold
def test_threshold(w, threshold):
    total = 0
    total_num_files = 10
    for i in range(1, total_num_files+1):
        touch = False
        filename = "data_motions/" + str(i) + "_single_may5"
        rssi, accel, mag, gyro = parse_data(filename)
        print("---------START---------- " + str(i))
        for i in range(len(rssi) - w):
            all_above_t = True
            for j in range(0, w):
                if rssi[i + j] < threshold:
                    all_above_t = False
            if all_above_t:
                print("TOUCH")
                touch = True
            else:
                print("---------")
        if touch:
            total += 1
        print("----------DONE----------")
    print(str(total) + "/" + str(total_num_files))


if __name__ == '__main__':
    window = 3
    threshold = -30
    test_median(window, threshold)
