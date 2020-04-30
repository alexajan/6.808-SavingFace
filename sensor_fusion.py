import csv
import sys
import matplotlib.pyplot as plt

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



parse_data()
