
import numpy as np
import math


def lvlh_frame(pos, vel):

    # Creating ECI to LVLH DCM
    z = pos / np.linalg.norm(pos)
    y = np.cross(pos, vel, axis=0) / np.linalg.norm(np.cross(pos, vel, axis=0))
    x = np.cross(y, z, axis=0)

    a_lvlh_eci = np.array([[x[0][0], x[1][0], x[2][0]],
                           [y[0][0], y[1][0], y[2][0]],
                           [z[0][0], z[1][0], z[2][0]]])

    return a_lvlh_eci


def pos_ecef_trans():

    year = 2021
    month = 11
    day = 12 + time // 86400
    h = time // 3600
    m = time // 60
    s = time

    t_0 = (1721013.5 + 367 * year - int((7 / 4) * (year + int((month + 9) / 12)))
           + int(275 * month / 9) + day - 2451545) / 36525

    theta = 24110.54841 + 8640184.812866*t_0+0.093194*t_0**2-6.2*10**-6*t_0**3+1.002737909350795*(3600*h+60*m+s)

    while theta > 86400:
        theta = theta-86400

    theta = theta/240*np.pi/180

    dcm = np.array([[np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

    return dcm


def get_julian_datetime(date):

    # Perform the calculation
    julian_datetime = 367 * date.year - int((7 * (date.year + int((date.month + 9) / 12.0))) / 4.0) + int(
        (275 * date.month) / 9.0) + date.day + 1721013.5 + (
                          date.hour + date.minute / 60.0 + date.second / math.pow(60,
                                                                                  2)) / 24.0 - 0.5 * math.copysign(
        1, 100 * date.year + date.month - 190002.5) + 0.5

    return julian_datetime


def q_dot(w, q):
    q1 = q[0][0]
    q2 = q[1][0]
    q3 = q[2][0]
    q4 = q[3][0]
    return 0.5 * np.matmul(np.array([[q4, -q3, q2],
                                     [q3, q4, -q1],
                                     [-q2, q1, q4],
                                     [-q1, -q2, -q3]]), w)


def q_to_dcm(att):
    q1 = att[0][0]
    q2 = att[1][0]
    q3 = att[2][0]
    q4 = att[3][0]

    return np.array([[q4 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 + q3 * q4), 2 * (q1 * q3 - q2 * q4)],
                     [2 * (q1 * q2 - q3 * q4), q4 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 + q1 * q4)],
                     [2 * (q1 * q3 + q2 * q4), 2 * (q2 * q3 - q1 * q4), q4 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]])


def att_err(curr, des):
    des = q_to_dcm(des)
    des = np.array([[np.arctan(des[0][1] / des[0][0])], [-np.arcsin(des[0][2])],
                    [np.arctan(des[1][2] / des[2][2])]])
    des = np.linalg.norm(des)

    curr = q_to_dcm(curr)
    curr = np.array([[np.arctan(curr[0][1] / curr[0][0])], [-np.arcsin(curr[0][2])],
                     [np.arctan(curr[1][2] / curr[2][2])]])
    curr = np.linalg.norm(curr)

    err = abs(curr-des)

    return err


def dcm_to_q(a):    # Computes a quaternion from the attitude matrix
    trace_a = np.trace(a)

    if a[0][0] > a[1][1] and a[0][0] > a[2][2] and a[0][0] > trace_a:
        q1 = np.array([[1 + 2 * a[0][0] - trace_a],
                       [a[0][1] + a[1][0]],
                       [a[0][2] + a[2][0]],
                       [a[1][2] - a[2][1]]])
        q1 = q1 / np.linalg.norm(q1)
        return q1

    elif a[1][1] > a[0][0] and a[1][1] > a[2][2] and a[1][1] > trace_a:
        q2 = np.array([[a[1][0] + a[0][1]],
                       [1 + 2 * a[1][1] - trace_a],
                       [a[1][2] + a[2][1]],
                       [a[2][0] - a[0][2]]])
        q2 = q2 / np.linalg.norm(q2)
        return q2

    elif a[2][2] > a[0][0] and a[2][2] > a[1][1] and a[2][2] > trace_a:
        q3 = np.array([[a[2][0] + a[0][2]],
                       [a[2][1] + a[1][2]],
                       [1 + 2 * a[2][2] - trace_a],
                       [a[0][1] - a[1][0]]])
        q3 = q3 / np.linalg.norm(q3)
        return q3

    else:
        q4 = np.array([[a[1][2] - a[2][1]],
                       [a[2][0] - a[0][2]],
                       [a[0][1] - a[1][0]],
                       [1 + trace_a]])
        q4 = q4 / np.linalg.norm(q4)
        return q4