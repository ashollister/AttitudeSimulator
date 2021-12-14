
import numpy as np


class Plate:

    def __init__(self, area: float, normal: np.array, pos: np.array, absorpt: float, spec: float, diffus: float):
        self.area = area  # m^2
        self.normal = normal  # unit vector | SBF
        self.pos = pos  # m | SBF
        self.absorpt = absorpt  # unitless
        self.spec = spec  # unitless
        self.diffus = diffus  # unitless


class ReactionWheel:

    def __init__(self, jw: float, ws: float, w_dot: float, max_ws: float, min_ws: float, unit: np.array):
        self.jw = jw  # Wheel Moment of Inertia | # kg*m^2
        self.ws = ws  # Wheel Angular Rate | Rad/s
        self.w_dot = w_dot  # Wheel Angular Acceleration | Rad/s^2
        self.max_ws = max_ws  # Wheel Max Angular Rate | Rad/s
        self.min_ws = min_ws  # Wheel Min Angular Rate | Rad/s
        self.unit = unit  # Wheel unit vector

    def return_gyro(self):
        return self.jw*self.ws*self.unit

    def return_torq(self):
        return self.jw * self.w_dot * self.unit


class Thruster:

    def __init__(self, unit: np.array, pos_ori: int, torq: float = 35, desat: bool = 0):
        self.torq = torq  # Torque the thruster is capable of producing
        self.unit = unit  # Unit vector of thrust torque
        self.desat = desat  # boolean to indicate desat in progress
        self.pos_ori = pos_ori  # boolean to indicate positive or negative orientation of thruster

    def thrust(self, ws: float, ws_max: float, ws_min: float):
        self.desat += abs(ws) > ws_max  # Checks for a desat trigger
        self.desat *= np.sign(ws) != self.pos_ori  # Checks to see if the thruster is the properly oriented thruster
        self.desat = (abs(ws) > ws_min) * bool(self.desat)  # Checks to see if desat is complete
        thrust = bool(self.desat) * self.torq * self.unit
        return thrust


class Satellite:

    def __init__(self,
                 pos: np.array = np.array([[0], [0], [0]]),
                 vel: np.array = np.array([[0], [0], [0]]),
                 att: np.array = np.array([[0], [0], [0], [1]]),
                 des_att: np.array = np.array([[0], [0], [0], [1]]),
                 w: np.array = np.array([[0], [0], [0]]),
                 minute: int = 0):

        # Current Time
        self.minute = minute

        # Satellite Position - ECI Frame
        self.pos = pos  # km

        # Satellite Velocity - ECI Frame
        self.vel = vel  # km/s

        # Satellite Attitude in Quaternion Form - ECI Frame
        self.att = att

        # Satellite Desired Attitude in Quaternion Form - ECI Frame
        self.des_att = des_att

        # Satellite Angular Velocity - ECI Frame
        self.w = w

        # Atmospheric Density
        self.rho = 3e-18  # kg/m^3

        # Principal Moments of Inertia - SBF
        self.jc = np.diag([77217, 77217, 25000])  # kg*m^2
        self.jc_inv = np.linalg.inv(self.jc)

        # Reaction Wheels' Moments of Inertia - SBF
        self.rw_1 = ReactionWheel(jw=3,
                                  ws=0,
                                  w_dot=0,
                                  max_ws=700,
                                  min_ws=1,
                                  unit=np.array([[1], [0], [0]]))

        self.rw_2 = ReactionWheel(jw=3,
                                  ws=0,
                                  w_dot=0,
                                  max_ws=700,
                                  min_ws=1,
                                  unit=np.array([[0], [1], [0]]))

        self.rw_3 = ReactionWheel(jw=3,
                                  ws=0,
                                  w_dot=0,
                                  max_ws=700,
                                  min_ws=1,
                                  unit=np.array([[0], [0], [1]]))

        self.jw_inv = np.linalg.inv(np.diag([self.rw_1.jw, self.rw_2.jw, self.rw_3.jw]))

        # Thrusters of the Satellite
        self.thruster1 = Thruster(unit=np.array([[1], [0], [0]]), pos_ori=1)
        self.thruster2 = Thruster(unit=np.array([[-1], [0], [0]]), pos_ori=-1)
        self.thruster3 = Thruster(unit=np.array([[0], [1], [0]]), pos_ori=1)
        self.thruster4 = Thruster(unit=np.array([[0], [-1], [0]]), pos_ori=-1)
        self.thruster5 = Thruster(unit=np.array([[0], [0], [1]]), pos_ori=1)
        self.thruster6 = Thruster(unit=np.array([[0], [0], [-1]]), pos_ori=-1)

        # Satellite Magnetic Residual - SBF
        self.mag_res = np.array([[20], [20], [20]])  # Am^2

        # Satellite Plates
        self.plate_1 = Plate(area=23,
                             normal=np.array([[1], [0], [0]]),
                             pos=np.array([[1], [0], [0]]),
                             absorpt=0.1,
                             spec=0.8,
                             diffus=0.1)

        self.plate_2 = Plate(area=18,
                             normal=np.array([[np.cos(np.deg2rad(30))], [np.sin(np.deg2rad(30))], [0]]),
                             pos=np.array([[0], [0.6], [0]]),
                             absorpt=0.2,
                             spec=0.75,
                             diffus=0.05)

        self.plate_3 = Plate(area=23,
                             normal=np.array([[np.cos(np.deg2rad(30))], [np.sin(np.deg2rad(30))], [0]]),
                             pos=np.array([[0], [-0.6], [0]]),
                             absorpt=0.2,
                             spec=0.75,
                             diffus=0.05)

    def propagate(self, pos: np.array, vel: np.array):

        # Updating Position and Velocity
        self.pos = pos
        self.vel = vel

        # Propagating Attitude and Angular Rates
        def odes(y, t):
            q_1, q_2, q_3, q_4, w_1, w_2, w_3, rw_1, rw_2, rw_3 = y

            # Angular rate terms
            w = np.array([[w_1], [w_2], [w_3]])

            # Disturbance torques
            l_grav = grav_grad(self.pos, self.jc, self.att)
            l_mag = mag_torq(self.pos, self.att, self.mag_res)
            l_aero = aero_torq([self.plate_1, self.plate_2, self.plate_3], self.pos, self.vel, self.att, self.rho)
            l_srp = srp_torq([self.plate_1, self.plate_2, self.plate_3], self.pos, self.att)
            l_dist = l_srp+l_grav+l_mag+l_aero

            # Control Torque
            ctrl_t = controller(self.des_att, self.att, self.w)
            # ctrl_t = np.array([[0], [0], [0]])

            # Capping Control Torques - Keeps wheels from over saturating
            ctrl_t[0][0] = ctrl_t[0][0] * (abs(self.rw_1.ws) < self.rw_1.max_ws
                                           or np.sign(ctrl_t[0][0]) != np.sign(self.rw_1.ws))
            ctrl_t[1][0] = ctrl_t[1][0] * (abs(self.rw_2.ws) < self.rw_2.max_ws
                                           or np.sign(ctrl_t[1][0]) != np.sign(self.rw_2.ws))
            ctrl_t[2][0] = ctrl_t[2][0] * (abs(self.rw_3.ws) < self.rw_3.max_ws
                                           or np.sign(ctrl_t[2][0]) != np.sign(self.rw_3.ws))

            # Thruster Torques
            t1 = self.thruster1.thrust(self.rw_1.ws, self.rw_1.max_ws, self.rw_1.min_ws)
            t2 = self.thruster2.thrust(self.rw_1.ws, self.rw_1.max_ws, self.rw_1.min_ws)
            t3 = self.thruster3.thrust(self.rw_2.ws, self.rw_2.max_ws, self.rw_2.min_ws)
            t4 = self.thruster4.thrust(self.rw_2.ws, self.rw_2.max_ws, self.rw_2.min_ws)
            t5 = self.thruster5.thrust(self.rw_3.ws, self.rw_3.max_ws, self.rw_3.min_ws)
            t6 = self.thruster6.thrust(self.rw_3.ws, self.rw_3.max_ws, self.rw_3.min_ws)
            desat_t = sum([t1, t2, t3, t4, t5, t6])
            # desat_t = np.array([[0], [0], [0]])

            # Wheel Dynamics
            gyro_rw = np.cross(w, self.rw_1.return_gyro() + self.rw_2.return_gyro() + self.rw_3.return_gyro(), axis=0)
            rw_dot = np.matmul(self.jw_inv, desat_t+ctrl_t)

            # Gyroscopic terms
            gyro = np.cross(-w, np.matmul(self.jc, w), axis=0)
            w_dot = np.matmul(self.jc_inv, l_dist+ctrl_t+gyro+gyro_rw)

            # Recording Commanded Torques
            cmd_torqs.append(ctrl_t)

            # Recording Quaternion Error
            att_err_list.append(att_err(self.att, self.des_att))

            # Attitude Terms
            q = np.array([[q_1], [q_2], [q_3], [q_4]])
            q_dot_b = q_dot(w, q)

            return [q_dot_b[0][0],
                    q_dot_b[1][0],
                    q_dot_b[2][0],
                    q_dot_b[3][0],
                    w_dot[0][0],
                    w_dot[1][0],
                    w_dot[2][0],
                    rw_dot[0][0],
                    rw_dot[1][0],
                    rw_dot[2][0]]

        # Performing Numerical Integration
        t = np.linspace(0, 1, 2)
        att_list = []
        w_list = []
        cmd_torqs = []
        rw_list = []
        att_err_list = []

        for step in range(60):
            # Command Torque Data
            cmd_torqs = []
            att_err_list = []

            # Updating time
            global time
            time = self.minute*60 + step

            y_0 = [self.att[0][0], self.att[1][0], self.att[2][0], self.att[3][0],
                   self.w[0][0], self.w[1][0], self.w[2][0],
                   self.rw_1.ws, self.rw_2.ws, self.rw_3.ws]

            sol = odeint(odes, y_0, t)

            q1 = sol[:, 0]
            q2 = sol[:, 1]
            q3 = sol[:, 2]
            q4 = sol[:, 3]
            w1 = sol[:, 4]
            w2 = sol[:, 5]
            w3 = sol[:, 6]
            rw1 = sol[:, 7]
            rw2 = sol[:, 8]
            rw3 = sol[:, 9]

            prop_q = np.transpose(([q1, q2, q3, q4]))
            prop_w = np.transpose(([w1, w2, w3]))

            self.att = np.reshape(prop_q[-1], (4, 1))
            self.w = np.reshape(prop_w[-1], (3, 1))
            self.rw_1.ws = rw1[-1]
            self.rw_2.ws = rw2[-1]
            self.rw_3.ws = rw3[-1]

            att_list.append(self.att)
            w_list.append(self.w)
            rw_list.append(np.array([[self.rw_1.ws], [self.rw_2.ws], [self.rw_3.ws]]))

        return att_list, w_list, cmd_torqs, rw_list, att_err_list


"""
PID Controller
"""


def controller(des_att: np.array, curr_att: np.array, curr_w: np.array):

    # Unpacking data elements for convenience
    # Current quaternion
    qc1 = curr_att[0][0]
    qc2 = curr_att[1][0]
    qc3 = curr_att[2][0]
    qc4 = curr_att[3][0]

    # Desired quaternion
    qd1 = des_att[0][0]
    qd2 = des_att[1][0]
    qd3 = des_att[2][0]
    qd4 = des_att[3][0]

    # Current angular rates
    p = curr_w[0][0]
    q = curr_w[1][0]
    r = curr_w[2][0]

    # Deriving error quaternion
    qd_err = np.array([[qc4, qc3, -qc2, qc1],
                       [-qc3, qc4, qc1, qc2],
                       [qc2, -qc1, qc4, qc3],
                       [-qc1, -qc2, -qc3, qc4]])
    qc_err = np.array([[-qd1], [-qd2], [-qd3], [qd4]])
    q_err = np.matmul(qd_err, qc_err)

    # Proportional gains
    kx = 10
    ky = 10
    kz = 10

    # Derivative gains
    kdx = 10000
    kdy = 10000
    kdz = 10000

    # Calculating control torque
    tx = -2*kx*q_err[0][0]*q_err[3][0] - kdx*p
    ty = -2*ky*q_err[1][0]*q_err[3][0] - kdy*q
    tz = -2*kz*q_err[2][0]*q_err[3][0] - kdz*r

    # Max Torque
    max_t = 1
    # if time > 60*90:
    #     max_t = 10e-2

    # Capping control torques at 1 Nm
    tx = np.sign(tx)*min(max_t, abs(tx))
    ty = np.sign(ty)*min(max_t, abs(ty))
    tz = np.sign(tz)*min(max_t, abs(tz))

    # Formatting Torque
    ctrl_t = np.array([[tx], [ty], [tz]])

    return ctrl_t
