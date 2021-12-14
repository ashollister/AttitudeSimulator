
import numpy as np
import datetime


def grav_grad(pos: np.array, jb: np.array, att: np.array):
    mu = 398600  # km^3/s^2

    # Transforming attitude into a DCM
    a_body_eci = q_to_dcm(att)

    # Expressing Position in Body Frame
    pos = np.matmul(a_body_eci, pos)

    # Calculating the gravity gradient torque
    g_torq = 3*mu/np.linalg.norm(pos)**5*np.cross(pos, np.matmul(jb, pos), axis=0)

    return g_torq


def mag_torq(pos, att, res):

    # Defining Magnetic Spherical Reference Radius
    a = 6371.2e3  # m

    # Defining Additional Magnetic Constants
    g_1_0 = -29554.63e-9
    g_1_1 = -1669.05e-9
    h_1_1 = 5077.99e-9

    # Earth Magnetic Dipole Moment in ECEF Frame
    m = a**3*np.array([[g_1_1], [h_1_1], [g_1_0]])

    # ECEF Transformation Frame
    a_ecef_eci = pos_ecef_trans()

    # Position vector expressed in ecef frame
    pos_ecef = np.matmul(a_ecef_eci, 1e3*pos)

    # Magnetic field in ecef frame
    b = (3*np.matmul(np.transpose(m), pos_ecef)[0]*pos_ecef - np.linalg.norm(pos_ecef)**2*m) / \
        np.linalg.norm(pos_ecef)**5

    # Magnetic torque in ecef frame
    l_mag = np.cross(res, b, axis=0)

    # Creating ecef to body transformation
    a_body_eci = q_to_dcm(att)
    a_body_ecef = np.matmul(a_body_eci, np.transpose(a_ecef_eci))

    # Expressing magnetic torque in body frame
    l_mag_bod = np.matmul(a_body_ecef, l_mag)

    return l_mag_bod


def aero_torq(plates: list, pos: np.array, v_eci: np.array, att: np.array, rho: float):

    # Calculation of aerodynamic torque for all plates
    l_aero_sum = np.array([[0], [0], [0]])
    for index, plate in enumerate(plates):

        # Defining earth rotation in eci frame
        we_eci = np.array([[0], [0], [2 * np.pi / (23 * 3600 + 56 * 60 + 4)]])

        # Defining plate relative velocity in the body frame
        v_rel_eci = v_eci + np.cross(we_eci, pos, axis=0)
        a_body_eci = q_to_dcm(att)
        v_rel_body = np.matmul(a_body_eci, v_rel_eci)

        # Inclination of the plate with respect to the relative velocity
        theta_aero = np.matmul(np.transpose(plate.normal), v_rel_body)/np.linalg.norm(v_rel_body)

        # Approximate drag coefficient
        cd = 2.25

        # Aero dynamic force
        f_aero = -0.5*rho*cd*np.linalg.norm(v_rel_body)*v_rel_body*plate.area*max(np.cos(theta_aero), 0)*1e6

        # Position of the plate in the body frame
        ri = plate.pos*plate.normal

        # Aerodynamic torque in body frame
        l_aero = np.cross(ri, f_aero, axis=0)
        l_aero_sum = np.add(l_aero_sum, l_aero)

    return l_aero_sum


def srp_torq(plates, pos, att):

    # Getting time
    year = 2021
    month = 11
    day = 12 + time // 86400
    h = time // 3600 % 24
    m = time // 60 % 60
    s = time % 60

    # Determining angles based on date
    date = datetime.datetime(year, month, day, h, m, s)
    t_ut1 = get_julian_datetime(date)
    m_sun = np.deg2rad(357.5277233 + 35999.05034 * t_ut1)
    epsilon = np.deg2rad(23.439291 - 0.0130042 * t_ut1)
    phi_sun = 280.46 + 36000.771 * t_ut1
    phi_ecliptic = np.deg2rad(phi_sun + 1.914666471 * np.sin(m_sun) + 0.019994643 * np.sin(2 * m_sun))

    # Sun to satellite vector
    dist = (1.000140612-0.016708617*np.cos(m_sun)-0.000139589*np.cos(2*m_sun))
    r_earth_sun = dist * np.array([[np.cos(phi_ecliptic)],
                                   [np.cos(epsilon) * np.sin(phi_ecliptic)],
                                   [np.sin(epsilon) * np.sin(phi_ecliptic)]])

    # Sun to satellite vector
    r_sat_sun = r_earth_sun * 149597870.7 - pos

    # Inertial to body transformation
    a_body_eci = q_to_dcm(att)

    # Sun to satellite unit vector
    r_sat_sun_unit = np.matmul(a_body_eci, r_sat_sun / np.linalg.norm(r_sat_sun))

    # Sun to satellite vector in AU
    r_sat_sun_au = r_sat_sun / 149597870.7

    # Solar radiation pressure
    srp = 1361 / (299792458 * np.linalg.norm(r_sat_sun_au) ** 2)

    # Calculating torques on each plate
    l_srp_sum = np.array([[0], [0], [0]])
    for index, plate in enumerate(plates):

        # Angle between plate normal and sun unit vector
        theta_i = np.cos(np.matmul(np.transpose(plate.normal), r_sat_sun_unit))

        # Reflectivity of plate
        r_plate = (2 * (plate.diffus / 3 + plate.spec * np.cos(theta_i)) * plate.normal +
                   (1 - plate.spec) * r_sat_sun_unit)

        # Force on plate
        f_srp_i = (srp * plate.area * r_plate * max(np.cos(theta_i), 0))

        # Adding torque due to plate to summation
        l_srp_sum = np.add(l_srp_sum, np.cross(plate.pos, f_srp_i, axis=0))

    return l_srp_sum
