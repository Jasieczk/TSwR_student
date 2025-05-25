import numpy as np

#from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManipulatorModel
#from models.ideal_model import IdealModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        p0 = p[0]
        p1 = p[1]
        self.model = ManipulatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array([[3*p0, 0], [0, 3*p1], [3*p0**2, 0],[0, 3*p1**2], [p0**3, 0],[0, p1**3]])
        W = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        A = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0]])
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot])
        M = self.model.M(x)
        C = self.model.C(x)
        M_invC = np.linalg.inv(M) @ C 
        A = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        A[2:4,2:4] = -M_invC
        self.eso.A = A 
        B = np.zeros((6, 2))
        B[2:4,:] = np.linalg.inv(M)
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        M = self.model.M(x)
        C = self.model.C(x)
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1,q2])
        state = self.eso.get_state()
        q_approx_dot = state[2:4]
        f = state[4:6]
        e = q_d - q
        e_dot = q_d_dot - q_approx_dot
        v = self.Kp @ e + self.Kd @ e_dot + q_d_ddot
        u = M@(v - f) + C@q_approx_dot
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1)) 
        return u
