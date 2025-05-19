import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        model1 = ManipulatorModel(Tp)
        model2 = ManipulatorModel(Tp)
        model3 = ManipulatorModel(Tp)
        model1.m3 = 0.1
        model1.r3 = 0.05
        model2.m3 = 0.01
        model2.r3 = 0.01
        model3.m3 = 1.0
        model3.r3 = 0.3

        self.models = [model1, model2, model3]
        self.i = 0
        self.u = np.zeros((2, 1))
        
    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        q1, q2, q1_dot, q2_dot = x
        q = [q1, q2]
        q_dot = np.array([q1_dot, q2_dot])
        x_hat = []
        errors = []

        for indx, model in enumerate(self.models):
            M = model.M(x)
            C = model.C(x)
            y = np.linalg.inv(M) @ self.u - C @ np.reshape(q_dot, (2, 1))
            x_hat.append((y[0], y[1]))

        for i in x_hat:
            errors.append((abs((q[0] - i[0]) + q[1] - i[1])))

        min = np.min(errors)
        self.i = errors.index(min)
        print(self.i)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]

        Kp = 150
        Kd = 10

        v = q_r_ddot  + Kd * (q_r_ddot - q_dot) + Kp * (q_r - q) # TODO: add feedback
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
