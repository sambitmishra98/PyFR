import math

from pyfr.integrators.dual.phys.base import BaseDualIntegrator


class BaseDualStepper(BaseDualIntegrator):
    pass


class BaseDIRKStepper(BaseDualStepper):
    stepper_nregs = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.fsal:
            self.b = self.a[-1]

        self.c = [sum(row) for row in self.a]

    @property
    def stage_nregs(self):
        return self.nstages

    def step(self, t, dt):
        for s, (sc, tc) in enumerate(zip(self.a, self.c)):
            self.pseudointegrator.init_stage(s, sc, dt)
            self.pseudointegrator.pseudo_advance(t + dt*tc)
            self.pseudointegrator.finalise_stage(s, t + dt*tc)

        if not self.fsal:
            bcoeffs = [bt*self._dt for bt in self.b]
            self.pseudointegrator.obtain_solution(bcoeffs)

        self.pseudointegrator.store_current_soln()

        if self.stepper_has_errest:
            s = 5
            sc = [(b - bh)*dt for b, bh in zip(self.b, self.bhat)]
            self.pseudointegrator.init_stage(s, sc, dt)
            self.pseudointegrator.pseudo_advance(t + dt)
            self.pseudointegrator.finalise_err_stage(t + dt)

            self.pseudointegrator.store_current_err()

            return (self.pseudointegrator._idxcurr,         # Current  solution
                    self.pseudointegrator._err_regidx[0],   #   Old    solution
                    self.pseudointegrator._err_regidx[1]    # Error in solution
                    )
        else:
            return self.pseudointegrator._idxcurr


class DualBackwardEulerStepper(BaseDIRKStepper):
    stepper_name = 'backward-euler'
    stepper_has_errest = False
    nstages = 1
    fsal = True

    a = [[1]]


class SDIRK33Stepper(BaseDIRKStepper):
    stepper_name = 'sdirk33'
    stepper_has_errest = False
    nstages = 3
    fsal = True

    _at = math.atan(0.5**1.5) / 3
    _al = (3**0.5*math.sin(_at) - math.cos(_at)) / 2**0.5 + 1

    a = [
        [_al],
        [0.5*(1 - _al), _al],
        [(4 - 1.5*_al)*_al - 0.25, (1.5*_al - 5)*_al + 1.25, _al]
    ]


class SDIRK43Stepper(BaseDIRKStepper):
    stepper_name = 'sdirk43'
    stepper_has_errest = False
    nstages = 3
    fsal = False

    _a_lam = (3 + 2*3**0.5*math.cos(math.pi/18))/6

    a = [
        [_a_lam],
        [0.5 - _a_lam, _a_lam],
        [2*_a_lam, 1 - 4*_a_lam, _a_lam]
    ]

    _b_rlam = 1/(6*(1 - 2*_a_lam)*(1 - 2*_a_lam))
    b = [_b_rlam, 1 - 2*_b_rlam, _b_rlam]


class ESDIRK35Stepper(BaseDIRKStepper):
    stepper_name = 'esdirk35'
    stepper_has_errest = True
    nstages = 5
    fsal = True

    _w = 9/40
    _x1 = 9*(1+math.sqrt(2))/80
    _y1 = (22+15*math.sqrt(2))/(80*(1+math.sqrt(2)))
    _y2 = -7/(40*(1+math.sqrt(2)))
    _z1 =  (2398+1205*math.sqrt(2))/( 2835*(4+3*math.sqrt(2)))
    _z2 = (-2374*(1+2*math.sqrt(2)))/(2835*(5+3*math.sqrt(2)))

    a = [
        [0],
        [_w, _w],
        [_x1, _x1, _w],
        [_y1, _y1, _y2, _w],
        [_z1, _z1, _z2, 5827/7560, _w]]

    b = [_z1, _z1, _z2, 5827/7560, _w]
    bhat = [4555948517383/24713416420891, 
            4555948517383/24713416420891, 
            -7107561914881/25547637784726, 
            30698249/44052120, 
            49563/233080]
