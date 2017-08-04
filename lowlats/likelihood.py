# coding: utf-8
from __future__ import division, print_function

# Third-party
import astropy.units as u
import numpy as np

__all__ = ['VelocityModel', 'LinearVelocityModel']

def ln_normal(x, mu, var):
    return -0.5*np.log(2*np.pi) - 0.5*np.log(var) - 0.5 * (x-mu)**2 / var

class VelocityModel(object):

    _param_names = None # subclasses must override with a list

    def __init__(self, mgiants, rrlyrae, frozen=None, **kwargs):
        """
        Parameters
        ----------
        mgiants : `~astropy.table.Table`
        rrlyrae : `~astropy.table.Table`
        frozen : iterable (optional)
            A dictionary of parameter names to freeze and their values to freeze
            to. For example, to freeze `dv_dl` to 0.5 km/s/deg, pass
            `{'dv_dl': 0.5}`.

        **kwargs
            Any extra arguments are stored as constants as attributes of the
            instance.
        """
        if frozen is None:
            frozen = dict()
        self.frozen = dict(frozen)

        self.mgiants = mgiants
        self._mg = (np.asarray(self.mgiants['l']),
                    np.asarray(self.mgiants['v_gsr']),
                    np.asarray(self.mgiants['v_err']))
        self.rrlyrae = rrlyrae
        self._rr = (np.asarray(self.rrlyrae['l']),
                    np.asarray(self.rrlyrae['v_gsr']),
                    np.asarray(self.rrlyrae['v_err']))

        # store extra metadata
        for k,v in kwargs.items():
            setattr(self, k, v)

    def pack_pars(self, **kwargs):
        vals = []
        for k in self._param_names:
            frozen_val = self.frozen.get(k, None)
            val = kwargs.get(k, frozen_val)
            if val is None:
                raise ValueError("No value passed in for parameter {0}, but "
                                 "it isn't frozen either!".format(k))
            vals.append(val)
        return np.array(vals)

    def unpack_pars(self, p):
        key_vals = []

        j = 0
        for name in self._param_names:
            if name in self.frozen:
                key_vals.append((name, self.frozen[name]))
            else:
                key_vals.append((name, p[j]))
                j += 1

        return dict(key_vals)

    def ln_prior(self, **kwargs):
        return 0.

    def ln_posterior(self, p):
        # unpack parameter vector, p
        kw_pars = self.unpack_pars(p)

        lnp = self.ln_prior(**kw_pars)
        if not np.isfinite(lnp):
            return -np.inf, None

        lnl, blob = self.ln_likelihood(**kw_pars)
        if not np.isfinite(lnl):
            return -np.inf, None

        return lnp + lnl.sum(), blob

    def __call__(self, p):
        return self.ln_posterior(p)


class LinearVelocityModel(VelocityModel):

    _param_names = ['dv_dl', 'v0', 'lnV', 'f_mg', 'f_rr', 'l0']

    # mixture model for everything
    def ln_prior(self, dv_dl, v0, lnV, f_mg, f_rr, l0):

        if dv_dl > 50. or dv_dl < -50:
            return -np.inf

        if v0 < -500 or v0 > 500:
            return -np.inf

        if lnV < 1E-1 or lnV > 9.: # < 1 km/s or > 90 km/s
            return -np.inf

        if f_mg > 1. or f_mg < 0.2:
            return -np.inf

        if f_rr > 1. or f_rr < 0.:
            return -np.inf

        return 0.

    def ln_likelihood_per_component(self, tracer, **kwargs):
        dv_dl = kwargs['dv_dl']
        v0 = kwargs['v0']
        lnV = kwargs['lnV']
        V1 = tracer[2]**2 + self.halo_sigma_v**2
        V2 = tracer[2]**2 + np.exp(lnV)
        term1 = ln_normal(tracer[1], 0., V1) # 0 = halo mean velocity
        term2 = ln_normal(tracer[1], dv_dl*(tracer[0]-kwargs['l0']) + v0, V2)
        return np.array([term1, term2])

    def ln_likelihood(self, **kw_pars):

        # M giants
        ll_mg = self.ln_likelihood_per_component(tracer=self._mg, **kw_pars)
        term1_mg = ll_mg[0] + np.log(1-kw_pars['f_mg'])
        term2_mg = ll_mg[1] + np.log(kw_pars['f_mg'])
        ll1 = np.logaddexp(term1_mg, term2_mg)

        # RR Lyrae
        ll_rr = self.ln_likelihood_per_component(tracer=self._rr, **kw_pars)
        term1_rr = ll_rr[0] + np.log(1-kw_pars['f_rr'])
        term2_rr = ll_rr[1] + np.log(kw_pars['f_rr'])
        ll2 = np.logaddexp(term1_rr, term2_rr)

        return ll1.sum() + ll2.sum(), (term1_mg, term2_mg, term1_rr, term2_rr)
