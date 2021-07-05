# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2020 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`ChiouYoungs2014`.
"""
import numpy as np
import math

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014 as CY14
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


class ChiouYoungs2013(GMPE):
    """
    Implements the CY13 GMPE by Chiou & Youngs (2013) for vertical-component
    ground motions from the PEER NGA-West2 Project

    This model follows the same functional form as in CY14 horizontal GMPE
    by Chiou & Youngs (2014) with minor modifications to the underlying
    parameters.

    **Reference:**

    Chiou, B. S.-J., & Youngs, R. R. (2013). Ground motion prediction model for
    vertical component of peak ground motions and response spectra, Chapter 5
    in *NGA-West2 Ground Motion Prediction Equations for Vertical Ground
    Motions*, PEER Report 2013/24, Pacific Earthquake Engineering Research
    Center, University of California, Berkeley, CA.

    Implements the global model that uses datasets from California, Taiwan,
    the Middle East, and other similar active tectonic regions to represent
    a typical or average Q region.

    Applies the average attenuation case
    """
    #: Supported tectonic region type is active shallow crust
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are spectral acceleration,
    #: peak ground velocity and peak ground acceleration
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([SA])

    #: Supported intensity measure component is the
    #: :attr:`~openquake.hazardlib.const.IMC.Vertical` direction component
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.VERTICAL

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see chapter "Variance model".
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    ])

    #: Required site parameters are Vs30, Vs30 measured flag
    #: and Z1.0.
    REQUIRES_SITES_PARAMETERS = {'vs30', 'vs30measured', 'z1pt0'}

    #: Required rupture parameters are magnitude, rake,
    #: dip and ztor.
    REQUIRES_RUPTURE_PARAMETERS = {'dip', 'rake', 'mag', 'ztor'}

    #: Required distance measures are RRup, Rjb and Rx.
    REQUIRES_DISTANCES = {'rrup', 'rjb', 'rx'}

    #: Reference shear wave velocity
    DEFINED_FOR_REFERENCE_VELOCITY = 1130

    #: Test suite data for code verification not available
    non_verified = True

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extracting dictionary of coefficients specific to required
        # intensity measure type.
        C = self.COEFFS[imt]

        # intensity on a reference soil is used for both mean
        # and stddev calculations.
        ln_y_ref = self._get_ln_y_ref(rup, dists, C)

        mean = self._get_mean(sites, C, ln_y_ref)
        stddevs = self._get_stddevs(sites, rup, C, stddev_types, ln_y_ref)

        return mean, stddevs

    def _get_mean(self, sites, C, ln_y_ref):
        """
        Add site effects to an intensity.

        Implements eq. 13b.
        """
        reg = self._get_regional_terms(C)
        # we do not support estimating of basin depth and instead
        # rely on it being available (since we require it).
        # centered_z1pt0
        centered_z1pt0 = self._get_centered_z1pt0(sites)

        # we consider random variables being zero since we want
        # to find the exact mean value.
        eta = epsilon = 0.

        # deep soil correction
        no_correction = sites.z1pt0 <= 0
        deep_s = reg['phi5'] * (1.0 - np.exp(-1. * centered_z1pt0
                        / reg['phi6']))
        deep_s[no_correction] = 0

        ln_y = (
            # first line of eq. 12
            ln_y_ref + eta
            # second + 3rd line
            + self._get_site_term(C, sites.vs30, ln_y_ref)
            # fourth line
            + deep_s
            # fifth line
            + epsilon
        )

        return ln_y

    def _get_site_term(self, C, vs30, ln_y_ref):
        """
        This implements the site term of the CY14 GMM. See
        :class:`openquake.hazardlib.gsim.chiou_youngs_2014.ChiouYoungs2014`
        for additional information.
        """
        reg = self._get_regional_terms(self, C)
        f_site = reg['phi1'] / (1. + ((vs30/reg['phi1a']) ** reg['phi1b']))
        return f_site

    def _get_regional_terms(self, C):
        """
        Retrieve regional terms

        Note that the estimates for the attenuation coefficient (gamma) in
        New Zealand, Taiwan, and Turkey are similar to those obtained for
        California earthquakes.
        """
        # get regional anelastic attenuation coefficient, gamma
        gamma = (C['cg1'] + C['cg2'] / (np.cosh(max(rup.mag - C['cg3'], 0.))))
        # collect regional terms into dictionary
        reg = { 'gamma': gamma,
                'phi1': C['phi1'],
                'phi1a': C['phi1a'],
                'phi1b': C['phi1b'],
                'phi5': C['phi5'],
                'phi6': C['phi6'],
                'sig2': C['sig2']
                }
        return reg


    def _get_stddevs(self, sites, rup, C, stddev_types, ln_y_ref):
        """
        Get standard deviation for a given intensity on reference soil.

        Implements equations 13 for inter-event, intra-event
        and total standard deviations.
        """
        reg = self._get_regional_terms(C)

        Fmeasured = sites.vs30measured
        Finferred = 1 - sites.vs30measured

        # eq. 13 to calculate inter-event standard error
        mag_test = min(max(rup.mag, 5.0), 6.5) - 5.0
        tau = C['tau1'] + (C['tau2'] - C['tau1']) / 1.5 * mag_test

        y_ref = np.exp(ln_y_ref)
        sigma = ((C['sig1'] + (reg['sig2'] - C['sig1']) * mag_test / 1.5)
                 * np.sqrt(C['sig3'] * Finferred + 0.7 * Fmeasured + 1.)
                 )

        ret = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                ret += [np.sqrt((tau ** 2) + (sigma ** 2))]
            elif stddev_type == const.StdDev.INTRA_EVENT:
                ret.append(sigma)
            elif stddev_type == const.StdDev.INTER_EVENT:
                ret.append(tau)
        return ret

    def _get_ln_y_ref(self, rup, dists, C):
        """
        Get median IMT on reference condition (Vs30 = 1100 m/s), equation 5.6

        Unlike in CY14 horizontal GMPE, directivity terms are excluded here
        """
        # reverse faulting flag
        Frv = 1. if 30 <= rup.rake <= 150 else 0.
        # normal faulting flag
        Fnm = 1. if -120 <= rup.rake <= -60 else 0.
        # hanging wall flag
        Fhw = np.zeros_like(dists.rx)
        idx = np.nonzero(dists.rx >= 0.)
        Fhw[idx] = 1.

        # evaluate cosh terms in SOF, ztor, and fault dip scaling
        mag_test1 = np.cosh(2. * max(rup.mag - 4.5, 0))

        # get centered_ztor value
        centered_ztor = self._get_centered_ztor(rup, Frv)

        # get regional terms
        reg = self._get_regional_terms(C)
        gamma = reg['gamma']

        ln_y_ref = (
            # EVENT SCALING TERMS
            C['c1']
            # SOF
            + (C['c1a'] + C['c1c'] / mag_test1) * Frv
            + (C['c1b'] + C['c1d'] / mag_test1) * Fnm
            # Rupture Depth
            + (C['c7'] + C['c7b'] / mag_test1) * centered_ztor
            # Fault Dip
            + (C['c11'] + C['c11b'] / mag_test1) *\
                np.cos(math.radians(rup.dip)) ** 2.
            # Magnitude Scaling
            + self.CONSTS['c2'] * (rup.mag - 6.)
            + ((self.CONSTS['c2'] - C['c3']) / C['cn'])
            * np.log(1. + np.exp(C['cn'] * (C['cm'] - rup.mag)))
            # DISTANCE SCALING
            # Near-source geometric spreading term and extended rupture effect
            + self.CONSTS['c4']
            * np.log(dists.rrup + C['c5']
                     * np.cosh(C['c6'] * max(rup.mag - C['chm'], 0.)))
            # Transition to far-source geometric spreading term
            + (self.CONSTS['c4a'] - self.CONSTS['c4'])
            * np.log(np.sqrt(dists.rrup ** 2. + self.CONSTS['crb'] ** 2.))
            # Anelastic attenuation and scattering term
            + gamma * dists.rrup
            # Hanging Wall Effects
            + C['c9'] * Fhw * np.cos(math.radians(rup.dip)) *
            (C['c9a'] + (1 - C['c9a']) * np.tanh(dists.rx / C['c9b']))
            * (1. - np.sqrt(dists.rjb ** 2. + rup.ztor ** 2.)
               / (dists.rrup + 1.0))
        )

        return ln_y_ref

    def _get_centered_z1pt0(self, sites):
        """
        Get z1pt0 centered on the Vs30- dependent average z1pt0(m)
        California and non-Japan regions
        """
        #: California and non-Japan regions
        return CY14._get_centered_z1pt0(self, sites)

    def _get_centered_ztor(self, rup, Frv):
        """
        Get ztor centered on the M- dependent avarage ztor(km)
        by different fault types.
        """
        return CY14._get_centered_ztor(self, rup, Frv)

    #: Coefficients obtained from Table 5.3 (Period-dependent coefficients of
    #: model for ln(yref)), Table 5.4 (Coefficients of site response model for
    #: ln(y)), and Table 5.5 (GMPE coefficients for non-California regions)
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT      c1         c1a       c1b        c1c        c1d        cn         cm        c3        c5         chm       c6        c7      c7b        c9        c9a       c9b       c11     c11b       cg1        cg2        cg3       phi1      phi1a    phi1b    phi5     phi6    g_JpIt    g_Wn      phi1_Jp   phi1_Tw   phi1a_Jp   phi1b_Jp   phi5_Jp   phi6_Jp   tau1      tau2      sig1      sig2      sig3      sig2_Jp
    0.01     -2.2621    0.165     -0.3729    -0.165     0.1977     16.0875    4.9993    1.8616    5.453      3.0956    0.508     0       0.0855     0.9228    0.1202    6.8607    0       -0.4536    -0.0084    -0.0048    4.2542    0.87      660.7    3        0        300     1.2818    0.6771    0.778     0.2       460.5      3          0.477     800       0.42      0.33      0.4912    0.3762    0.8       0.4528
    0.02     -2.2629    0.165     -0.3772    -0.165     0.218      15.7118    4.9993    1.8523    5.0265     3.0963    0.508     0       0.0871     0.9296    0.1217    6.8697    0       -0.4536    -0.0085    -0.0049    4.2386    0.87      660.6    3        0        300     1.2769    0.673     0.7598    0.2       461.9      3          0.473     800       0.423     0.3289    0.4904    0.3762    0.8       0.4551
    0.03     -2.1389    0.165     -0.4429    -0.165     0.3484     15.8819    4.9993    1.807     4.582      3.0974    0.508     0       0.0957     0.9396    0.1194    6.9113    0       -0.4536    -0.0089    -0.0051    4.2519    0.87      660.3    3        0        300     1.2764    0.6712    0.7062    0.2       461.3      3          0.468     800       0.4271    0.3273    0.4988    0.3849    0.8       0.4571
    0.04     -1.9451    0.165     -0.5122    -0.165     0.4733     16.4556    4.9993    1.786     4.4501     3.0988    0.508     0       0.1032     0.9661    0.1166    7.0271    0       -0.4536    -0.0097    -0.0049    4.296     0.87      660      3        0        300     1.2769    0.6704    0.6572    0.2       460.5      3          0.462     800       0.4309    0.3259    0.5049    0.391     0.8       0.4642
    0.05     -1.7424    0.165     -0.5544    -0.165     0.5433     17.6453    4.9993    1.7827    4.6504     3.1011    0.508     0       0.1066     0.9794    0.1176    7.0959    0       -0.4536    -0.0105    -0.0047    4.3578    0.87      659.5    3        0        300     1.2652    0.6701    0.6389    0.2       459.9      3          0.458     800       0.4341    0.3247    0.5096    0.3957    0.8       0.4716
    0.075    -1.3529    0.165     -0.5929    -0.165     0.5621     20.1772    5.0031    1.8426    5.8073     3.1094    0.508     0       0.0952     1.026     0.1171    7.3298    0       -0.4536    -0.0117    -0.0037    4.5455    0.87      657.9    3.005    0        300     1.2172    0.6694    0.7371    0.2       460.6      2.995      0.451     800       0.4404    0.3223    0.5179    0.4043    0.8       0.5022
    0.1      -1.2191    0.165     -0.576     -0.165     0.4633     19.9992    5.0172    1.9156    6.9412     3.2381    0.508     0       0.0829     1.0177    0.1146    7.2588    0       -0.4536    -0.0121    -0.0026    4.7603    0.87      655.6    3.36     0        300     1.17      0.665     0.874     0.2       461.7      2.94       0.448     800       0.445     0.3206    0.5236    0.4104    0.8       0.523
    0.12     -1.2007    0.165     -0.5583    -0.165     0.4        18.7106    5.0315    1.9704    7.6152     3.3407    0.508     0       0.075      1.0008    0.1128    7.2372    0       -0.4536    -0.0118    -0.0021    4.8963    0.87      653.2    4.062    0        300     1.1615    0.6586    0.9769    0.2       459.8      2.842      0.447     800       0.4479    0.3195    0.5221    0.4109    0.8       0.5235
    0.15     -1.2392    0.165     -0.5345    -0.165     0.3337     16.6246    5.0547    2.0474    8.3585     3.43      0.508     0       0.0654     0.9801    0.1106    7.2109    0       -0.4536    -0.0112    -0.0018    5.0644    0.87      648.8    4.929    0        300     1.171     0.6461    1.1222    0.2       451.7      2.652      0.445     800       0.4514    0.3182    0.5202    0.4116    0.8       0.5209
    0.17     -1.2856    0.165     -0.5188    -0.165     0.2961     15.3709    5.0704    2.0958    8.7181     3.4688    0.508     0       0.0601     0.9652    0.115     7.2491    0       -0.4536    -0.0107    -0.0016    5.1371    0.87      645.4    5.262    0        300     1.1785    0.6372    1.2109    0.2       443.1      2.478      0.445     800       0.4533    0.3175    0.5191    0.4119    0.8       0.5187
    0.2      -1.3599    0.165     -0.4944    -0.165     0.2438     13.7012    5.0939    2.1638    9.117      3.5146    0.508     0       0.0531     0.9459    0.1208    7.2988    0       -0.444     -0.0102    -0.0014    5.188     0.87      639.6    5.553    0        300     1.1845    0.624     1.3421    0.2       427.7      2.155      0.444     800       0.4558    0.3166    0.5177    0.4124    0.8       0.5152
    0.25     -1.4633    0.165     -0.4517    -0.165     0.162      11.2667    5.1315    2.2628    9.5761     3.5746    0.5068    0       0.043      0.9196    0.1208    7.3691    0       -0.3539    -0.0095    -0.0012    5.2164    0.8652    628.5    5.854    0        300     1.1864    0.6036    1.5379    0.2       399.8      1.698      0.445     800       0.459     0.3154    0.5159    0.413     0.7999    0.51
    0.3      -1.5533    0.165     -0.4122    -0.165     0.0881     9.1908     5.167     2.3439    9.8569     3.6232    0.505     0       0.034      0.8829    0.1175    6.8789    0       -0.2688    -0.0089    -0.0009    5.1954    0.8434    616.3    6.061    0        300     1.1846    0.5849    1.6771    0.2       373.9      1.436      0.447     800       0.4615    0.3144    0.5143    0.4135    0.7997    0.5059
    0.4      -1.7318    0.165     -0.3532    -0.165     -0.0287    6.5459     5.2317    2.4636    10.1521    3.6945    0.5007    0       0.0183     0.8302    0.106     6.5334    0       -0.1793    -0.0081    -0.0005    5.0899    0.7698    590.8    6.292    0        300     1.1858    0.5494    1.76      0.2       340        1.206      0.456     800       0.4652    0.313     0.5119    0.4144    0.7988    0.5002
    0.5      -1.9025    0.165     -0.3101    -0.165     -0.1158    5.2305     5.2893    2.5461    10.2969    3.7401    0.4961    0       0.0056     0.7884    0.1061    6.526     0       -0.1428    -0.0073    -0.0004    4.7854    0.7263    566.9    6.379    0        300     1.2158    0.5156    1.731     0.2       336.1      1.14       0.47      800       0.4679    0.312     0.51      0.415     0.7966    0.4959
    0.75     -2.274     0.165     -0.2219    -0.165     -0.2708    3.7896     5.4109    2.6723    10.4613    3.7941    0.4846    0       -0.0158    0.6754    0.1       6.5       0       -0.1138    -0.0055    -0.0009    4.3304    0.736     522.2    6.36     0.046    300     1.3014    0.4429    1.4999    0.2       391.8      1.37       0.521     800       0.4724    0.3103    0.4973    0.4256    0.7792    0.4985
    1        -2.5805    0.165     -0.1694    -0.165     -0.3527    3.3024     5.5106    2.7479    10.5397    3.8144    0.4704    0       -0.028     0.6196    0.1       6.5       0       -0.1062    -0.0043    -0.0015    4.1667    0.796     496.2    6.22     0.11     300     1.4162    0.3886    1.29      0.2       435.5      1.6        0.591     800       0.4753    0.3093    0.4882    0.4331    0.7504    0.4998
    1.5      -3.047     0.165     -0.1376    -0.165     -0.3454    2.8498     5.6705    2.8355    10.5992    3.8284    0.4401    0       -0.0422    0.5101    0.1       6.5       0       -0.102     -0.0029    -0.0022    4.0029    0.9023    472.3    5.716    0.199    300     1.7863    0.3315    1.0539    0.214     454.1      2.088      0.757     800       0.4788    0.3079    0.4755    0.4436    0.7136    0.5001
    2        -3.3941    0.1645    -0.1218    -0.1645    -0.2605    2.5417     5.7981    2.8806    10.6045    3.833     0.4264    0       -0.0511    0.3917    0.1       6.5       0       -0.1009    -0.0021    -0.0026    3.8949    1.0001    462.7    4.952    0.26     300     2.0498    0         0.9199    0.3285    455.7      2.422      0.924     800       0.4811    0.3071    0.4681    0.4511    0.7035    0.4979
    3        -3.8807    0.1168    -0.1053    -0.1168    -0.0914    2.1488     5.9983    2.9304    10.6005    3.8361    0.4183    0       -0.0573    0.1244    0.1       6.5       0       -0.1003    -0.0015    -0.0025    3.7928    1.1271    455.7    3.347    0.312    300     2.1545    0         0.7245    0.6632    454.4      2.824      1.157     800       0.4838    0.3061    0.4617    0.4617    0.7006    0.4917
    """)

    CONSTS = {  'c2': 1.06,
                'c4': -2.1,
                'c4a': -0.5,
                'crb': 50.,
                }


class ChiouYoungs2013RegJPN(ChiouYoungs2013):
    """
    This implements the Chiou & Youngs (2013) GMPE
    """
    def _get_regional_terms(self, C):
        """
        Retrieve regional terms
        """
        reg = { 'gamma': C['g_JpIt'],
                'phi1': C['phi1_Jp'],
                'phi1a': C['phi1a_Jp'],
                'phi1b': C['phi1b_Jp'],
                'phi5': C['phi5_Jp'],
                'phi6': C['phi6_Jp'],
                'sig2': C['sig2_Jp']
                }
        return reg

    def _get_centered_z1pt0(self, sites):
        """
        Get z1pt0 centered on the Vs30- dependent average z1pt0(m)
        for Japan
        """
        mean_z1pt0 = (-5.23 / 2.) * np.log(((sites.vs30) ** 2. + 412. ** 2.)
                                           / (1360. ** 2. + 412. ** 2.))
        centered_z1pt0 = sites.z1pt0 - np.exp(mean_z1pt0)

        return centered_z1pt0


class ChiouYoungs2013RegTWN(ChiouYoungs2013):
    """
    This implements the Chiou & Youngs (2014) GMPE
    """
    def _get_regional_terms(self, C):
        """
        Retrieve regional terms
        """
        # get regional anelastic attenuation coefficient, gamma
        gamma = (C['cg1'] + C['cg2'] / (np.cosh(max(rup.mag - C['cg3'], 0.))))
        # collect regional terms into dictionary
        reg = { 'gamma': gamma,
                'phi1': C['phi1_Tw'],
                'phi1a': C['phi1a'],
                'phi1b': C['phi1b'],
                'phi5': C['phi5'],
                'phi6': C['phi6'],
                'sig2': C['sig2']
                }
        return reg


class ChiouYoungs2013RegCHN(ChiouYoungs2013):
    """
    This implements the Chiou & Youngs (2013) GMPE
    """
    def _get_regional_terms(self, C):
        """
        Retrieve regional terms
        """
        reg = { 'gamma': C['g_Wn'],
                'phi1': C['phi1'],
                'phi1a': C['phi1a'],
                'phi1b': C['phi1b'],
                'phi5': C['phi5'],
                'phi6': C['phi6'],
                'sig2': C['sig2']
                }
        return reg


class ChiouYoungs2013RegITA(ChiouYoungs2013):
    """
    This implements the Chiou & Youngs (2013) GMPE
    """
    def _get_regional_terms(self, C):
        """
        Retrieve regional terms
        """
        reg = { 'gamma': C['g_JpIt'],
                'phi1': C['phi1'],
                'phi1a': C['phi1a'],
                'phi1b': C['phi1b'],
                'phi5': C['phi5'],
                'phi6': C['phi6'],
                'sig2': C['sig2']
                }
        return reg
