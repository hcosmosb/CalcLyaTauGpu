#include "math.h"
double sig_a_app(double dl_a_Angst, double T_in) {
    // Constants
    const double c_sl = 2.99792e10;  // speed of light in cm/s
    const double k_boltz = 1.380648e-16;  // Boltzmann constant in erg/K
    const double m_p = 1.6726219e-24;  // proton mass in g
    const double pi = M_PI;  // Pi from math.h
    const double l_a_Angst = 1215.67;  // Lyman-alpha wavelength in Angstroms

    // Calculations
    double a_V = 4.7e-4 * std::pow(T_in / 1e4, -0.5);
    double dnu_a = c_sl / ((dl_a_Angst + l_a_Angst) * 1e-8) - c_sl / (l_a_Angst * 1e-8);
    double Delnu_D = 2.46e15 * std::sqrt(2.0 * k_boltz * T_in / (m_p * c_sl * c_sl));

    double x = std::abs(dnu_a / Delnu_D);
    double x2 = x * x;
    double z = (x2 - 0.855) / (x2 + 3.42);

    double q;
    if (z <= 0) {
        q = 0.0;
    } else {
        q = z * (1 + 21 / x2) * a_V / pi / (x2 + 1) * (0.1117 + z * (4.421 + z * (-9.207 + 5.674 * z)));
    }

    double phi_app = q + std::exp(-x2) / std::sqrt(pi);

    // Final result
    double sig_a_app_result = phi_app * 5.889e-14 * std::pow(T_in / 1e4, -0.5);

    return sig_a_app_result;
}
