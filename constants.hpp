#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <string>
#include <vector>

extern bool isCoDaIII;
extern std::string SIMDIR, OUTDIR;
extern const double h, Om, Ob;
extern const double Grav, Mpc, c_sl, lLya, m_p, k_boltz;

extern int isn;
extern int ndim, nchdim, nchunk;
extern double Lbox;
extern double z;

extern double rhoCrit, hubbleRate;
extern double cell, cell_cm;
extern double dvaCell, dlPerKmps;
extern double nHmean, NHCellMean;
extern double unit_l, unit_d, unit_t;
extern double vconv, tconv;

int printConstants();

#endif
