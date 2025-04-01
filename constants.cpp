#include "constants.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "math.h"
#include <omp.h> 

bool isCoDaIII=true;
std::string SIMDIR, OUTDIR;
const double h = 0.6777, Om = 0.307, Ob = 0.0482;
const double Grav = 6.67e-8, Mpc = 3.086e24, c_sl = 2.99792e10, lLya = 1215.67, m_p = 1.67262192e-24, k_boltz = 1.380648e-16;

int isn;
int ndim, nchdim, nchunk;
double Lbox;
double z;

double cell, cell_cm;
double hubbleRate, rhoCrit, nHmean, NHCellMean, dlPerKmps, dvaCell;
double unit_l, unit_d, unit_t;
double vconv, tconv;



int checkAndProcessInputs(){
    if (SIMDIR.empty()){ std::cerr<<"# Please define SIMDIR."<<std::endl; return 1;}
    if (OUTDIR.empty()){ std::cerr<<"# Please define OUTDIR."<<std::endl; return 1;}
    if (!isn){ std::cerr<<"# Please define isn."<<std::endl; return 1;}
    if (!Lbox){ std::cerr<<"# Please define Lbox."<<std::endl; return 1;}
    if (!ndim){ std::cerr<<"# Please define ndim."<<std::endl; return 1;}
    cell = Lbox/ndim;
    if (!nchdim){ std::cerr<<"# Please define nchdim."<<std::endl; return 1;}
    nchunk = pow(ndim/nchdim,3);
    if (!h)  { std::cerr<<"# Please define h."  <<std::endl; return 1;}
    if (!z)  { std::cerr<<"# Please define z."  <<std::endl; return 1;}
    cell_cm = Lbox/ndim*Mpc/h/(1.+z);
    hubbleRate = 1e2*pow(1.-Om + Om*pow(1.+z, 3),0.5);
    dvaCell = hubbleRate/(1.+z)*cell;
    dlPerKmps = lLya*(1e5/c_sl);
    rhoCrit = 3*pow(h*1e7/Mpc,2)/(8*M_PI*Grav);
    nHmean = rhoCrit*pow(1.+z,3)*Ob*0.76/m_p;
    NHCellMean = nHmean*Mpc*cell/h/(1+z);
    
    if (!unit_l){ std::cerr<<"# Please define unit_l."<<std::endl; return 1;}
    if (!unit_d){ std::cerr<<"# Please define unit_d."<<std::endl; return 1;}
    if (!unit_t){ std::cerr<<"# Please define unit_t."<<std::endl; return 1;}    
    vconv = unit_l/unit_t/1e5;tconv = m_p*pow(unit_l/unit_t,2)/k_boltz;   
    return 0;
}

int printConstants(){
    checkAndProcessInputs();
    std::cout<<"######### Printing key cosmological quantities #########"<<std::endl;
    std::cout<<"# Redshift: "<<z<<std::endl;
    std::cout<<"# Hubble rate: "<<hubbleRate/(1+z)<<" km/s per cMpc/h"<<std::endl;
    std::cout<<"# Cell size: "<<cell<<" cm"<<std::endl;
    std::cout<<"# dva per cell: "<<dvaCell<<" km/s per cell"<<std::endl;
    std::cout<<"# rho_crit: "<<rhoCrit<<" g/cm^3"<<std::endl;
    std::cout<<"# Mean H number density: "<<nHmean<<" per cm^3"<<std::endl;
    std::cout<<"# H column density of a mean density cell: "<<NHCellMean<<" cm^-2"<<std::endl;
    std::cout<<"# dl per km/s: "<<dlPerKmps<<" Å"<<std::endl;
    std::cout<<"# vconv: "<<vconv<<"; tconv: "<<tconv<<std::endl;
    std::cout<<"########################################################"<<std::endl;
    return 0;
}




