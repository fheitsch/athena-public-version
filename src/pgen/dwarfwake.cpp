//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dwarfwake.cpp
//  \brief Problem generator for a dark matter sub-halo traveling through the MW gas halo
//

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <bits/stdc++.h>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../fft/athena_fft.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/utils.hpp"
// extra headers
#include "../hydro/srcterms/hydro_srcterms.hpp"


#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#if (NSCALARS != 2)
#error: coolcloud requires NSCALARS==2
#endif


// Cooling variables. These need to be set in InitUserMeshData (restarts!!)
Real n0, T0, gm1, v0, grav_acc, pcool, dtcool = HUGE_NUMBER;

typedef Real (*CoolingFunc_t)(const Real dens, const Real temp);

//====================================================================================
// local functions
Real CoolingFuncShull(const Real dens, const Real temp); // heating and cooling
Real CoolingFuncSlyz(const Real dens, const Real temp);
Real CoolingFuncSlyzMod(const Real dens, const Real temp);
Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real dt);
Real BracketRoot(const Real dens, const Real temp0, const Real dt);
Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt);
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);
void ProjectPressureInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh);
void ProjectPressureOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh);
void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);
CoolingFunc_t CoolingFunc; //type declaration, a pointer to a function
Real gravpot_darkhalo(const Real x1, const Real x2, const Real x3, const Real time);
static void stop_this();

//====================================================================================
// short for debugging interrupt
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//====================================================================================
// Real CoolingFuncShull(const Real dens, const Real temp)
//   cooling function from Shull & Moss 2020.
//   Returns de/dt [erg/s] in code units assuming ISM units with n0=T0=1.
//   Modified to provide constant temperature in center ("exclusion") region.
Real CoolingFuncShull(const Real dens, const Real temp) {
  const Real Lambda0 = 2e-22;
  const Real fac = 2.167177868e+31;
  const Real nenH = 1.165/SQR(2.247);
  const Real Tref = 1e6;
  const Real Ts   = 1e5;
  // from Shull & Moss
  Real Lambda = Lambda0 * std::pow(temp/Tref,-0.7);
  //Real dedt   = -dens*nenH*Lambda*fac;
  // modified Shull & Moss to allow for equilibrium (cutoff at 10^5)
  const Real Gamma0 = n0*Lambda0*std::pow(Ts/Tref,1.3); 
  Real Gamma = Gamma0*std::pow(Tref/temp,3.0);
  Real dedt = nenH*(Gamma-dens*Lambda)*fac;
  return dedt;
}

//====================================================================================
// Real CoolingFuncSlyz(const Real dens, const Real temp)
//   Cooling function from Slyz et al. 2005 (piece-wise powerlaw fit)
//   Returns de/dt [ergs/s] in code units assuming ISM units with n0=T0=1.
Real CoolingFuncSlyz(const Real dens, const Real temp) {
  const Real fac = 2.167177868e+31;
  Real lambda, dedt;
  if (temp < 3.1e2) {
    lambda = 0.0;
  } else if (temp < 2.0e3) {
    lambda = 2.2380e-32*SQR(temp);
  } else if (temp < 8.0e3) {
    lambda = 1.0012e-30*std::pow(temp,1.5);
  } else if (temp < 3.9811e4) {
    lambda = 4.6240e-36*std::pow(temp,2.867);
  } else if (temp < 1.0e5) {
    lambda = 3.162e-30*std::pow(temp,1.6);
  } else if (temp < 2.884e5) {
    lambda = 3.162e-21*std::pow(temp,-0.2);
  } else if (temp < 4.732e5) {
    lambda = 6.3100e-6*std::pow(temp,-3.0);
  } else if (temp < 2.113e6) {
    lambda = 1.047e-21*std::pow(temp,-0.22);
  } else if (temp < 3.981e6) {
    lambda = 3.981e-4*std::pow(temp,-3.0);
  } else if (temp < 1.995e7) {
    lambda = 4.169e-26*std::pow(temp,0.33);
  } else {
    lambda = 2.399e-27*std::sqrt(temp);
  }
  dedt = -dens*lambda*fac;
  return dedt;
}

//====================================================================================
Real CoolingFuncSlyzMod(const Real dens, const Real temp) {
  const Real fac = 2.167177868e+31;
  Real gam, lambda, dedt;
  int itemp;
  
  std::valarray<Real> temprange = {3.0000e+00, 5.0000e+01, 1.0000e+03, 8.0000e+03, 3.9811e+04, 4.0000e+04,
  9.0000e+04, 1.5000e+05, 2.6500e+05, 7.0000e+05, 1.0000e+07};

  std::valarray<Real> coeff = {1.89059200e-31, 4.72647999e-28, 1.18723810e-25, 4.62400000e-36,
  3.16200000e-30, 3.16200000e-21, 2.61576294e-12, 2.61576294e-12,
  5.08130713e-15, 3.98372334e-26, 7.94857304e-27};

  std::valarray<Real> expo = { 3., 1., 0.2, 2.867, 1.6, -0.2, -2., -2., -1.5, 0.4, 0.5};
  
  Real lt;
  std::valarray<Real> ltr;
  std::valarray<Real> lc;

  lt     = std::log(temp);
  ltr    = std::log(temprange);
  lc     = std::log(coeff);
    
  if (lt < ltr[0]) { // no cooling bc below lower cut-off
    lambda = 0.0;
  } else if (lt > ltr[10]) {
    lambda =  std::exp(lc[10]+expo[10]*lt);
  } else {
    itemp = 0;
    while (ltr[itemp] < lt) {
      itemp = itemp+1;
    }
    lambda = std::exp(lc[itemp-1]+expo[itemp-1]*lt); 
  }
  if (temp < 5e3) {
    gam = 2e-25;
  } else {
    gam = 2e-25*(5e3/temp);
  } 
  dedt = (gam - dens*lambda)*fac;
  return dedt;
}

//====================================================================================
// Real RootFunc(const Real dens, const Real temp0, const Real, temp1, const Real dt)
// This is not the thermal equilibrium, but the implicit update for the thermal ODE
//   dT = dt*(gamma-1)/kB * (Gamma(T)-n*Lambda(T))
//
Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real dt) {
  return temp0 + dt*gm1*CoolingFunc(dens,temp1) - temp1;
}

//====================================================================================
// Real BracketRoot(const Real temp0, const Real dt)
Real BracketRoot(const Real dens, const Real temp0, const Real dt) {
  Real rf    = RootFunc(dens,temp0,temp0,dt);
  Real sig   = (Real) ((rf > 0) - (rf < 0));
  Real fac   = 1.0 + sig*0.1;
  Real temp1 = temp0;
  while (rf*RootFunc(dens,temp0,temp1,dt) > 0)
    temp1 *= fac;
  return temp1;
}

//====================================================================================
// Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt)
Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt) {
  if (CoolingFunc(dens,temp0) == 0.0) return temp0; // Nothing to do for thermal equilibrium
  // Otherwise, temp1 and temp0 bracket the temperature down to which we should integrate.
  const Real tol = 1e-6;
  int nit = (int) (log(fabs(temp1-temp0)/tol)/log(2.0));
  Real T[3], L[2];
  T[0]         = temp0;
  T[1]         = temp1;
  T[2]         = 0.5*(T[0]+T[1]);
  L[0]         = RootFunc(dens,temp0,T[0],dt);
  L[1]         = RootFunc(dens,temp0,T[2],dt);
  for (int i=0; i<nit; i++) {
    int w = (L[0]*L[1] < 0); // 0 if >0, 1 if <= 0
    T[w]  = T[2];
    L[w]  = L[1];
    T[2]  = 0.5*(T[0]+T[1]);
    L[1]  = RootFunc(dens,temp0,T[2],dt);
  }
  return T[2];
}


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {

  int icool, ipot;

  icool    = pin->GetInteger("problem","icool"); // 0: none , 1: Slyz
  ipot     = pin->GetInteger("problem","ipot"); // 0: none, 1: static dwarf potential
  gm1   = pin->GetReal("hydro","gamma")-1.0;
  pcool = pin->GetReal("problem","pcool");
  grav_acc = pin->GetOrAddReal("hydro","grav_acc3",0.0);

  if (icool != 0) {
    EnrollUserExplicitSourceFunction(HeatCool);
    EnrollUserTimeStepFunction(HeatCoolTimeStep);
  }
   
  if (ipot == 1) {
    EnrollStaticGravPotFunction(gravpot_darkhalo);
  }

  if (grav_acc != 0.0) {
    EnrollUserBoundaryFunction(INNER_X3, ProjectPressureInnerX3);
    EnrollUserBoundaryFunction(OUTER_X3, ProjectPressureOuterX3);
  }
  
  if (v0 != 0.0) {
    EnrollUserBoundaryFunction(INNER_X1, InflowBoundary);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Setup for dwarfwake problem.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  int iprob,icool;
  Real x0, y0, z0, dtc = HUGE_NUMBER;
  std::stringstream msg;
  Real avg[2], my_avg[2];
#ifdef MPI_PARALLEL
  int mpierr, myid = Globals::my_rank;
  Real my_dtc;
#endif

  iprob    = pin->GetInteger("problem","iprob"); // 0: constant density, 1: gaussian perturbation
  icool    = pin->GetInteger("problem","icool"); // 0: Shull \& Moss, 1: Slyz
  n0       = pin->GetOrAddReal("problem","n0",1.0); // background density
  T0       = pin->GetOrAddReal("problem","T0",1.0); // background temperature
  v0       = pin->GetOrAddReal("problem","v0",0.0); // wind x-velocity
  x0       = pin->GetOrAddReal("problem","x0",0.0); // x-center of cloud
  y0       = pin->GetOrAddReal("problem","y0",0.0); // y-center of cloud
  z0       = pin->GetOrAddReal("problem","z0",0.0); // z-center of cloud


  // Set the cooling function
  if (icool == 0) {
    CoolingFunc = CoolingFuncShull; 
  } else if (icool == 1) {
    CoolingFunc = CoolingFuncSlyz;
  } else if (icool == 2) {
    CoolingFunc = CoolingFuncSlyzMod;
  } else {
    msg << "[dwarfwake]: icool must have values 0, 1, or 2. " << icool << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // Constant density with x-velocity wind
  if (iprob == 0) { 
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IDN,k,j,i) = n0;
          phydro->u(IM1,k,j,i) = v0*n0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          
          phydro->u(IEN,k,j,i) = n0*T0/gm1 + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                                  +SQR(phydro->u(IM2,k,j,i))
                                                  +SQR(phydro->u(IM3,k,j,i)))
                                                /phydro->u(IDN,k,j,i);
          phydro->u(IIE,k,j,i) = n0*T0/gm1;
        }
      }
    } 
  }

// Hydrostatic density profile within a dwarf satellite DM potential
// with a background x-velocity wind  ***EDIT THIS FUNCTION*** need tanh profile to make dwarf gas not move
  if (iprob == 1) {
    const Real a = 2.275e3; 
    for (int k=ks; k<=ke; k++) {
      Real z = pcoord->x3v(k) ;
      for (int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x = pcoord->x1v(i);
          Real r = std::sqrt(SQR(x-x0)+SQR(y-y0)+SQR(z-z0));
          phydro->u(IDN,k,j,i) = 0.0858*std::exp(-7.448*(1-std::log(1+r/a)/(r/a)));
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
         
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)*T0/gm1 + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                                  +SQR(phydro->u(IM2,k,j,i))
                                                  +SQR(phydro->u(IM3,k,j,i)))
                                                /phydro->u(IDN,k,j,i);
          phydro->u(IIE,k,j,i) = phydro->u(IDN,k,j,i)*T0/gm1;
        }
      }
    }
  }

  // Set cooling time over all Meshblocks
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real dens0 = phydro->u(IDN,k,j,i);
        Real temp0 = phydro->u(IIE,k,j,i)*gm1/dens0;
        dtc = std::min(dtc,temp0/(gm1*(fabs(CoolingFunc(dens0,temp0))+1e-60)));
      }
    }
  }
#ifdef MPI_PARALLEL
  my_dtc = dtc;
  mpierr = MPI_Allreduce(&my_dtc, &dtc, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[coolcloud]: MPI_Allreduce error = " << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
#endif
  dtcool = dtc;
  //if (myid == 0) {
    //std::cout << "[coolcloud]: dtcool = " << std::scientific << std::setprecision(5) << dtcool << std::endl;
  //}

}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {

}

//========================================================================================
//! \fn void heatcool(...)
//  \brief Heating and cooling for user-defined cooling function
//  Implicit solution of ODE for temperature change (see RootFunc)
//========================================================================================

void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  //if (dtcool == HUGE_NUMBER) // for the first iteration.
  //  dtcool = 0.25*dt;    // global variable for timestep, to be sent to HeatCoolTimeStep 
  //  dtcool = HUGE_NUMBER;
  Real g1  = pmb->peos->GetGamma()-1.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real dens  = prim(IDN,k,j,i);
        Real temp0 = prim(IGE,k,j,i)*g1; // This should have gam-1, bc it's the specific internal energy
        if (temp0 <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in dwarfwake.cpp: HeatCool: temp0 <=0 for DUAL_ENERGY" << std::endl
              << "    p=" << std::setw(5) << Globals::my_rank << " i=" << std::setw(5) << i << " j=" << std::setw(5) << j << " k=" << std::setw(5) << k << std::endl
              << "    temp0=" << std::scientific << std::setw(13) << std::setprecision(5) << temp0
              << "    dens=" << std::scientific << std::setw(13) << std::setprecision(5) << dens << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
        Real temp1       = BracketRoot(dens,temp0,dt);
        Real temp2       = FindRoot(dens,temp0,temp1,dt);
        Real dener       = dens*(temp2-temp0);
        dtcool           = std::min(dtcool,temp0/(fabs(CoolingFunc(dens,temp0))+1e-60));
        cons(IEN,k,j,i) += dener/g1;
        cons(IIE,k,j,i) += dener/g1;

        //Real dedt        = CoolingFunc(dens,temp0);
        //Real dener       = dt*dens*dedt;
        //dtcool           = std::min(dtcool,temp0/(g1*fabs(dedt)+1e-60));
        //cons(IEN,k,j,i) += dener;
        //cons(IIE,k,j,i) += dener;
        fprintf(stdout,"[HeatCool]: i,j,k=%4i%4i%4i ien=%13.5e iie=%13.5e dener=%13.5e dtcool=%13.5e\n",i,j,k,cons(IEN,k,j,i),cons(IIE,k,j,i),dener,dtcool);

      }
    }
  }
  fprintf(stdout,"[HeatCool]: time = %13.5e dt = %13.5e dtcool = %13.5e\n",time,dt,dtcool);
  return;
}

//========================================================================================
//! \fn void HeatCoolTimeStep(...)
//  \brief Calculates cooling timestep and sends it to new_blockdt
//    Geometric mean between cfl timestep and cooling time, dt = dtcfl^(1-p) * dtcool^p,
//    if dtcool < dtcfl.
//========================================================================================

Real HeatCoolTimeStep(MeshBlock *pmb)
{
  Real dt = pmb->pmy_mesh->dt;
  return dt*std::min(1.0,pow(dtcool/dt,pcool));
}


//========================================================================================
//// Inflow boundary condition
//// Inputs:
////   pmb: pointer to MeshBlock
////   pcoord: pointer to Coordinates
////   is,ie,js,je,ks,ke: indices demarkating active region
//// Outputs:
////   prim: primitives set in ghost zones
//========================================================================================

void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // Set hydro variables
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        prim(IDN,k,j,i) = n0;
        prim(IPR,k,j,i) = n0*T0;
        prim(IVX,k,j,i) = v0;
        prim(IVY,k,j,i) = 0.0;
        prim(IVZ,k,j,i) = 0.0;
        prim(IGE,k,j,i) = T0/gm1;
      }
    }
  }
  return;
} 



//========================================================================================
//! \fn void ProjectPressureInnerX3()
//  \brief  Pressure is integated into ghost cells to improve hydrostatic eqm
//========================================================================================
//
//
void ProjectPressureInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      if (n==(IVZ)) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IVZ,ks-k,j,i) = -prim(IVZ,ks+k-1,j,i);  // reflect 3-vel
        }
      } else if (n==(IPR)) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IPR,ks-k,j,i) = prim(IPR,ks+k-1,j,i)
             - prim(IDN,ks+k-1,j,i)*grav_acc*(2*k-1)*pco->dx3f(k);
        }
      } else {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,ks-k,j,i) = prim(n,ks+k-1,j,i);
        }
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b3

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ks-k),j,i) =  b.x1f((ks+k-1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ks-k),j,i) =  b.x2f((ks+k-1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ks-k),j,i) = -b.x3f((ks+k  ),j,i);  // reflect 3-field
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProjectPressureOuterX3()
//  \brief  Pressure is integated into ghost cells to improve hydrostatic eqm

void ProjectPressureOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      if (n==(IVZ)) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IVZ,ke+k,j,i) = -prim(IVZ,ke-k+1,j,i);  // reflect 3-vel
        }
      } else if (n==(IPR)) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IPR,ke+k,j,i) = prim(IPR,ke-k+1,j,i)
             + prim(IDN,ke-k+1,j,i)*grav_acc*(2*k-1)*pco->dx3f(k);
        }
      } else {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,ke+k,j,i) = prim(n,ke-k+1,j,i);
        }
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b3

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ke+k  ),j,i) =  b.x1f((ke-k+1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ke+k  ),j,i) =  b.x2f((ke-k+1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ke+k+1),j,i) = -b.x3f((ke-k+1),j,i);  // reflect 3-field
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn static gravpot_darkhalo
//  \brief spherical DM potential, centered at (0,0,0) 
//  ISM units
//----------------------------------------------------------------------------------------
static Real gravpot_darkhalo(const Real x1, const Real x2, const Real x3, const Real time){
  Real M200 = 1e10; //solar mass
  const Real h = 0.673; 
  Real logc200 = (0.905 - 0.101*log10(M200/(1e12/h))); 
  Real c200 = std::pow(10,logc200);
  Real delta_c = (200.0/3)*std::pow(c200,3)/(log(1+c200)-c200/(1+c200));
  const Real rho_crit = (1e-29)/(1.677e-24); // 10^29 g/cm^3 --> ISM units
  Real rho_s = rho_crit*delta_c; //  ISM density unit
  Real r200 = std::pow((M200/16.84)/(200.0*rho_crit*(4*M_PI/3)), 1.0/3); // ISM length unit
  Real rs = r200/c200; // ISM length unit
  Real r = std::sqrt(SQR(x1)+SQR(x2)+SQR(x3));
  Real Phi = (-4*M_PI*rho_s*std::pow(rs,3)*log(1+r/rs)/r);//*0.5*(1.0-std::tanh((r-r200)/(0.1*r200)));
  return Phi;
}


