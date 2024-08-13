
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements functions in class EquationOfState for adiabatic hydrodynamics`
//    Version for dual energy

// C/C++ headers
#include <cmath>   // sqrt()
#include <cfloat>  // FLT_MIN
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>


// Athena++ headers
#include "eos.hpp"
#include "../hydro/hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"

// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) {
  pmy_block_ = pmb;
  gamma_ = pin->GetReal("hydro", "gamma");
  ieta1_ = pin->GetOrAddReal("hydro", "ieta1",1e-3);
  ieta2_ = pin->GetOrAddReal("hydro", "ieta2",1e-1);
  density_floor_  = pin->GetOrAddReal("hydro","dfloor", std::sqrt(1024*(FLT_MIN)));
  pressure_floor_ = pin->GetOrAddReal("hydro","pfloor", std::sqrt(1024*(FLT_MIN)));
}

// destructor

EquationOfState::~EquationOfState() {
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//           const AthenaArray<Real> &prim_old, const FaceField &b,
//           AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.

void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old, const FaceField &b, AthenaArray<Real> &prim,
  AthenaArray<Real> &bcc, Coordinates *pco, int il,int iu, int jl,int ju, int kl,int ku) {
  Real gm1 = GetGamma() - 1.0;
  Real i1 = GetIeta1();

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      Real& u_d  = cons(IDN,k,j,i);
      Real& u_m1 = cons(IM1,k,j,i);
      Real& u_m2 = cons(IM2,k,j,i);
      Real& u_m3 = cons(IM3,k,j,i);
      Real& u_e  = cons(IEN,k,j,i); // total energy
      Real& u_ie = cons(IIE,k,j,i); // internal energy

      Real& w_d  = prim(IDN,k,j,i);
      Real& w_vx = prim(IVX,k,j,i);
      Real& w_vy = prim(IVY,k,j,i);
      Real& w_vz = prim(IVZ,k,j,i);
      Real& w_p  = prim(IPR,k,j,i); // pressure, from total energy
      Real& w_ge = prim(IGE,k,j,i); // pressure, pdV track
				
      //if (!RECOVER_ENABLED) {
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
      //}
      w_d = u_d;

      Real di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      Real ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1*(u_e - ke);
      w_ge= gm1*u_ie;

      // Use dual-energy 
      if (((w_p/u_e) < i1) || (u_e <= 0.0) || (w_p <= 0.0) || isnan(w_p) || isnan(u_e)) {
        // Do not use temperature, but pressure instead (same as standard energy fh211001)
        w_p = w_ge; 
        u_e = u_ie + ke; 
      }
    }
  }}

  // passive scalars 
  for (int n=(NHYDRO-NSCALARS); n<NHYDRO; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real& u_s = cons(n  ,k,j,i);
          Real& u_d = cons(IDN,k,j,i);
          Real   di = 1./u_d; 
          Real& w_s = prim(n,k,j,i);
          w_s = u_s*di;
        }
      }
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
//           const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
//           int il, int iu, int jl, int ju, int kl, int ku);
// \brief Converts primitive variables into conservative variables

void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
     int il, int iu, int jl, int ju, int kl, int ku) {
  Real igm1 = 1.0/(GetGamma() - 1.0);

  // Force outer-loop vectorization
#pragma omp simd
  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma novector
    for (int i=il; i<=iu; ++i) {
      Real& u_d  = cons(IDN,k,j,i);
      Real& u_m1 = cons(IM1,k,j,i);
      Real& u_m2 = cons(IM2,k,j,i);
      Real& u_m3 = cons(IM3,k,j,i);
      Real& u_e  = cons(IEN,k,j,i);
      Real& u_ie = cons(IIE,k,j,i);

      const Real& w_d  = prim(IDN,k,j,i);
      const Real& w_vx = prim(IVX,k,j,i);
      const Real& w_vy = prim(IVY,k,j,i);
      const Real& w_vz = prim(IVZ,k,j,i);
      const Real& w_p  = prim(IPR,k,j,i);
      const Real& w_ge = prim(IGE,k,j,i);

      u_d = w_d;
      u_m1 = w_vx*w_d;
      u_m2 = w_vy*w_d;
      u_m3 = w_vz*w_d;
      u_e  = w_p*igm1 + 0.5*w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
      u_ie = w_ge*igm1; // use internal branch pressure instead 
    }
  }}

  // passive scalars 
  for (int n=(NHYDRO-NSCALARS); n<NHYDRO; ++n) { 
#pragma omp simd
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma novector
        for (int i=il; i<=iu; ++i) {
          Real& u_s = cons(n,k,j,i);

          const Real& w_s = prim(n  ,k,j,i);
          const Real& w_d = prim(IDN,k,j,i);

          u_s = w_s*w_d; 
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
// \!fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
// \brief returns adiabatic sound speed given vector of primitive variables

Real EquationOfState::SoundSpeed(const Real prim[NHYDRO]) {
  return std::sqrt(gamma_*prim[IGE]/prim[IDN]); // use internal branch pressure
}

//---------------------------------------------------------------------------------------
// \!fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim,
//           int k, int j, int i)
// \brief Apply density and pressure floors to reconstructed L/R cell interface states
//   Pressure floor should not be applied when using dual energy. (fh211001)

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,k,j,i);

  //if (!RECOVER_ENABLED) {
    // apply density floor
    w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  //}

  return;
}
