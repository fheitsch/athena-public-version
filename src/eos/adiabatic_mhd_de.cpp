//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adiabatic_mhd.cpp
//  \brief implements functions in class EquationOfState for adiabatic MHD, 
//    dual energy version

// C++ headers
#include <cmath>   // sqrt()
#include <cfloat>  // FLT_MIN

// Athena++ headers
#include "eos.hpp"
#include "../hydro/hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"

// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) {
  pmy_block_ = pmb;
  gamma_ = pin->GetReal("hydro", "gamma");
  ieta1_ = pin->GetOrAddReal("hydro", "ieta1",1e-3);
  ieta2_ = pin->GetOrAddReal("hydro", "ieta2",1e-1);
  density_floor_  = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*(FLT_MIN)));
  pressure_floor_ = pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024*(FLT_MIN)));
}

// destructor

EquationOfState::~EquationOfState() {
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//    const AthenaArray<Real> &prim_old, const FaceField &b,
//    AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//    int il, int iu, int jl, int ju, int kl, int ku);
// \brief For the Hydro, converts conserved into primitive variables in adiabatic MHD.
//  For the Field, computes cell-centered from face-centered magnetic field.

void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
    const AthenaArray<Real> &prim_old, const FaceField &b, AthenaArray<Real> &prim,
    AthenaArray<Real> &bcc, Coordinates *pco,
    int il, int iu, int jl, int ju, int kl, int ku) {
  Real gm1 = GetGamma() - 1.0;
  Real i1 = GetIeta1();

  pmy_block_->pfield->CalculateCellCenteredField(b,bcc,pco,il,iu,jl,ju,kl,ku);

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      Real& u_d  = cons(IDN,k,j,i);
      Real& u_m1 = cons(IVX,k,j,i);
      Real& u_m2 = cons(IVY,k,j,i);
      Real& u_m3 = cons(IVZ,k,j,i);
      Real& u_e  = cons(IEN,k,j,i);
      Real& u_ie = cons(IIE,k,j,i);

      Real& w_d  = prim(IDN,k,j,i);
      Real& w_vx = prim(IVX,k,j,i);
      Real& w_vy = prim(IVY,k,j,i);
      Real& w_vz = prim(IVZ,k,j,i);
      Real& w_p  = prim(IPR,k,j,i);
      Real& w_ge = prim(IGE,k,j,i);

      //if (!RECOVER_ENABLED) {
        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
      //}
      w_d = u_d;

      Real di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      const Real& bcc1 = bcc(IB1,k,j,i);
      const Real& bcc2 = bcc(IB2,k,j,i);
      const Real& bcc3 = bcc(IB3,k,j,i);

      Real pb = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
      Real ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1*(u_e - ke - pb);
      w_ge= gm1*u_ie;

      //Use dual-energy 
      if (((w_p/u_e) < i1) || (u_e <= 0.0) || (w_p <= 0.0) || isnan(w_p) || isnan(u_e)) {
        // Do not use temperature, but pressure instead (same as standard energy fh211001)
        w_p = w_ge;
        u_e = u_ie + ke + pb;
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
          Real& w_s = prim(n,k,j,i);
          w_s = u_s/u_d;
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
//        Note that this function assumes cell-centered fields are already calculated

void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
     int il, int iu, int jl, int ju, int kl, int ku) {
  Real igm1 = 1.0/(GetGamma() - 1.0);

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma omp simd
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

      const Real& bcc1 = bc(IB1,k,j,i);
      const Real& bcc2 = bc(IB2,k,j,i);
      const Real& bcc3 = bc(IB3,k,j,i);

      u_d  = w_d;
      u_m1 = w_vx*w_d;
      u_m2 = w_vy*w_d;
      u_m3 = w_vz*w_d;
      u_e  = w_p*igm1 + 0.5*(  w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz))
                             +     (SQR(bcc1) + SQR(bcc2) + SQR(bcc3)));
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
//   IGE is pressure - we use here the "safe" dual energy branch
Real EquationOfState::SoundSpeed(const Real prim[NHYDRO+2]) {
  return std::sqrt(GetGamma()*prim[IGE]/prim[IDN]);
}

//----------------------------------------------------------------------------------------
// \!fn Real EquationOfState::FastMagnetosonicSpeed(const Real prim[], const Real bx)
// \brief returns fast magnetosonic speed given vector of primitive variables
// Note the formula for (C_f)^2 is positive definite, so this func never returns a NaN
// Is used differently than in adiabatic_mhd.cpp. Passes NHYDRO+2, not NWAVE, to 
// account for scalars and dual energy.
// NHYDRO = IDN,IVX,IVY,IVZ,IPR,IGE,NSCALARS
// Last two elements in PRIM are IBY, IBZ (see athena.hpp)
Real EquationOfState::FastMagnetosonicSpeed(const Real prim[(NHYDRO+2)], const Real bx) {
  Real asq = GetGamma()*prim[IGE];
  Real vaxsq = bx*bx;
  Real ct2 = (prim[IBY]*prim[IBY] + prim[IBZ]*prim[IBZ]);
  Real qsq = vaxsq + ct2 + asq;
  Real tmp = vaxsq + ct2 - asq;
  return std::sqrt(0.5*(qsq + std::sqrt(tmp*tmp + 4.0*asq*ct2))/prim[IDN]);
}

//---------------------------------------------------------------------------------------
// \!fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim,
//           int k, int j, int i)
// \brief Apply density and pressure floors to reconstructed L/R cell interface states
void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,k,j,i);
  //if (!RECOVER_ENABLED) {
    // apply density floor
    w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  //}

  return;
}
