//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file exp_advect.cpp
//  \brief Advection test for expanding grid. 
//  Needs to run on a domain -2, 2. Profile centered on -1 with width 1, moving to the right
//  at v=1.0. After t=2, profile is centered on +1. 
//
//========================================================================================

// C headers
#include <stdio.h>

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../cless/cless.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"


Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
void UpdateGridData(Mesh *pm);
void OuterX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void InnerX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

//====================================================================================
// Implement Expanding Functions

//Wall Velocity. Depending on direction and GridData and time, return the velocity at xf, which
//will be the location of the ith cell wall at time time
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData) {
  Real retVal = 0.0;
  Real x0l = gridData(3);
  Real x0r = gridData(5);
  Real myX = xf;
  //std::cout << gridData(0) << std::endl;
  if (dir != gridData(1)){
    retVal = 0.0;
  } else if (myX==gridData(0)){
    retVal = 0.0;
  } else if (myX < gridData(0)){
    if (gridData(2)==0.0) retVal = 0.0;
    else retVal = gridData(2) * (gridData(0)-myX)/(gridData(0)-x0l);
  } else if (myX > gridData(0)){ 
    if (gridData(4)==0.0) retVal = 0.0;
    else retVal = gridData(4) * (myX-gridData(0))/(x0r-gridData(0));
  } 
  return retVal; 
}

//====================================================================================

void UpdateGridData(Mesh *pm) {
  Real xMin;
  Real xMax;
  xMin = pm->mesh_size.x1min;
  xMax = pm->mesh_size.x1max;
  pm->GridData(1) = 1;
  pm->GridData(3) = xMin;
  pm->GridData(5) = xMax;
  return;
}

//====================================================================================

void OuterX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IVX,k,j,ie+i) = 1.0;
        prim(IDN,k,j,ie+i) = 1.0;
        prim(IPR,k,j,ie+i) = 1.0;  
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = 0.0;
      }
    }
  }

  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(ie+i)) = 1.0;  
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(ie+i)) = 1.0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(ie+i)) =  0.0;
        }
      }
    }
  }
  return;

}

//====================================================================================

void InnerX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IVX,k,j,is-i) = 1.0;
        prim(IDN,k,j,is-i) = 1.0;
        prim(IPR,k,j,is-i) = 1.0;  
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;
      }
    }
  }

  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(is-i)) = 1.0;  
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(is-i)) = 1.0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(is-i)) = 0.0;
        }
      }
    }
  }
  return;

}

//====================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  
  if (EXPANDING_ENABLED) {
    SetGridData(6);
    EnrollGridDiffEq(WallVel);
    EnrollCalcGridData(UpdateGridData);
    GridData(1) = pin->GetReal("problem","shock_dir");
    Real xMin = pin->GetReal("mesh","x1min");
    Real xMax = pin->GetReal("mesh","x1max");
    Real n = (Real)(pin->GetInteger("mesh","nx1"));
    GridData(0) = 0.0; //(xMin+xMax)/n*0.5;
    GridData(2) = -0.5;
    GridData(3) = xMin;
    GridData(4) = 0.5;
    GridData(5) = xMax;
  }
    
  if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X1,OuterX1_UniformMedium);
  }
  if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(INNER_X1,InnerX1_UniformMedium);
  }
    
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the shock tube tests
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;

//--- shock in 1-direction
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1v = pcoord->x1v(i);
        phydro->u(IDN,k,j,i) = 1.0+0.25*(1.0+std::tanh((x1v+1.5)/0.01))*(1.0-std::tanh((x1v+0.5)/0.01));
        phydro->u(IM1,k,j,i) = 1.0*phydro->u(IDN,k,j,i);
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) 
          phydro->u(IEN,k,j,i) =   1.0/(peos->GetGamma() - 1.0) 
                                 + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                        +SQR(phydro->u(IM2,k,j,i))
                                        +SQR(phydro->u(IM3,k,j,i)))
                                      /phydro->u(IDN,k,j,i);
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          pfield->b.x1f(k,j,i) = 1.0;
          pfield->b.x2f(k,j,i) = 1.0;
          if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN,k,j,i) += 0.5*(SQR(pfield->b.x1f(k,j,i))
            + SQR(pfield->b.x2f(k,j,i)) + SQR(pfield->b.x3f(k,j,i)));
          }
        }
      }
    }

    // end by adding bi.x1 at ie+1, bi.x2 at je+1, and bi.x3 at ke+1

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pfield->b.x1f(k,j,ie+1) = pfield->b.x1f(k,j,ie);
    }}
    for (int k=ks; k<=ke; ++k) {
    for (int i=is; i<=ie; ++i) {
      pfield->b.x2f(k,je+1,i) = pfield->b.x2f(k,je,i);
    }}
    for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
      pfield->b.x3f(ke+1,j,i) = pfield->b.x3f(ke,j,i);
    }}
  }
  return;
}

//====================================================================================

void Mesh::UserWorkInLoop(void) { 
  return;
}
