//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ionfront.cpp
//  \brief Problem generator for ionization front tracking using expanding grid.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"

#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if (NSCALARS != 1)
#error: Requires NSCALARS = 1
#endif

static void stop_this();
//========================================================================================
// short for debugging interrupt
//========================================================================================
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//========================================================================================
// Time Dependent Grid Functions
//  \brief Functions for time dependent grid, including two example boundary conditions
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
void UpdateGridData(Mesh *pm);
//Global variables for GridUpdate
int ivexp, iweight, iequat = 0, nx1;
int maxntrack = 20;
int ncycold=-1;
Real vtrack0, boost, strack;
AthenaArray<Real> ttrack,rtrack;

//Global Variables for OuterX1
Real ambDens;
Real ambVel;
Real ambPres;
Real b0, bz0, angle;
Real alphab, nstar; // ionization parameter
void OuterX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

Real PowerGridX1(Real x, RegionSize rs);

// 
void IonizeSourceTerm(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                      const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);


//========================================================================================
//! \fn void WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData)
//  \brief Function that returns the velocity of cell wall i at location xf. Time, total
//  time step and direction are all given. Direction is one of 0,1,2, corresponding to x1,x2,x3
//  and gridData is an athena array that contains overall mesh data. gridData is updated
//  before every time sub-step by the UpdateGridData function. Some instances do not need
//  this data to be updated and the UpdateGridData function can be left blank. The gridData
//  array is supposed to carry all mesh-level information, i.e. the information used for 
//  multiple cell walls in the simulation.
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData) {
  Real retVal = 0.0;
  
 
  Real myX = xf;
  if (COORDINATE_SYSTEM == "spherical_polar") {
    if (dir != gridData(1)){
      retVal = 0.0;
    } else if (myX<=gridData(0)){
      retVal = 0.0;
    } else if (myX > gridData(0)){ 
      if (gridData(2)==0.0) retVal = 0.0;
      else retVal = gridData(2) * (myX-gridData(0))/(gridData(3)-gridData(0));
    } 
  }
  return retVal; 
}

//========================================================================================
//! \fn void UpdateGridData(Mesh *pm)
//  \brief Function which can edit and calculate any terms in gridData, which is used 
//  in the WallVel function. The object in mesh is GridData(i) and i can range over the
//  integers, limited by SetGridData argument in InitMeshUserData. See exp_blast for an 
//  example use of this function.
//  This is an example function, showing various options of expansion tracking.
//    iweight    ==  0: no weight
//    iweight    ==  1: density
//    iweight    ==  2: first scalar
//    iweight    ==  3: thermal pressure
//    iweight    ==  4: magnetic pressure
//    iweight    ==  5: temperature
//    iweight    == -1: density gradient
//    iweight    == -2: first scalar gradient
//    iweight    == -3: pressure gradient
//    iweight    == -4: magnetic pressure gradient
//    iweight    == -5: temperature gradient
//
//    iweight    == -15 etc: weight equatorial plane (assumed to be x, or x,z)
//  
//    ivexp      ==  0: expansion velocity set to constant vtrack0
//    ivexp      ==  1: expansion velocity via radial velocity
//    ivexp      ==  2: expansion velocity via radius: calculate velocity via finite differences
//
//  For hydrodynamics, iweight = -3 and ivexp = 2 (tracking on pressure gradient position)
//  works well, for MHD, iweight = -4 and ivexp = 2 keeps fast magnetosonic mode in box.
//  Both work well with boost = 1.0.
//
//  Since the expansion velocity is calculated at the location of the shell, it needs
//  to be rescaled by xmax/mrad, where mrad is the (weighted) radius corresponding to the 
//  location of vrad. 
//========================================================================================

void UpdateGridData(Mesh *pm) {
  MeshBlock *pmb = pm->pblock;
  Real vtrack = 0.0, mrad = 0.0;
  Real gamma  = pmb->peos->GetGamma();
  int ntr=0;

  pm->GridData(3) = pm->mesh_size.x1max;

  if (ivexp == 0) { // constant velocity 
    vtrack = vtrack0;
  } else { // if (ivexp == 0)
    AthenaArray<Real> weight, quant, radius;
    Real totweight = 0.0, totquant = 0.0, totradius = 0.0;
    while (pmb != NULL) {
      int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
      int il = is, iu = ie, jl = js, ju = je, kl = ks, ku = ke;
      //if (ks==ke) {
      //  kl = ks;
      //  ku = ke; 
      //} else {
      //  kl = ks-1;
      //  ku = ke+1;
      //}
      if (pmb->block_size.nx3 > 1) {
        weight.NewAthenaArray(pmb->block_size.nx3+2*NGHOST,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        quant.NewAthenaArray(pmb->block_size.nx3+2*NGHOST,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        radius.NewAthenaArray(pmb->block_size.nx3+2*NGHOST,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
      } else {
        weight.NewAthenaArray(1,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        quant.NewAthenaArray(1,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        radius.NewAthenaArray(1,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
      }
      if (ivexp == 1) { // radial velocity
        for (int k=kl; k<=ku; ++k) {
          Real z = pmb->pcoord->x3v(k);
          for (int j=jl; j<=ju; ++j) {
            Real y = pmb->pcoord->x2v(j);
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              Real x        = pmb->pcoord->x1v(i);
              quant(k,j,i)  = pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i);
              radius(k,j,i) = x;
            }
          }
        }
      } else if (ivexp == 2) { //radius
        for (int k=kl; k<=ku; ++k) {
          Real z = pmb->pcoord->x3v(k);
          for (int j=jl; j<=ju; ++j) {
            Real y = pmb->pcoord->x2v(j);
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              Real x    = pmb->pcoord->x1v(i);
              quant(k,j,i)  = x;
              radius(k,j,i) = x;
            }
          }
        }
      }

      if (fabs(iweight) == 0) { // no weight
        for (int k=kl; k<=ku; ++k) {
          for (int j=jl; j<=ju; ++j) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              weight(k,j,i) = 1.0;
            }
          }
        }
      } else if (fabs(iweight) == 1) { // density
        for (int k=kl; k<=ku; ++k) {
          for (int j=jl; j<=ju; ++j) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              weight(k,j,i) = pmb->phydro->u(IDN,k,j,i);
            }
          }
        }
      } else if (fabs(iweight) == 2) { // first scalar
        for (int k=kl; k<=ku; ++k) {
          for (int j=jl; j<=ju; ++j) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              weight(k,j,i) = pmb->phydro->u(NHYDRO,k,j,i);
            }
          }
        }
      } else if (fabs(iweight) == 3) { // thermal pressure: needed for 3 and 5
        if (DUAL_ENERGY) {
          for (int k=kl; k<=ku; ++k) {
            for (int j=jl; j<=ju; ++j) {
#pragma omp simd
              for (int i=il; i<=iu; ++i) {
                weight(k,j,i) = pmb->phydro->u(IIE,k,j,i);
              }
            }
          }
        } else {
          for (int k=kl; k<=ku; ++k) {
            for (int j=jl; j<=ju; ++j) {
#pragma omp simd
              for (int i=il; i<=iu; ++i) {
                Real ekin = 0.5*( SQR(pmb->phydro->u(IM1,k,j,i))
                                 +SQR(pmb->phydro->u(IM2,k,j,i))
                                 +SQR(pmb->phydro->u(IM3,k,j,i)))
                               / pmb->phydro->u(IDN,k,j,i);
                Real emag = 0.0;
                if (MAGNETIC_FIELDS_ENABLED) {
                  emag = 0.5*( SQR(pmb->pfield->bcc(IB1,k,j,i))
                              +SQR(pmb->pfield->bcc(IB2,k,j,i))
                              +SQR(pmb->pfield->bcc(IB3,k,j,i)));
                }
                weight(k,j,i) = pmb->phydro->u(IEN,k,j,i)-ekin-emag;
              }
            }
          }
        }
      } else if (fabs(iweight) == 4) { // magnetic pressure
        if (MAGNETIC_FIELDS_ENABLED) {
          for (int k=kl; k<=ku; ++k) {
            for (int j=jl; j<=ju; ++j) {
#pragma omp simd
              for (int i=il; i<=iu; ++i) {
                weight(k,j,i) = SQR(pmb->pfield->bcc(IB1,k,j,i))
                               +SQR(pmb->pfield->bcc(IB2,k,j,i))
                               +SQR(pmb->pfield->bcc(IB3,k,j,i));
              }
            }
          }
        }
      } else if (fabs(iweight) == 5) { // temperature
        if (DUAL_ENERGY) {
          for (int k=kl; k<=ku; ++k) {
            for (int j=jl; j<=ju; ++j) {
#pragma omp simd
              for (int i=il; i<=iu; ++i) {
                weight(k,j,i) = pmb->phydro->u(IIE,k,j,i)
                               /pmb->phydro->u(IDN,k,j,i);
              }
            }
          }
        } else {
          for (int k=kl; k<=ku; ++k) {
            for (int j=jl; j<=ju; ++j) {
#pragma omp simd
              for (int i=il; i<=iu; ++i) {
                Real ekin = 0.5*( SQR(pmb->phydro->u(IM1,k,j,i))
                                 +SQR(pmb->phydro->u(IM2,k,j,i))
                                 +SQR(pmb->phydro->u(IM3,k,j,i)))
                               / pmb->phydro->u(IDN,k,j,i);
                Real emag = 0.0;
                if (MAGNETIC_FIELDS_ENABLED) {
                  emag = 0.5*( SQR(pmb->pfield->bcc(IB1,k,j,i))
                              +SQR(pmb->pfield->bcc(IB2,k,j,i))
                              +SQR(pmb->pfield->bcc(IB3,k,j,i)));
                }
                weight(k,j,i) = (pmb->phydro->u(IEN,k,j,i)-ekin-emag)
                               /pmb->phydro->u(IDN,k,j,i);
              }
            }
          }
        }
      }

      // modify weight by angular dependence (here focusing on polar ejecta
      // since they are faster. Change angular dependence as needed
      if (iequat == 1) { 
        for (int k=kl; k<=ku; ++k) {
          Real z = pmb->pcoord->x3v(k);
          for (int j=jl; j<=ju; ++j) {
            Real y = pmb->pcoord->x2v(j);
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              Real x    = pmb->pcoord->x1v(i);
              Real rad, cthe2;
              if (COORDINATE_SYSTEM == "cartesian") {
                rad   = std::sqrt(SQR(x)+SQR(y)+SQR(z));
                cthe2 = std::pow(SQR(y/rad),strack); // cosine of polar angle
              } else { // this assumes spherical_polar
                rad   = x;
                cthe2 = std::pow(SQR(std::cos(y)),strack);
              }
              weight(k,j,i) *= cthe2;
            }
          }
        } 
      }

      if (iweight > 0) { // straight weights
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              Real q = quant(k,j,i);
              Real r = radius(k,j,i);
              Real v = pmb->pcoord->GetCellVolume(k,j,i);
              Real w = weight(k,j,i);
              w           *= v;
              q           *= w;
              r           *= w;
              totquant    += q;
              totweight   += w;
              totradius   += r;
            }
          }
        }
      } else { // if (iweight > 0): gradients
        // Calculate the gradient of the quantity, and use gradient as weight.
        if (pmb->block_size.nx3 > 1) { // 3D
          for (int k=ks+1; k<=ke-1; ++k) {
            Real z = pmb->pcoord->x3v(k);
            Real zm= pmb->pcoord->x3v(k-1);
            Real zp= pmb->pcoord->x3v(k+1);
            for (int j=js+1; j<=je-1; ++j) {
              Real y = pmb->pcoord->x2v(j);
              Real ym= pmb->pcoord->x2v(j-1);
              Real yp= pmb->pcoord->x2v(j+1);
#pragma omp simd
              for (int i=is+1; i<=ie-1; ++i) {
                Real x      = pmb->pcoord->x1v(i);
                Real xm     = pmb->pcoord->x1v(i-1);
                Real xp     = pmb->pcoord->x1v(i+1);
                Real gx     = (weight(k  ,j  ,i+1)-weight(k  ,j  ,i-1))/(xp-xm);
                Real gy     = (weight(k  ,j+1,i  )-weight(k  ,j-1,i  ))/(yp-ym);
                Real gz     = (weight(k+1,j  ,i  )-weight(k-1,j  ,i  ))/(zp-zm);
                Real r      = radius(k,j,i);
                Real g      = (gx*x+gy*y+gz*z)/r;
                Real q      = quant(k,j,i);
                Real w      = g > 0.0 ? 0.0 : -g;
                q          *= w;
                r          *= w;
                totquant   += q;
                totweight  += w;
                totradius  += r;
              }
            }
          }
        } else { // two dimensions
          for (int j=js+1; j<=je-1; ++j) {
            Real y = pmb->pcoord->x2v(j);
            Real ym= pmb->pcoord->x2v(j-1);
            Real yp= pmb->pcoord->x2v(j+1);
#pragma omp simd
            for (int i=is+1; i<=ie-1; ++i) {
              Real x        = pmb->pcoord->x1v(i);
              Real xm       = pmb->pcoord->x1v(i-1);
              Real xp       = pmb->pcoord->x1v(i+1);
              Real gx       = (weight(ks ,j  ,i+1)-weight(ks ,j  ,i-1))/(xp-xm);
              Real gy       = (weight(ks ,j+1,i  )-weight(ks ,j-1,i  ))/(yp-ym);
              Real r        = radius(ks,j,i);
              Real g        = (gx*x+gy*y)/r;
              Real q        = quant(ks,j,i);
              Real w        = g > 0.0 ? 0.0 : -g;
              q            *= w;
              r            *= w;
              totquant     += q;
              totweight    += w;
              totradius    += r;
            }
          }
        }
      } // if (iweight > 0)

      weight.DeleteAthenaArray();
      quant.DeleteAthenaArray();
      radius.DeleteAthenaArray();
      pmb = pmb->next;
    } // while (pmb != NULL)

    // now totquant and totweight contain the summed rad or vrad, and the appropriate normalization
#ifdef MPI_PARALLEL
    Real myval[3];
    myval[0] = totquant;
    myval[1] = totradius;
    myval[2] = totweight;
    MPI_Allreduce(MPI_IN_PLACE,&myval,3,MPI_ATHENA_REAL,MPI_SUM,
                  MPI_COMM_WORLD);
    totquant  = myval[0];
    totradius = myval[1];
    totweight = myval[2];
#endif
    totquant /= totweight;
    totradius/= totweight;

    if (ivexp == 1) { // use weighted radial velocity to determine vtrack
      vtrack  = totquant;
      vtrack  = ((vtrack < 0.0) ? 0.0 : vtrack);
      vtrack *= (pm->GridData(3)/totradius); // boost velocity
    } else if (ivexp == 2) { // use weighted radius to determine vtrack
      ntr = (pm->ncycle >= maxntrack) ? maxntrack-1 : pm->ncycle; // number of elements to track
      if (ncycold < pm->ncycle) { // do not update during substep
        ncycold = pm->ncycle;
        if (ntr == 0) { // first iteration
          ttrack(0) = 0.0;
          rtrack(0) = totquant;
          vtrack    = 0.0;
        } else { // all subsequent iterations
          for (int m=0; m<ntr; ++m) { // shift old elements 
            ttrack(ntr-m) = ttrack(ntr-m-1);
            rtrack(ntr-m) = rtrack(ntr-m-1);
          }
          ttrack(0) = pm->time; // add new element
          rtrack(0) = totquant;
        }
      }
      vtrack = 0.0;
      for (int m=1; m<ntr; ++m) { // calculate front velocity as average over tracking elements
        Real vt = (rtrack(m-1)-rtrack(m))/(ttrack(m-1)-ttrack(m)) * (pm->GridData(3)/totradius);
        vtrack += vt;
      }
      if (ntr > 0) vtrack /= ntr;
    }
    vtrack = (vtrack <= 0.0) ? 0.0 : vtrack; // enforce expansion
    if (Globals::my_rank == 0)
      fprintf(stdout,"[UpdateGrid] vtrack=%13.5e totradius=%13.5e totquant=%13.5e totweight=%13.5e xmax =%13.5e\n", vtrack,totradius,totquant,totweight,pm->GridData(3));
  } // if (ivexp == 0)

  vtrack *= boost;

  pm->GridData(2) = vtrack;
  if (COORDINATE_SYSTEM=="cartesian") {
    pm->GridData( 6) = vtrack;
    pm->GridData(10) = vtrack;
  }

  return;
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  //========================================================================================
  //! \brief For a time dependent grid, make sure to use SetGridData, EnrollGridDiffEq, and
  //   EnrollCalcGridData here. The boundary conditions are of course optional. Reflecting 
  //   is a good boundary function if a wall of the simulation is static. But if there is
  //   any expansion of the grid, it is recommended that you use the UniformMedium condition
  //   for the expanding boundary. Otherwise, reconstruction might fail because the data is
  //   inaccurate (for example, periodic boundary conditions do not make sense
  //   for an expanding grid).
  //========================================================================================

  if (COORDINATE_SYSTEM != "spherical_polar") {
    std::stringstream msg;
    msg << "[ionfront.cpp]: requires spherical_polar coordinates" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  EnrollUserExplicitSourceFunction(IonizeSourceTerm);
  Real x1rat = pin->GetOrAddReal("mesh","x1rat",1.0);
  nx1        = pin->GetInteger("mesh","nx1");
  if (x1rat < 0.0)
    EnrollUserMeshGenerator(X1DIR,PowerGridX1);

  if (EXPANDING_ENABLED) {
    EnrollGridDiffEq(WallVel);
      
    SetGridData(4);
    if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
      EnrollUserBoundaryFunction(OUTER_X1,OuterX1_UniformMedium);
    }

    EnrollCalcGridData(UpdateGridData);
    ttrack.NewAthenaArray(maxntrack); // for position tracking
    rtrack.NewAthenaArray(maxntrack);
    
    ambDens    = pin->GetReal("problem","damb");
    ambVel     = 0.0;
    ambPres    = pin->GetReal("problem","pamb");
    ivexp      = pin->GetInteger("problem","ivexp"); // see UpdateGridData
    iweight    = pin->GetInteger("problem","iweight"); // see UpdateGridData
    boost      = pin->GetOrAddReal("problem","boost",1.0); // enhancement of vtrack
    strack     = pin->GetOrAddReal("problem","strack",1.0); // focuses tracking to polar angle by cos(theta)**(2*strack)

    // add angular weight scheme
    if (fabs(iweight) > 10) {
      iweight += (iweight > 0 ? -10 : 10);
      iequat   = 1;
    }

    if (ivexp == 0) {
      vtrack0 = pin->GetReal("problem","vtrack0"); // constant tracking velocity for test purposes
    }

    GridData(0) = mesh_size.x1min;
    GridData(1) = 1; 
    GridData(2) = 0.0;
    GridData(3) = mesh_size.x1max; 
  }

  return;
}

//========================================================================================
//! \fn void OuterX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for outer boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================
void OuterX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,ie+i) = ambDens;
        prim(IPR,k,j,ie+i) = ambPres;
        prim(IVX,k,j,ie+i) = 0.0;
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = 0.0;
        prim(IS0,k,j,ie+i) = 0.0; // ionization degree 
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) { // only spherical-polar
    Real theta, phi;
    for (int k=ks; k<=ke; ++k) {
      phi = pco->x3v(k);
      for (int j=js; j<=je; ++j) {
        theta = pco->x2v(j);
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,ie+i+1) = b0 * std::abs(std::sin(theta))
                                 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
        }
      }
    }

    for (int k=ks; k<=ke; ++k) {
      phi = pco->x3v(k);
      for (int j=js; j<=je+1; ++j) {
        theta = pco->x2v(j);
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,ie+i) = b0 * std::cos(theta)
                               * (   std::cos(angle) * std::cos(phi)
                                   + std::sin(angle) * std::sin(phi));
          if (std::sin(theta) < 0.0)
            b.x2f(k,j,ie+i) *= -1.0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      phi = pco->x3v(k);
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,ie+i) = b0 * (   std::sin(angle) * std::cos(phi)
                                   - std::cos(angle) * std::sin(phi));
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn Real PowerGridX1(Real x, RegionSize rs)
//  \brief Generates grid following r_i = r_0*(rmax/rmin)**x, with 0<=x<=1
//========================================================================================
Real PowerGridX1(Real x, RegionSize rs) {
  Real delta = rs.x1max/rs.x1min;
  Real r     = rs.x1min*pow(delta,x);
  return r;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//    Here: sets the initial condition for ionization front in 1D. Ionization 
//    source is assumed to be located beyond the inner (radial) boundary. No attenuation
//    until grid starts.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real pa     = pin->GetReal("problem","pamb");
  Real da     = pin->GetReal("problem","damb");
  alphab      = pin->GetReal("problem","alphab"); //recombination rate
  nstar       = pin->GetReal("problem","nstar");  // ionizing photon rate 
  if (MAGNETIC_FIELDS_ENABLED) {
    b0 = pin->GetReal("problem","b0");
    bz0= pin->GetOrAddReal("problem","bz0",0.0);
    angle = (PI/180.0)*pin->GetReal("problem","angle");
    if (COORDINATE_SYSTEM == "spherical_polar")
      bz0 = 0.0;
  }
  Real gamma = peos->GetGamma(); 
  Real gm1 = gamma - 1.0;
 
  if (Globals::my_rank==0) {
    fprintf(stdout,"IDN=%2i IVX=%2i IVY=%2i IVZ=%2i IPR=%2i IBY=%2i IBZ=%2i NHYDRO-SCALARS=%2i NHYDRO=%2i NWAVE=%2i\n",IDN,IVX,IVY,IVZ,IPR,IBY,IBZ,NHYDRO-NSCALARS,NHYDRO,NWAVE);
  }

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad;
        rad = pcoord->x1v(i);
        Real den = da;
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          Real pres = pa;
          phydro->u(IEN,k,j,i) = pres/gm1;
          if (DUAL_ENERGY) {
            phydro->u(IIE,k,j,i) = pres/gm1;
          }
        }
        // ionization degree
        phydro->u(IS0,k,j,i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie+1; ++i) {
          Real theta = pcoord->x2v(j);
          Real phi = pcoord->x3v(k);
          pfield->b.x1f(k,j,i) = b0 * std::abs(std::sin(theta))
              * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real theta = pcoord->x2v(j);
          Real phi = pcoord->x3v(k);
          pfield->b.x2f(k,j,i) = b0 * std::cos(theta)
              * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          if (std::sin(theta) < 0.0)
            pfield->b.x2f(k,j,i) *= -1.0;
        }
      }
    }
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real phi = pcoord->x3v(k);
          pfield->b.x3f(k,j,i) =
              b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          phydro->u(IEN,k,j,i) += 0.5*(SQR(b0)+SQR(bz0));
        }
      }
    }
  } 

  return;
}

//====================================================================================

void Mesh::UserWorkInLoop(void) {

  MeshBlock *pmb=pblock;

  const int nq = 7;
  Real qtot[nq]; // 0: vol, 1: dens, 2: vtot, 3: etot, 4: eint, 5: ekin, 6: emag
  Real qmin[nq];
  Real qmax[nq];
  Real ener[nq];
  Real trac[3];
  Real lengrat[2];

  for (int q=0; q<nq; q++) {
    qtot[q] = 0.0;
    qmin[q] = (FLT_MAX);
    qmax[q] = -(FLT_MAX);
    ener[q] = 0.0;
  }
  for (int q=0; q<3; q++)
    trac[q] = 0.0;
  for (int q=0; q<2; q++)
    lengrat[q] = (FLT_MAX);
  Real u[NHYDRO];

  while (pmb != NULL) { // collect results from individual pmbs
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      Real x3  = pmb->pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; j++) {
        Real x2  = pmb->pcoord->x2v(j);
        for (int i=pmb->is; i<=pmb->ie; i++) {
          if (DUAL_ENERGY) {
            if (   isnan(pmb->phydro->u(IEN,k,j,i))
                || isnan(pmb->phydro->u(IDN,k,j,i))
                || isnan(pmb->phydro->u(IIE,k,j,i))
                || (pmb->phydro->u(IEN,k,j,i) <= 0.0)
                || (pmb->phydro->u(IDN,k,j,i) <= 0.0)
                || (pmb->phydro->u(IIE,k,j,i) <= 0.0)) {
              std::cout << "[UserWorkInLoop]: Warning: i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
                        << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IDN,k,j,i)
                        << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM1,k,j,i)
                        << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM2,k,j,i)
                        << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM3,k,j,i)
                        << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IEN,k,j,i)
                        << " ei=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IIE,k,j,i)
                        << std::endl;
            }
          } else {
            if (   isnan(pmb->phydro->u(IEN,k,j,i))
                || isnan(pmb->phydro->u(IDN,k,j,i))
                || (pmb->phydro->u(IEN,k,j,i) <= 0.0)
                || (pmb->phydro->u(IDN,k,j,i) <= 0.0)) {
              std::cout << "[UserWorkInLoop]: Warning: i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
                        << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IDN,k,j,i)
                        << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM1,k,j,i)
                        << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM2,k,j,i)
                        << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM3,k,j,i)
                        << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IEN,k,j,i)
                        << std::endl;
            }
          }
          Real dx1  = pmb->pcoord->dx1f(i);
          Real x1  = pmb->pcoord->x1v(i);
          Real dvol = pmb->pcoord->GetCellVolume(k,j,i);

          //energy
          for (int q=0; q<NHYDRO; q++)
            u[q] = pmb->phydro->u(q,k,j,i);
          qtot[0] += dvol;
          ener[1]  = u[IDN];
          ener[2]  = u[IEN];
          ener[4]  = 0.5*(SQR(u[IM1])+SQR(u[IM2])+SQR(u[IM3]))/u[IDN];
          if (MAGNETIC_FIELDS_ENABLED)
            ener[5] = 0.5*(  SQR(pmb->pfield->b.x1f(k,j,i))
                           + SQR(pmb->pfield->b.x2f(k,j,i))
                           + SQR(pmb->pfield->b.x3f(k,j,i)));
          if (DUAL_ENERGY) {
            ener[6] = u[IIE];
          }
          ener[3] = ener[2]-ener[4]-ener[5];
          for (int q=1; q<nq; q++) {
            qtot[q] += ener[q]*dvol;
            if (ener[q] < qmin[q]) qmin[q] = ener[q];
            if (ener[q] > qmax[q]) qmax[q] = ener[q];
          }

          //shell tracking
          Real rad = std::sqrt(x1*x1+x2*x2+x3*x3);
          trac[0] += u[IDN];
          trac[1] += (u[IM1]*x1+u[IM2]*x2+u[IM3]*x3)/rad;
          trac[2] += u[IDN]*rad;
//          Real temp = gm1*ener[2]/u[IDN];
//          if (CoolingFunc != NULL)
//            lengrat[0] = std::min(lengrat[0],temp*std::sqrt(temp)/(dx1*fabs(CoolingFunc(u[IDN],temp)))); // cooling length
//          lengrat[1] = std::min(lengrat[1],std::sqrt(PI*temp/u[IDN])/dx1); // Jeans length
        }
      }
    }
    pmb = pmb->next;
  }

#ifdef MPI_PARALLEL
  int ierr;
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qtot,nq,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qmin,nq,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qmax,nq,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&trac,3 ,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&lengrat,2,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
#endif
  for (int q=1; q<nq; q++) qtot[q] /= qtot[0];
  for (int q=1; q<3;  q++) trac[q] /= trac[0];

  if (Globals::my_rank==0) {
//    std::cout << "[UserWorkInLoop]: lcool= " << std::scientific << std::setw(13) << std::setprecision(5) << lengrat[0]
//              << " lgrv= "                   << std::scientific << std::setw(13) << std::setprecision(5) << lengrat[1]
//              << std::endl;
    std::cout << "[UserWorkInLoop]: vrad = " << std::scientific << std::setw(13) << std::setprecision(5) << trac[1]
              << " rad = "                   << std::scientific << std::setw(13) << std::setprecision(5) << trac[2]
              << std::endl;
    std::cout << "[UserWorkInLoop]: dens = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[1]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[1]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[1]
              << std::endl;
    std::cout << "[UserWorkInLoop]: etot = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[2]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[2]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[2]
              << std::endl;
    std::cout << "[UserWorkInLoop]: eint0= " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[3]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[3]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[3]
              << std::endl;
    std::cout << "[UserWorkInLoop]: ekin = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[4]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[4]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[4]
              << std::endl;
    if (MAGNETIC_FIELDS_ENABLED)
      std::cout << "[UserWorkInLoop]: emag = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[5]
                << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[5]
                << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[5]
                << std::endl;
    if (DUAL_ENERGY)
      std::cout << "[UserWorkInLoop]: eint1= " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[6]
                << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[6]
                << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[6]
                << std::endl;

  }

  if (qmin[3] <= 0.0) {
    std::cout << "[UserWorkInLoop]: eint < 0" << std::endl;
    stop_this();
  }

  return;
}

//========================================================================================
//! \fn void IonizeSourceTerm(...)
//  \brief Ionization Source Term
//========================================================================================

void IonizeSourceTerm(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                      const AthenaArray<Real> &bcc, AthenaArray<Real> &cons) {

  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke; 

  return;
}
