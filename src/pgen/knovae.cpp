//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file knovae.cpp
//  \brief Problem generator for spherical knova  problem.  
//
//

// C++ headers
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <iomanip>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

//====================================================================================
// global variables
FILE *otffile;

//========================================================================================
// Time Dependent Grid Functions
//  \brief Functions for time dependent grid, including two example boundary conditions
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
void UpdateGridData(Mesh *pm);
//Global variables for GridUpdate
int ivexp, iweight;
int maxntrack = 20;
int ncycold=-1;
Real vtrack0, boost,x1rat;
AthenaArray<Real> ttrack,rtrack;

//Global Variables for OuterX1
int ibtype;
Real ambdens, ambvel, ambpres, drat, prat;
Real b0, bx0, by0, bz0, angle;
// The expanding grid requires user-defined boundary functions only for axes
// along which expansion is possible. 
void OuterX1_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void OuterX2_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void OuterX3_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void InnerX1_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void InnerX2_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void InnerX3_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void OuterX1_Spherical(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void InnerX1_Spherical(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);


//====================================================================================
// local functions
Real LogMeshSpacingX1(Real x, RegionSize rs);

void ReflectInnerX1_nonuniform(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

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
  Real retval = 0.0;

  if (COORDINATE_SYSTEM == "cartesian") {
    if (dir == gridData(1)){
      if ((xf > 0.0)&&(gridData(3)>0.0)){
        if (gridData(2)==0.0) retval = 0.0;
        else retval = gridData(2) * xf/gridData(3);
      } else if ((xf < 0.0)&&(gridData(0)<0.0)){
        if (gridData(2) == 0.0) retval = 0.0;
        else retval = -1.0*gridData(2) * xf/gridData(0);
      }
    } else if (dir == gridData(5)) {
      if ((xf > 0.0)&&(gridData(7)>0.0)){
        if (gridData(6)==0.0) retval = 0.0;
        else retval = gridData(6) * xf/gridData(7);
      } else if ((xf < 0.0)&&(gridData(4)<0.0)){
        if (gridData(6) == 0.0) retval = 0.0;
        else retval = -1.0*gridData(6) * xf/gridData(4);
      }
    } else if (dir == gridData(9)) {
      if ((xf > 0.0)&&(gridData(11)>0.0)){
        if (gridData(10)==0.0) retval = 0.0;
        else retval = gridData(10) * xf/gridData(11);
      } else if ((xf < 0.0)&&(gridData(8)<0.0)){
        if (gridData(10) == 0.0) retval = 0.0;
        else retval = -1.0*gridData(10) * xf/gridData(8);
      }
    }
  } else if (COORDINATE_SYSTEM == "cylindrical") {
    if (dir != gridData(1)){
      retval = 0.0;
    } else if (xf<=gridData(0)){
      retval = 0.0;
    } else if (xf > gridData(0)){
      if (gridData(2)==0.0) retval = 0.0;
      else retval = gridData(2) * (xf-gridData(0))/(gridData(3)-gridData(0));
    }
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    if (dir == gridData(1)) {
      if (gridData(2) != 0.0) {
        Real x;
        if (x1rat < 0.0) { // for PowerGridX1, the velocities must be adapted (p. 63)
          x      = std::log(xf/gridData(0))/std::log(gridData(3)/gridData(0));
          retval = x * std::pow(gridData(3)/gridData(0),x-1.0) * gridData(2);
          //vf = s*(r1/r0)**(s-1)*vex
        } else {
          x      = (xf-gridData(0))/(gridData(3)-gridData(0));
          retval = gridData(2) * x;
        }
      }
    }
  }
  return retval;
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
//    iweight    ==  5: vrad
//    iweight    == -1: density gradient
//    iweight    == -2: first scalar gradient
//    iweight    == -3: pressure gradient
//    iweight    == -4: magnetic pressure gradient
//    iweight    == -5: vrad gradient
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
  Real vtrack = 0.0;
  Real gamma  = pmb->peos->GetGamma();
  int ntr=0;

  if (COORDINATE_SYSTEM == "cartesian") {
    pm->GridData(3)  = pm->mesh_size.x1max;
    pm->GridData(0)  = pm->mesh_size.x1min;
    pm->GridData(7)  = pm->mesh_size.x2max;
    pm->GridData(4)  = pm->mesh_size.x2min;
    pm->GridData(11) = pm->mesh_size.x3max;
    pm->GridData(8)  = pm->mesh_size.x3min;
  } else {
    pm->GridData(3) = pm->mesh_size.x1max;
  }
  if (ivexp == 0) { // constant velocity 
    vtrack = vtrack0;
  } else { // if (ivexp == 0)
    AthenaArray<Real> weight, quant, radius;
    Real totweight = 0.0, totquant = 0.0, totradius = 0.0;
    while (pmb != NULL) {
      int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
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
        for (int k=ks; k<=ke; ++k) {
          Real z = pmb->pcoord->x3v(k);
          for (int j=js; j<=je; ++j) {
            Real y = pmb->pcoord->x2v(j);
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              Real x    = pmb->pcoord->x1v(i);
              Real vrad, rad;
              if (COORDINATE_SYSTEM == "cartesian") {
                rad  = std::sqrt(SQR(x)+SQR(y)+SQR(z));
                vrad =  (  pmb->phydro->u(IM1,k,j,i)*x
                         + pmb->phydro->u(IM2,k,j,i)*y
                         + pmb->phydro->u(IM3,k,j,i)*z)
                       / (pmb->phydro->u(IDN,k,j,i)*rad);
              } else {
                vrad = pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i);
              }
              quant(k,j,i)  = vrad;
              radius(k,j,i) = rad;
            }
          }
        }
      } else if (ivexp == 2) { //radius
        for (int k=ks; k<=ke; ++k) {
          Real z = pmb->pcoord->x3v(k);
          for (int j=js; j<=je; ++j) {
            Real y = pmb->pcoord->x2v(j);
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              Real x    = pmb->pcoord->x1v(i);
              Real rad;
              if (COORDINATE_SYSTEM == "cartesian") {
                rad  = std::sqrt(SQR(x)+SQR(y)+SQR(z));
              } else {
                rad = x;
              }
              quant(k,j,i)  = rad;
              radius(k,j,i) = rad;
            }
          }
        }
      }

      if (fabs(iweight) == 0) { // no weight
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              weight(k,j,i) = 1.0;
            }
          }
        }
      } else if (fabs(iweight) == 1) { // density
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              weight(k,j,i) = pmb->phydro->u(IDN,k,j,i);
            }
          }
        }
      } else if (fabs(iweight) == 2) { // first scalar
         for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              weight(k,j,i) = pmb->phydro->u(NHYDRO-NSCALARS,k,j,i);
            }
          }
        }
      } else if (fabs(iweight) == 3) { // thermal pressure
        if (DUAL_ENERGY) {
          for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
                weight(k,j,i) = pmb->phydro->u(IIE,k,j,i);
              }
            }
          }
        } else {
          for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
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
          for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
                weight(k,j,i) = SQR(pmb->pfield->bcc(IB1,k,j,i))
                               +SQR(pmb->pfield->bcc(IB2,k,j,i))
                               +SQR(pmb->pfield->bcc(IB3,k,j,i));
              }
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
                Real q      = quant(k,j,i);
                Real r      = radius(k,j,i);
                Real w      = std::fabs((gx*x+gy*y+gz*z)/r);
                q          *= w;
                r          *= w;
                totquant   += q;
                totweight  += w;
                totradius  += r;
              }
            }
          }
        } else { // two dimensions
          if (COORDINATE_SYSTEM == "cartesian") {
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
                Real q        = quant(ks,j,i);
                Real r        = radius(ks,j,i);
                Real w        = std::fabs((gx*x+gy*y)/r);
                q            *= w;
                r            *= w;
                totquant     += q;
                totweight    += w;
                totradius    += r;
              }
            }
          } else if (COORDINATE_SYSTEM == "spherical_polar") {
            for (int j=js; j<=je; ++j) { // only radial gradient here, hence use whole j range
#pragma omp simd
              for (int i=is+1; i<=ie-1; ++i) {
                Real gr    =   (weight(ks ,j, i+1)-weight(ks ,j, i-1))
                              /(pmb->pcoord->x1v(i+1)-pmb->pcoord->x1v(i-1));
                Real q     = quant(ks,j,i);
                Real r     = radius(ks,j,i);
                Real w     = std::fabs(gr);
                q         *= w;
                r         *= w;
                totquant  += q;
                totweight += w;
                totradius += r;
              }
            }
          } else {
            fprintf(stdout,"Cylindrical coordinates not supported for gradient tracking.\n");
            stop_this();
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
    //if (Globals::my_rank == 0)
    //  fprintf(stdout,"[UpdateGrid] vtrack=%13.5e rtrack=%13.5e xmax =%13.5e\n", vtrack,totquant,pm->GridData(3));
  } // if (ivexp == 0)

  vtrack *= boost;

  pm->GridData(2) = vtrack;
  if (COORDINATE_SYSTEM=="cartesian") {
    pm->GridData( 6) = vtrack;
    pm->GridData(10) = vtrack;
  }

  return;
}

// DOWN TO HERE.

//====================================================================================
// Enroll user-specific functions
void Mesh::InitUserMeshData(ParameterInput *pin) {
  Real x1rat = pin->GetOrAddReal("mesh","x1rat",0.0);
  
  if (x1rat < 0.0) {
    EnrollUserMeshGenerator(X1DIR, LogMeshSpacingX1);
    if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
      EnrollUserBoundaryFunction(INNER_X1, ReflectInnerX1_nonuniform);
    }
  }
  return;
}

//====================================================================================
// "Logarithmic" (power-law) mesh
// Note that the grid setup asks for the **local** x1min, x1max etc. 
// x is the "logical" position in the grid, with the logical grid runing
// from 0 to 1, i.e. x = i/Nx
Real LogMeshSpacingX1(Real x, RegionSize rs) {
  Real xf, xrat;
  xrat   = pow(rs.x1max/rs.x1min,1.0/((Real) rs.nx1)); // Only valid for fixed grid, no MPI
  xf     = rs.x1min*pow(xrat,x*rs.nx1); // x = i/Nx
  return xf;
}

//========================================================================================
// Reflecting inner X1 boundary conditions for radially non-uniform grids

void ReflectInnerX1_nonuniform(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, reflecting v1
  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVX)) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(IVX,k,j,is-i) = -prim(IVX,k,j,(is+i-1));  // reflect 1-velocity
          }
        }
      }
    } else {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,is-i) = prim(n,k,j,(is+i-1));
          }
        }
      }
    }
  }
  // copy face-centered magnetic fields into ghost zones, reflecting b1
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(is-i)) = -b.x1f(k,j,(is+i  ));  // reflect 1-field
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(is-i)) =  b.x2f(k,j,(is+i-1));
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(is-i)) =  b.x3f(k,j,(is+i-1));
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real b0=0.0, angle=0.0;
  Real pi         = 4.0*std::atan(1.0);
  Real r0         = pin->GetReal("problem","r0"); // radius of initial ejecta
  Real dr         = pin->GetReal("problem","dr"); // width of transition
  Real p0         = pin->GetReal("problem","p0"); // ambient pressure
  Real d0         = pin->GetReal("problem","d0"); // ambient density
  Real v0         = pin->GetReal("problem","v0"); // velocity of ejecta
  Real m0         = pin->GetReal("problem","m0"); // mass of ejecta
  Real E0         = pin->GetReal("problem","E0"); // total energy of ejecta
  int  ihomol     = pin->GetOrAddInteger("problem","ihomol",0); // initial homologous expansion (no "ring")
  Real thopen     = pin->GetOrAddReal("problem","thopen",pi); // opening angle of disk ejecta. pi is full polar coverage
  Real rwindtidev = pin->GetOrAddReal("problem","rwindtidev",1.0); // ratio between wind and tidal ejecta speed.
  Real rwindtiden = pin->GetOrAddReal("problem","rwindtiden",1.0); // ratio between wind and tidal ejecta density.
  int  ilog       = pin->GetOrAddInteger("mesh", "ilog", 0);
  Real x1min      = pin->GetReal("mesh","x1min");
  if ((!ihomol) && (x1min >= 0.5*r0)) { // for ring, make sure that x1min < r0/2
    std::stringstream msg;
    msg << "### FATAL ERROR in knovae.cpp ProblemGenerator" << std::endl
        << "x1min > 0.5*r0 " << x1min << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    b0    = pin->GetReal("problem","b0");
    angle = (PI/180.0)*pin->GetReal("problem","angle");
  }
  Real gamma   = peos->GetGamma();
  Real gm1     = gamma - 1.0;
  Real d1      = 3.0*m0/(4.0*pi*pow(r0,3));
  Real e1      = 3.0*E0/(4.0*pi*pow(r0,3));
  Real e0      = p0/gm1;

  // get coordinates of center of blast, and convert to Cartesian if necessary
  Real x1_0   = pin->GetOrAddReal("problem","x1_0",0.0);
  Real x2_0   = pin->GetOrAddReal("problem","x2_0",0.0);
  Real x3_0   = pin->GetOrAddReal("problem","x3_0",0.0);
  Real x0,y0,z0;
  if (COORDINATE_SYSTEM == "cartesian") {
    x0 = x1_0;
    y0 = x2_0;
    z0 = x3_0;
  } else if (COORDINATE_SYSTEM == "cylindrical") {
    x0 = x1_0*std::cos(x2_0);
    y0 = x1_0*std::sin(x2_0);
    z0 = x3_0;
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    x0 = x1_0*std::sin(x2_0)*std::cos(x3_0);
    y0 = x1_0*std::sin(x2_0)*std::sin(x3_0);
    z0 = x1_0*std::cos(x2_0);
  } else {
    // Only check legality of COORDINATE_SYSTEM once in this function
    std::stringstream msg;
    msg << "### FATAL ERROR in knovae.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad,v0rad,radv,thet;
        if (COORDINATE_SYSTEM == "cartesian") {
          Real x   = pcoord->x1v(i);
          Real y   = pcoord->x2v(j);
          Real z   = pcoord->x3v(k);
          rad      = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          thet   = std::acos(z/rad);
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          Real z = pcoord->x3v(k);
          rad    = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          thet   = std::acos(z/rad);
        } else { // if (COORDINATE_SYSTEM == "spherical_polar")
          Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          rad    = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          thet   = pcoord->x2v(j);
        }

        // factors for radial and polar dependence
        Real rho2              = d1;
        Real rho1              = rho2*rwindtiden;
        Real rho0              = d0;
        Real fm1rad            = 0.5*(1.0-std::tanh((rad-r0)/dr)); // radial drop off
        Real fm1phi            = 0.25*((1.0+std::tanh((thet-0.5*(pi-thopen))/0.1))*(1.0-std::tanh((thet-0.5*(pi+thopen))/0.1))); // polar drop off
        if (ilog) {
          radv = (rad-x1min)/(r0-x1min);
        } else {
          radv = rad;
        }
        if (ihomol) { // ramp to make ring
          v0rad = v0*rad/r0;
        } else {
          if (rad <= 0.5*r0) {
            v0rad         = 2.0*v0*rad/r0;
          } else {
            v0rad         = v0;
          }
        }
        //phydro->u(IDN,k,j,i) = d0+(d1-d0)*fm1rad*fm1phi;
        phydro->u(IDN,k,j,i) = rho0+(rho1-rho0)*fm1rad+(rho2-rho1)*fm1phi*fm1rad; // that correct?
        phydro->u(IM1,k,j,i) = (v0rad*(1.0-rwindtidev)*fm1rad*fm1phi+v0rad*rwindtidev*fm1rad)*phydro->u(IDN,k,j,i);
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          //phydro->u(IEN,k,j,i) = p0/gm1 + 0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) = e0+(e1-e0)*0.5*(1.0-std::tanh((rad-r0)/dr));
          if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
            phydro->u(IEN,k,j,i) += d0;
        }
        if (DUAL_ENERGY) {
          phydro->u(IIE,k,j,i) = e0+(e1-e0)*0.5*(1.0-std::tanh((rad-r0)/dr));
        }
        if (NSCALARS == 2) {
          // the first index corresponds to tidal ejecta, the second to wind ejecta
          phydro->u(NHYDRO-NSCALARS  ,k,j,i) = rho2*fm1rad*fm1phi;       // tidal ejecta
          phydro->u(NHYDRO-NSCALARS+1,k,j,i) = rho1*fm1rad*(1.0-fm1phi); // wind ejecta
        }
      }
    }
  }

  // initialize interface B and total energy
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie+1; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            pfield->b.x1f(k,j,i) = b0 * std::cos(angle);
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real phi = pcoord->x2v(j);
            pfield->b.x1f(k,j,i) =
                b0 * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x1f(k,j,i) = b0 * std::abs(std::sin(theta))
                * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            pfield->b.x2f(k,j,i) = b0 * std::sin(angle);
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real phi = pcoord->x2v(j);
            pfield->b.x2f(k,j,i) =
                b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x2f(k,j,i) = b0 * std::cos(theta)
                * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              pfield->b.x2f(k,j,i) *= -1.0;
          }
        }
      }
    }
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
            pfield->b.x3f(k,j,i) = 0.0;
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real phi = pcoord->x3v(k);
            pfield->b.x3f(k,j,i) =
                b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          phydro->u(IEN,k,j,i) += 0.5*b0*b0;
        }
      }
    }
  }

  // only the root process outputs the data
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign("knovae.otf");
    std::stringstream msg;
    if ((otffile = fopen(fname.c_str(),"wb")) == NULL) {
      msg << "### FATAL ERROR in function [Meshblock::ProblemGenerator]"
          << std::endl << "knovae.otf could not be opened" <<std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  }

}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief otf diagnostics (shell position, sphericity etc)
//========================================================================================
void MeshBlock::UserWorkInLoop(void) {
#ifdef MPI_PARALLEL
  int mpierr;
#endif
  if (NSCALARS == 2) {
    // Average shock position based on scalar. Assumes center at origin.
    int nscl=2,nsum=3; // number of scalars. 
    int nelt=nsum*(nscl+1); // quantities to calculate (rad, thet, mass)
    Real 
    
    Real *inbuf  = (Real*) calloc(nelt,sizeof(Real));
    for (int l=0; l<nelt; l++) {
      inbuf[l] = 0.0;
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real rad, thet;
          Real dv = pcoord->GetCellVolume(k,j,i);
          if (COORDINATE_SYSTEM == "cartesian") {
            Real x   = pcoord->x1v(i);
            Real y   = pcoord->x2v(j);
            Real z   = pcoord->x3v(k);
            rad      = std::sqrt(SQR(x) + SQR(y) + SQR(z));
            thet     = std::acos(z/rad);
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
            Real z = pcoord->x3v(k);
            rad    = std::sqrt(SQR(x) + SQR(y) + SQR(z));
            thet   = std::acos(z/rad);
          } else { // if (COORDINATE_SYSTEM == "spherical_polar")
            Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
            Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            rad    = std::sqrt(SQR(x) + SQR(y) + SQR(z));
            thet   = pcoord->x2v(j);
          }
          for (int l=0; l<nscl; l++) {  // for the individual tracers
            Real s = phydro->u(NHYDRO-NSCALARS+l,k,j,i);
            inbuf[nsum*l+0] += rad *s*dv; // for the radius
            inbuf[nsum*l+1] += thet*s*dv; // for theta
            inbuf[nsum*l+2] += s*dv;      // total mass
          }
          Real d = phydro->u(NHYDRO-NSCALARS,k,j,i)+phydro->u(NHYDRO-NSCALARS+1,k,j,i); // total
          inbuf[nsum*nscl+0] += rad *d*dv;
          inbuf[nsum*nscl+1] += thet*d*dv;
          inbuf[nsum*nscl+2] += d*dv;
        }
      }
    }
#ifdef MPI_PARALLEL
    mpierr = MPI_Allreduce(MPI_IN_PLACE,inbuf,nelt,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
    //if (mpierr) {
    //  msg << "[MeshBlock::UserWorkInLoop]: MPI_Allreduce error = "  << mpierr << std::endl;
    //  throw std::runtime_error(msg.str().c_str());
    //}
    //for (int l=0; l<nsum*(nscl+1); l++) inbuf[l] = outbuf[l];
#endif // MPI_PARALLEL
    for (int l=0; l<=nscl; l++) {
      inbuf[nsum*l+0] /= inbuf[nsum*l+2]; // divide radius by mass, for each tracer l
      inbuf[nsum*l+1] /= inbuf[nsum*l+2]; // divide theta by mass, for each tracer l
    }
    // only the root process writes to file 
    if (Globals::my_rank == 0) {
      Real *data = (Real*) calloc(nelt+1,sizeof(Real));
      data[0] = pmy_mesh->time;
      for (int l=0; l<nelt; l++) {
        data[l+1] = inbuf[l];
      }
      fwrite(data,sizeof(Real),nelt+1,otffile); 
      for (int l=0; l<=nelt; l++) {
        std::cout << std::setprecision(5) << data[l] << ' ';
      }
      std::cout << std::endl;
      free(data);
    }
    free(inbuf);
  } // if (NSCALARS == 2)
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Check radius of sphere to make sure it is round
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (Globals::my_rank == 0) {
    fclose(otffile);
  }
  return;
}
