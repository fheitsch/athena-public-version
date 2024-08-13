//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file exp_followblast.cpp
//  \brief Default blast test with expanding grid.   
//  Check UpdateGridData for examples of shell tracking.
//

//#define DEBUG

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
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

// trying to find the issue with random crashes
//#define SIMPLETRACK

#if (NSCALARS != 1)
#error: Requires NSCALARS = 1
#endif

static void stop_this();

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

void ShockDetector(AthenaArray<Real> data, AthenaArray<Real> grid, int outArr[], Real eps);

Real PowerGridX1(Real x, RegionSize rs);

//========================================================================================
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//========================================================================================
//! \fn Real PowerGridX1(Real x, RegionSize rs)
//  \brief Generates grid following r_s = r_0*(r1/r0)**s, 0<=s<=1
//========================================================================================
Real PowerGridX1(Real x, RegionSize rs) {
  Real delta = rs.x1max/rs.x1min;
  Real r     = rs.x1min*pow(delta,x);
  return r;
}

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
//    ivexp      == -1: expansion velocity set to time-dependent function
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

#ifdef SIMPLETRACK
  vtrack = 0.25*std::pow(pm->time+1.1e-4,-0.75);
#else
  if (ivexp == -1) { // imposed profile
    vtrack = 0.25*std::pow(pm->time+1.1e-4,-0.75); // see plotvtrack.py - need to adapt for different models
  } else if (ivexp == 0) { // constant velocity 
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
//#pragma omp simd
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
//#pragma omp simd
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
//#pragma omp simd
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
      if (ntr < maxntrack-1) vtrack = 0.0; // Prevent oscillations due to poor statistics early on. 
    }
    vtrack = (vtrack <= 0.0) ? 0.0 : vtrack; // enforce expansion
    //if (Globals::my_rank == 0)
    //  fprintf(stdout,"[UpdateGrid] vtrack=%13.5e rtrack=%13.5e xmax =%13.5e\n", vtrack,totquant,pm->GridData(3));
  } // if (ivexp == 0)

#endif // SIMPLETRACK

  vtrack *= boost;

  //if (Globals::my_rank == 0)
  //  fprintf(stdout,"[UpdateGrid]: time=%13.5e vtrack=%13.5e xmax =%13.5e\n",pm->time,vtrack,pm->GridData(3));

  pm->GridData(2) = vtrack;
  if (COORDINATE_SYSTEM=="cartesian") {
    pm->GridData( 6) = vtrack;
    pm->GridData(10) = vtrack;
  }

  return;
}

//Harten Van Leer Shock detection algorithm Out data should be a 1 dimensional array,
// with the same length as indata and grid. indata is the array of Real values where
// we look for the shocks. eps is the slope magnitude limiter, i.e. if the slope is
// above eps, then the location has a shock.
void ShockDetector(AthenaArray<Real> data, AthenaArray<Real> grid, int outArr[], Real eps ) {
  int n, loc;
  Real a, b, c;
  n = data.GetDim1();
  AthenaArray<Real> shockData;
  shockData.NewAthenaArray(n-1);
  loc = 0;
  for (int i=1; i<(n-1); ++i) {
    a = 0;
    b = 0;
    a = std::abs(data(i)-data(i-1));
    b = std::abs(data(i+1)-data(i));
    c = a+b;
    shockData(i-1) = SQR(a-b);

    if ( c <= eps) { 
      shockData(i-1) = 0.0;
    } else { 
      shockData(i-1) /= SQR(a+b);
    }  
  }  
  int k=0;
  for (int i=0; i< (n-1); ++i) {
    if (shockData(i) >= 0.95){
      outArr[k] = i;
      k+=1;    
    }

  }
 
  shockData.DeleteAthenaArray();
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
  if (EXPANDING_ENABLED) {
    EnrollGridDiffEq(WallVel);
      
    if (COORDINATE_SYSTEM == "cartesian") {
      SetGridData(12);

      if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(OUTER_X1,OuterX1_Cartesian);
      }
      if (mesh_bcs[OUTER_X2] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(OUTER_X2,OuterX2_Cartesian);
      }
      if (mesh_bcs[OUTER_X3] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(OUTER_X3,OuterX3_Cartesian);
      }

      if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(INNER_X1,InnerX1_Cartesian);
      }
      if (mesh_bcs[INNER_X2] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(INNER_X2,InnerX2_Cartesian);
      }
      if (mesh_bcs[INNER_X3] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(INNER_X3,InnerX3_Cartesian);
      }

    } else if (COORDINATE_SYSTEM == "spherical_polar") {
      SetGridData(4);
      if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(OUTER_X1,OuterX1_Spherical);
      }
      if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(INNER_X1,InnerX1_Spherical);
      }
    }
    EnrollCalcGridData(UpdateGridData);
    ttrack.NewAthenaArray(maxntrack); // for position tracking
    rtrack.NewAthenaArray(maxntrack);
    ivexp      = pin->GetInteger("problem","ivexp"); // see UpdateGridData
    iweight    = pin->GetInteger("problem","iweight"); // see UpdateGridData
    boost      = pin->GetOrAddReal("problem","boost",1.0); // enhancement of vtrack
    if (ivexp == 0) {
      vtrack0 = pin->GetReal("problem","vtrack0"); // constant tracking velocity for test purposes
    }
    if ((!MAGNETIC_FIELDS_ENABLED) && (fabs(iweight)==4)) {
      fprintf(stdout,"[InitUserMeshData]: iweight set to thermal pressure (3)\n");
      iweight = 3*iweight/fabs(iweight);
    }
 
    if (COORDINATE_SYSTEM == "cartesian") {
      GridData(0) = mesh_size.x1min;
      GridData(1) = 1; 
      GridData(2) = 0.0;
      GridData(3) = mesh_size.x1max; 

      GridData(4) = mesh_size.x2min;
      GridData(5) = 2; 
      GridData(6) = 0.0;
      GridData(7) = mesh_size.x2max; 

      GridData(8) = mesh_size.x3min;
      GridData(9) = 3; 
      GridData(10) = 0.0;
      GridData(11) = mesh_size.x3max; 
    } else {
      GridData(0) = mesh_size.x1min;
      GridData(1) = 1; 
      GridData(2) = 0.0;
      GridData(3) = mesh_size.x1max; 
    }
  }

  ambdens    = pin->GetReal("problem","damb");
  ambvel     = 0.0;
  ambpres    = pin->GetReal("problem","pamb");
  drat       = pin->GetReal("problem","drat");
  prat       = pin->GetReal("problem","prat");
  Real rout  = pin->GetReal("problem","radius");
  Real rin   = rout - pin->GetOrAddReal("problem","ramp",0.0);
  Real vs    = pin->GetOrAddReal("problem","vel",0.0);
  x1rat      = pin->GetOrAddReal("mesh","x1rat",1.0);
  if (x1rat < 0.0)
    EnrollUserMeshGenerator(X1DIR,PowerGridX1);


  return;
}

//========================================================================================
//! \fn void OuterX1_Cartesian(MeshBlock *pmb, Coordinates *pco, 
//                             AthenaArray<Real> &prim,FaceField &b, Real time,
//                             Real dt, int is, int ie, int js, int je,
//                             int ks, int ke, int ngh) {
//  \brief Function for outer boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================
void OuterX1_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,ie+i)    = ambdens;
        prim(IPR,k,j,ie+i)    = ambpres;  
        if (DUAL_ENERGY) prim(IGE,k,j,ie+i) = ambpres;
        prim(IVX,k,j,ie+i)    = 0.0;
        prim(IVY,k,j,ie+i)    = 0.0;
        prim(IVZ,k,j,ie+i)    = 0.0;
        prim(NHYDRO-NSCALARS,k,j,ie+i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,ie+i+1) = bx0;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,ie+i) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,ie+i) = bz0;
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void OuterX2_Cartesian(MeshBlock *pmb, Coordinates *pco, 
//                             AthenaArray<Real> &prim,FaceField &b, Real time,
//                             Real dt, int is, int ie, int js, int je,
//                             int ks, int ke, int ngh) {
//  \brief Function for outer boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================
void OuterX2_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,k,je+j,i)    = ambdens;
        prim(IPR,k,je+j,i)    = ambpres;  
        if (DUAL_ENERGY) prim(IGE,k,je+j,i) = ambpres;
        prim(IVX,k,je+j,i)    = 0.0;
        prim(IVY,k,je+j,i)    = 0.0;
        prim(IVZ,k,je+j,i)    = 0.0;
        prim(NHYDRO-NSCALARS,k,je+j,i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(k,je+j,i) = bx0;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f(k,je+j+1,i) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f(k,je+j,i) = bz0;
        }
      }
    }
  }

  return;
}

//========================================================================================
//! \fn void OuterX3_Cartesian(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for outer boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================
void OuterX3_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,ke+k,j,i)    = ambdens;
        prim(IPR,ke+k,j,i)    = ambpres;  
        if (DUAL_ENERGY) prim(IGE,ke+k,j,i) = ambpres;
        prim(IVX,ke+k,j,i)    = 0.0;
        prim(IVY,ke+k,j,i)    = 0.0;
        prim(IVZ,ke+k,j,i)    = 0.0;
        prim(NHYDRO-NSCALARS,ke+k,j,i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(ke+k,j,i) = bx0;
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f(ke+k,j,i) = by0;
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f(ke+k+1,j,i) = bz0;
        }
      }
    }
  }

  return;

}
//========================================================================================
//! \fn void InnerX1_Cartesian(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for inner boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================

void InnerX1_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,is-i) = ambdens;
        prim(IPR,k,j,is-i) = ambpres;  
        if (DUAL_ENERGY) prim(IGE,k,j,is-i) = ambpres;
        prim(IVX,k,j,is-i) = 0.0;
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;
        prim(NHYDRO-NSCALARS,k,j,is-i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,is-i) = bx0;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,is-i) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,is-i) = bz0;
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void InnerX2_Cartesian(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for inner boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================

void InnerX2_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,k,js-j,i)    = ambdens;
        prim(IPR,k,js-j,i)    = ambpres;  
        if (DUAL_ENERGY) prim(IGE,k,js-j,i) = ambpres;
        prim(IVX,k,js-j,i)    = 0.0;
        prim(IVY,k,js-j,i)    = 0.0;
        prim(IVZ,k,js-j,i)    = 0.0;
        prim(NHYDRO-NSCALARS,k,js-j,i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(k,js-j,i) = bx0;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f(k,js-j,i) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f(k,js-j,i) = bz0;
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void InnerX3_Cartesian(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for inner boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================

void InnerX3_Cartesian(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,ks-k,j,i)    = ambdens;
        prim(IPR,ks-k,j,i)    = ambpres;  
        if (DUAL_ENERGY) prim(IGE,ks-k,j,i) = ambpres;
        prim(IVX,ks-k,j,i)    = 0.0;
        prim(IVY,ks-k,j,i)    = 0.0;
        prim(IVZ,ks-k,j,i)    = 0.0;
        prim(NHYDRO-NSCALARS,ks-k,j,i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(ks-k,j,i) = bx0;
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f(ks-k,j,i) = by0;
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f(ks-k,j,i) = bz0;
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void OuterX1_Spherical(MeshBlock *pmb, Coordinates *pco, 
//                             AthenaArray<Real> &prim,FaceField &b, Real time,
//                             Real dt, int is, int ie, int js, int je,
//                             int ks, int ke, int ngh) {
//  \brief Function for outer boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================
void OuterX1_Spherical(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,ie+i)    = ambdens;
        prim(IPR,k,j,ie+i)    = ambpres;
        if (DUAL_ENERGY) prim(IGE,k,j,ie+i) = ambpres;
        prim(IVX,k,j,ie+i)    = 0.0;
        prim(IVY,k,j,ie+i)    = 0.0;
        prim(IVZ,k,j,ie+i)    = 0.0;
        prim(NHYDRO-NSCALARS,k,j,ie+i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      Real phi = pco->x3v(k);
      for (int j=js; j<=je; ++j) {
        Real theta = pco->x2v(j);
        if (ibtype == 0) {
#pragma omp simd 
          for (int i=1; i<=ngh; ++i) {
            b.x1f(k,j,ie+i+1) =   std::sin(theta)*std::cos(phi)*bx0
                                + std::sin(theta)*std::sin(phi)*by0
                                + std::cos(theta)*bz0;
          }
        } else if (ibtype == 1) {
#pragma omp simd 
          for (int i=1; i<=ngh; ++i) {
            b.x1f(k,j,ie+i+1) = 0.0;
          }
        } else if (ibtype == 2) {
#pragma omp simd 
          for (int i=1; i<=ngh; ++i) {
            b.x1f(k,j,ie+i+1) = 0.0;
          }
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      Real phi = pco->x3v(k);
      for (int j=js; j<=je+1; ++j) {
        Real theta = pco->x2f(j);
        if (ibtype == 0) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x2f(k,j,ie+i) =   std::cos(theta)*std::cos(phi)*bx0
                              + std::cos(theta)*std::sin(phi)*by0
                              - std::sin(theta)*bz0;
          }
        } else if (ibtype == 1) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x2f(k,j,ie+1) = 0.0;
          }
        } else if (ibtype == 2) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x2f(k,j,ie+1) = bz0;
          }
        }
      } 
    } 
    for (int k=ks; k<=ke+1; ++k) {
      Real phi = pco->x3f(k);
      for (int j=js; j<=je; ++j) {
        if (ibtype == 0) {
#pragma omp simd 
          for (int i=1; i<=ngh; ++i) {
            b.x3f(k,j,ie+i) = - std::sin(phi)*bx0
                              + std::cos(phi)*by0;
          }
        } else if (ibtype == 1) {
#pragma omp simd 
          for (int i=1; i<=ngh; ++i) {
            b.x3f(k,j,ie+i) = bz0;
          } 
        } else if (ibtype == 2) {
#pragma omp simd 
          for (int i=1; i<=ngh; ++i) {
            b.x3f(k,j,ie+i) = 0.00;
          } 
        }
      } 
    }
  } 
  return;
}

//========================================================================================
//! \fn void InnerX1_Spherical(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for inner boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================

void InnerX1_Spherical(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,is-i) = ambdens*drat;
        prim(IPR,k,j,is-i) = ambpres*prat;
        if (DUAL_ENERGY) prim(IGE,k,j,is-i) = ambpres*prat;
        prim(IVX,k,j,is-i) = 0.0;
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    Real theta, phi;
    for (int k=ks; k<=ke; ++k) {
      phi = pco->x3v(k);
      for (int j=js; j<=je; ++j) {
        theta = pco->x2v(j);
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,is-i) =   std::sin(theta)*std::cos(phi)*bx0
                            + std::sin(theta)*std::sin(phi)*by0
                            + std::cos(theta)*bz0;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      phi = pco->x3v(k);
      for (int j=js; j<=je+1; ++j) {
        theta = pco->x2f(j);
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,is-i) =    std::cos(theta)*std::cos(phi)*bx0
                             + std::cos(theta)*std::sin(phi)*by0
                             - std::sin(theta)*bz0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      phi = pco->x3f(k);
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,is-i) = - std::sin(phi)*bx0
                            + std::cos(phi)*by0;
        }
      }
    }
  }
  return;
}



//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int iprob = pin->GetInteger("problem","iprob"); // -1: field loop; 0: uniform; > 0: blast
  Real rout = pin->GetReal("problem","radius");
  Real dr   = pin->GetReal("problem","ramp");
  Real pa   = pin->GetReal("problem","pamb");
  Real da   = pin->GetReal("problem","damb");
  prat      = pin->GetReal("problem","prat");
  drat      = pin->GetReal("problem","drat");
  Real vSh   = pin->GetReal("problem","vel");
  if (MAGNETIC_FIELDS_ENABLED) {
    b0 = pin->GetReal("problem","b0");
    bz0= pin->GetOrAddReal("problem","bz0",0.0);
    angle = (PI/180.0)*pin->GetReal("problem","angle");
    bx0= std::cos(angle)*b0;
    by0= std::sin(angle)*b0;
    ibtype = pin->GetOrAddReal("problem","ibtype",0); // 0: uniform field (bx0, by0, bz0), 1: azimuthal field (bz0=bphi0), 2: bz0 = btheta0
    if (ibtype == 1) { // azimuthal field: no radial or polar component
      b0 = 0.0;
      bx0 = 0.0;
      by0 = 0.0;
    }
  }
  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;

  if (Globals::my_rank==0) {
    fprintf(stdout,"IDN=%2i IVX=%2i IVY=%2i IVZ=%2i IPR=%2i IBY=%2i IBZ=%2i NHYDRO-SCALARS=%2i NHYDRO=%2i NWAVE=%2i NWAVE+NINT+NSCAL=%2i\n",
            IDN,IVX,IVY,IVZ,IPR,IBY,IBZ,NHYDRO-NSCALARS,NHYDRO,NWAVE,NWAVE+NINT+NSCALARS);
  }

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
    std::cout << "### FATAL ERROR in blast.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl;
  }

  if (iprob == -1) { // field loop
    AthenaArray<Real> ax,ay,az;
    int nx1 = (ie-is)+1 + 2*(NGHOST);
    int nx2 = (je-js)+1 + 2*(NGHOST);
    int nx3 = (ke-ks)+1 + 2*(NGHOST);
    ax.NewAthenaArray(nx3,nx2,nx1);
    ay.NewAthenaArray(nx3,nx2,nx1);
    az.NewAthenaArray(nx3,nx2,nx1);
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie+1; i++) {
          ax(k,j,i) = 0.0;
          ay(k,j,i) = 0.0;
          if ((SQR(pcoord->x1f(i)-x0) + SQR(pcoord->x2f(j)-y0)) < rout*rout) {
            az(k,j,i) = 1e-3*(rout - std::sqrt(SQR(pcoord->x1f(i)-x0) +
                                              SQR(pcoord->x2f(j)-y0)));
          } else {
            az(k,j,i) = 0.0;
          }
        }
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IDN,k,j,i) = 1.0;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        }
      }
    }

    // initialize interface B
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          pfield->b.x1f(k,j,i) = (az(k,j+1,i) - az(k,j,i))/pcoord->dx2f(j) -
                                 (ay(k+1,j,i) - ay(k,j,i))/pcoord->dx3f(k);
        }
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x2f(k,j,i) = (ax(k+1,j,i) - ax(k,j,i))/pcoord->dx3f(k) -
                                 (az(k,j,i+1) - az(k,j,i))/pcoord->dx1f(i);
        }
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x3f(k,j,i) = (ay(k,j,i+1) - ay(k,j,i))/pcoord->dx1f(i) -
                                 (ax(k,j+1,i) - ax(k,j,i))/pcoord->dx2f(j);
        }
      }
    }

    // initialize total energy
    if (NON_BAROTROPIC_EOS) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            phydro->u(IEN,k,j,i) = 1.0/gm1 +
            0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
                 SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
                 SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i)))) + (0.5)*
            (SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i))
             + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
          }
        }
      }
    }

    ax.DeleteAthenaArray();
    ay.DeleteAthenaArray();
    az.DeleteAthenaArray();

    return;
  }

  if (iprob == 0) { // uniform everything
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IDN,k,j,i) = 1.0;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        }
      }
    }
    // initialize interface B
    if (COORDINATE_SYSTEM == "cartesian") {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie+1; i++) {
            pfield->b.x1f(k,j,i) = bx0;
          }
        }
      }
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je+1; j++) {
          for (int i=is; i<=ie; i++) {
            pfield->b.x2f(k,j,i) = by0;  
          }
        }
      }
      for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            pfield->b.x3f(k,j,i) = bz0;
          }
        }
      }
    } else if (COORDINATE_SYSTEM == "spherical_polar") {
      for (int k=ks; k<=ke; k++) {
        Real phi = pcoord->x3v(k);
        for (int j=js; j<=je; j++) {
          Real theta = pcoord->x2v(j);
          for (int i=is; i<=ie+1; i++) {
            if (ibtype == 0) {
              pfield->b.x1f(k,j,i) =   std::sin(theta)*std::cos(phi)*bx0
                                     + std::sin(theta)*std::sin(phi)*by0
                                     + std::cos(theta)*bz0;
            } else if (ibtype == 1) {
              pfield->b.x1f(k,j,i) = 0.0; 
            } else if (ibtype == 2) {
              pfield->b.x1f(k,j,i) = 0.0;  
            }
          }
        }
      }
      for (int k=ks; k<=ke; k++) {
        Real phi   = pcoord->x3v(k);
        for (int j=js; j<=je+1; j++) {
          Real theta = pcoord->x2f(j);
          for (int i=is; i<=ie; i++) {
            if (ibtype == 0) {
              pfield->b.x2f(k,j,i) =   std::cos(theta)*std::cos(phi)*bx0
                                     + std::cos(theta)*std::sin(phi)*by0
                                     - std::sin(theta)*bz0;
            } else if (ibtype == 1) {
              pfield->b.x2f(k,j,i) = 0.0;
            } else if (ibtype == 2) {
              pfield->b.x2f(k,j,i) = bz0;
            }
          }
        }
      }
      for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            if (ibtype == 0) {
              Real phi = pcoord->x3f(k);
              pfield->b.x3f(k,j,i) = - std::sin(phi)*bx0
                                     + std::cos(phi)*by0;
            } else if (ibtype == 1) {
              pfield->b.x3f(k,j,i) = bz0;
            } else if (ibtype == 2) {
              pfield->b.x3f(k,j,i) = 0.0;
            }
          }
        }
      }
    }
    // initialize total energy
    if (NON_BAROTROPIC_EOS) {
      for (int k=ks; k<=ke; k++) { 
        for (int j=js; j<=je; j++) { 
          for (int i=is; i<=ie; i++) {
            phydro->u(IEN,k,j,i) =  1.0/gm1 + 0.5*(SQR(bx0)+SQR(by0)+SQR(bz0))
                                   +0.5*(  SQR(phydro->u(IM1,k,j,i)) 
                                         + SQR(phydro->u(IM2,k,j,i))
                                         + SQR(phydro->u(IM3,k,j,i)))
                                       /phydro->u(IDN,k,j,i);
          }
        }
      }
    }
    return;
  } 

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad, r, x, y, z;
        if (COORDINATE_SYSTEM == "cartesian") {
          x   = pcoord->x1v(i);
          y   = pcoord->x2v(j);
          z   = pcoord->x3v(k);
          r   = std::sqrt(SQR(x)+SQR(y)+SQR(z));
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          x   = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          y   = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          z   = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        } else { // if (COORDINATE_SYSTEM == "spherical_polar")
          x   = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          y   = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          z   = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        }
        Real den = da;
        Real v1  = 0.0;
 
        den += da*(drat-1.0)*0.5*(1.0-std::tanh((rad-rout)/dr));
        v1  += vSh*0.5*(1.0-std::tanh((rad-rout)/dr))*(rad/rout);
 
        phydro->u(IDN,k,j,i) = den;
        if (COORDINATE_SYSTEM == "cartesian") {
          phydro->u(IM1,k,j,i) = den*v1*x/r;
          phydro->u(IM2,k,j,i) = den*v1*y/r;
          phydro->u(IM3,k,j,i) = den*v1*z/r;
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phydro->u(IM1,k,j,i) = den*v1;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        } else if (COORDINATE_SYSTEM == "spherical_polar") {
          phydro->u(IM1,k,j,i) = den*v1;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        }
        if (NON_BAROTROPIC_EOS) {
          Real pres = pa;
          pres += pa*(prat-1.0)*0.5*(1.0-std::tanh((rad-rout)/dr));
          phydro->u(IEN,k,j,i) = 0.5*den*SQR(v1)+pres/gm1;
          if (DUAL_ENERGY) {
            phydro->u(IIE,k,j,i) = pres/gm1;
          }
        }
        phydro->u(NHYDRO-NSCALARS,k,j,i) = den*0.25*(1.0+std::tanh((rad-1.0*rout)/(0.01*rout)))
                                                   *(1.0-std::tanh((rad-1.2*rout)/(0.01*rout)));
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie+1; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            pfield->b.x1f(k,j,i) = bx0;
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real phi = pcoord->x2v(j);
            pfield->b.x1f(k,j,i) =
                b0 * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x1f(k,j,i) =   std::sin(theta)*(std::cos(phi)*bx0+std::sin(phi)*by0)
                                   + std::cos(theta)*bz0; 
          }
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            pfield->b.x2f(k,j,i) = by0;
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real phi = pcoord->x2f(j);
            pfield->b.x2f(k,j,i) =
                b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real theta = pcoord->x2f(j); // needs to be located on wall for b.x2f
            Real phi   = pcoord->x3v(k); // here we need center position
            pfield->b.x2f(k,j,i) =   std::cos(theta)*(std::cos(phi)*bx0+std::sin(phi)*by0)
                                   - std::sin(theta)*bz0;
          }
        }
      }
    }
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
            pfield->b.x3f(k,j,i) = bz0;
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real phi = pcoord->x3f(k); // Needs to be located on wall for b.x3f.
            pfield->b.x3f(k,j,i) = - std::sin(phi)*bx0
                                   + std::cos(phi)*by0;
          }
        }
      }
    }

    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc,
                                       pcoord, is, ie, js, je, ks, ke);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IEN,k,j,i) += 0.5*( SQR(pfield->bcc(IB1,k,j,i))
                                       +SQR(pfield->bcc(IB2,k,j,i))
                                       +SQR(pfield->bcc(IB3,k,j,i)));
        }
      }
    }
  } 

  return;
}

//====================================================================================

void Mesh::UserWorkInLoop(void) {

    if (ncycle_out != 0) {
    if (ncycle % ncycle_out != 0) {
      return;
    }
  }

  bool fail = false, allfail = false;
  int ifail = 0;

  MeshBlock *pmb=pblock;

  // For spherical-polar coordinate print diagnostic values of radial and angular resolution
  if ((EXPANDING_ENABLED) && (COORDINATE_SYSTEM == "spherical_polar")) {
    if (Globals::my_rank==0) {
      if (x1rat < 0.0) {
        Real nx2exp = (PI*std::pow(mesh_size.x1max/mesh_size.x1min,0.5/mesh_size.nx1))
                     /(std::pow(mesh_size.x1max/mesh_size.x1min,1.0/mesh_size.nx1)-1.0);
        fprintf(stdout,"[UserWorkInLoop]: rmax = %13.5e nx2exp/nx2 = %13.5e\n",mesh_size.x1max,nx2exp/((Real) mesh_size.nx2));
      } else {
        fprintf(stdout,"[UserWorkInLoop]: rmax = %13.5e\n",mesh_size.x1max);
      }
    }
  }

  while (pmb != NULL) { // collect results from individual pmbs
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      for (int j=pmb->js; j<=pmb->je; j++) {
        for (int i=pmb->is; i<=pmb->ie; i++) {
          fail =    isnan(pmb->phydro->u(IEN,k,j,i))
                 || isnan(pmb->phydro->u(IDN,k,j,i))
                 || (pmb->phydro->u(IEN,k,j,i) <= 0.0)
                 || (pmb->phydro->u(IDN,k,j,i) <= 0.0);
          if (DUAL_ENERGY) {
            fail = fail || isnan(pmb->phydro->u(IIE,k,j,i)) || (pmb->phydro->u(IIE,k,j,i) <= 0.0);
          }
          if (fail) {
            ifail++;
            //if (DUAL_ENERGY) {
            //  std::cout << "[UserWorkInLoop]: Warning: p=" << std::setw(3) << Globals:my_rank 
            //            << " i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
            //            << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IDN,k,j,i)
            //            << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM1,k,j,i)
            //            << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM2,k,j,i)
            //            << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM3,k,j,i)
            //            << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IEN,k,j,i)
            //            << " ei=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IIE,k,j,i)
            //            << std::endl;
            //} else {
            //  std::cout << "[UserWorkInLoop]: Warning: p=" << std::setw(3) << Globals:my_rank 
            //            << " i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
            //            << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IDN,k,j,i)
            //            << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM1,k,j,i)
            //            << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM2,k,j,i)
            //            << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM3,k,j,i)
            //            << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IEN,k,j,i)
            //            << std::endl;
            //}
          }
          allfail = (fail || allfail);
        }
      }
    }
    pmb = pmb->next;
  } 

  if (allfail) {
    std::cout << "[UserWorkInLoop]: p=" << std::setw(4) << Globals::my_rank << ": failure in " << std::setw(9) << ifail << " cells." << std::endl;
  }

#ifdef MPI_PARALLEL
  int ierr = MPI_Allreduce(MPI_IN_PLACE,&allfail,1,MPI_C_BOOL,MPI_LOR,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&ifail,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
  if (allfail) {
    if (Globals::my_rank == 0) {
      std::cout << "[UserWorkInLoop]: failure in " << std::setw(9) << ifail << " cells." << std::endl;
    }
    stop_this();
  }

  return;
}
