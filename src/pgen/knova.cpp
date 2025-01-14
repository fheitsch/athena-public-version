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

#if (NSCALARS != 3)
#error: Requires NSCALARS = 3
#endif

static void stop_this();

//====================================================================================
// global variables
FILE *otffile;

//========================================================================================
// Time Dependent Grid Functions
//  \brief Functions for time dependent grid, including two example boundary conditions
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
void UpdateGridDataKnova(Mesh *pm);
//Global variables for GridUpdate
int maxntrack = 20;
int ncycold=-1;
int ivexp;
Real vtrack0, boost,x1rat;
Real dtramp0, dtramp1, tramp0;
Real dlnrdlnt0=0.0, dlnrdlnt1=0.0;
AthenaArray<Real> ttrack,rtrack;

//Cooling function
void *CoolingFunc=NULL;
int iheatcool;
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);
Real RampTimeStep(MeshBlock *pmb);

//Global Variables for OuterX1
int ibtype;
Real ambdens, ambprss, time_free=HUGE_NUMBER;
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
//! \fn void UpdateGridDataKnova(Mesh *pm)
//  \brief Function which can edit and calculate any terms in gridData, which is used 
//  in the WallVel function. The object in mesh is GridData(i) and i can range over the
//  integers, limited by SetGridData argument in InitMeshUserData. See exp_blast for an 
//  example use of this function.
// This is a reduced version, equivalent to ivexp ==2, iweight = 12.
// It calculates the mean radius weighted by density with non-zero momentum, and the 
// mean radius of the shell scalar. The larger of these two radii is used to determine
// the current expansion velocity. This allows us to track the transition from free expansion
// (advection of shell scalar) to ST phase (density shell formed at shock). 
//========================================================================================
void UpdateGridDataKnova(Mesh *pm) {
  MeshBlock *pmb = pm->pblock;
  Real vtrack = 0.0;
  Real gamma  = pmb->peos->GetGamma();
  int ntr=0;

  pm->GridData(3)  = pm->mesh_size.x1max;
  pm->GridData(0)  = pm->mesh_size.x1min;
  pm->GridData(7)  = pm->mesh_size.x2max;
  pm->GridData(4)  = pm->mesh_size.x2min;
  pm->GridData(11) = pm->mesh_size.x3max;
  pm->GridData(8)  = pm->mesh_size.x3min;

  if (ivexp == 0) { // constant velocity 
    vtrack  = vtrack0;
    vtrack  = boost*vtrack0;
    pm->GridData( 2) = vtrack;
    pm->GridData( 6) = vtrack;
    pm->GridData(10) = vtrack;
    return;
  }
  // the rest is ivexp==2, iweight==12
  AthenaArray<Real> weight, weight2, radius, vol;
  Real totweight = 0.0, totradius = 0.0, totweight2=0.0,totradius2=0.0;
  // assuming that all the meshblocks have the same size
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  int ncells3 = pmb->block_size.nx3+2*NGHOST;
  int ncells2 = pmb->block_size.nx2+2*NGHOST;
  int ncells1 = pmb->block_size.nx1+2*NGHOST;
  weight.NewAthenaArray(ncells3,ncells2,ncells1);
  weight2.NewAthenaArray(ncells3,ncells2,ncells1);
  radius.NewAthenaArray(ncells3,ncells2,ncells1);
  vol.NewAthenaArray(ncells1);

  while (pmb != NULL) {

    for (int k=ks; k<=ke; ++k) {
      Real z = pmb->pcoord->x3v(k);
      for (int j=js; j<=je; ++j) {
        Real y = pmb->pcoord->x2v(j);
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x = pmb->pcoord->x1v(i);
          radius(k,j,i) = std::sqrt(SQR(x)+SQR(y)+SQR(z));
          Real mk = SQR(pmb->phydro->u(IM1,k,j,i))+SQR(pmb->phydro->u(IM2,k,j,i))+SQR(pmb->phydro->u(IM3,k,j,i));
          weight(k,j,i) = pmb->phydro->u(IDN,k,j,i) * (mk > 1e-2); // use density with non-zero momentum
          weight2(k,j,i)= pmb->phydro->u(NHYDRO-NSCALARS,k,j,i);   // shell scalar
        }
      }
    }

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        pmb->pcoord->CellVolume(k,j,is,ie,vol);
#pragma omp simd reduction(+:totweight,totradius,totweight2,totradius2)
        for (int i=is; i<=ie; ++i) {
          Real w = weight(k,j,i)*vol(i);
          Real w2= weight2(k,j,i)*vol(i);
          totweight   += w;
          totradius   += radius(k,j,i)*w;
          totweight2  += w2;
          totradius2  += radius(k,j,i)*w2;
        }
      }
    }
    pmb = pmb->next;
  }

  weight.DeleteAthenaArray();
  weight2.DeleteAthenaArray();
  radius.DeleteAthenaArray();
  vol.DeleteAthenaArray();

#ifdef MPI_PARALLEL
  Real myval[4];
  myval[0] = totradius;
  myval[1] = totweight;
  myval[2] = totradius2;
  myval[3] = totweight2;
  MPI_Allreduce(MPI_IN_PLACE,&myval,4,MPI_ATHENA_REAL,MPI_SUM,
                MPI_COMM_WORLD);
  totradius = myval[0];
  totweight = myval[1];
  totradius2= myval[2];
  totweight2= myval[3];
#endif
  totradius  /= totweight;
  totradius2 /= totweight2;
  if (Globals::my_rank==0)
    fprintf(stdout,"[UpdateGridDataKnova]: totradius=%13.5e totradius2=%13.5e\n",
             totradius,totradius2);
  totradius   = std::max(totradius,totradius2);

  ntr         = (pm->ncycle >= maxntrack) ? maxntrack-1 : pm->ncycle; // number of elements to track
  if (ncycold < pm->ncycle) { // do not update during substep
    ncycold = pm->ncycle;
    if (ntr == 0) { // first iteration
      ttrack(0) = 0.0;
      rtrack(0) = totradius;
      vtrack    = 0.0;
    } else { // all subsequent iterations
      for (int m=0; m<ntr; ++m) { // shift old elements 
        ttrack(ntr-m) = ttrack(ntr-m-1);
        rtrack(ntr-m) = rtrack(ntr-m-1);
      }
      ttrack(0) = pm->time; // add new element
      rtrack(0) = totradius;
    }
  }
  vtrack   = 0.0;
  for (int m=1; m<ntr; ++m) { // calculate front velocity as average over tracking elements
    Real vt = (rtrack(m-1)-rtrack(m))/(ttrack(m-1)-ttrack(m)) * (pm->GridData(3)/totradius);
    vtrack += vt;
  }
  if (ntr > 0) vtrack /= ntr;

  // global variables to control switch-off
  dlnrdlnt0 = dlnrdlnt1;
  dlnrdlnt1 = 0.0;
  for (int m=2; m< ntr; ++m) {
    Real dlrdlt = (std::log(rtrack(m))-std::log(rtrack(m-1)))
                 /(std::log(ttrack(m))-std::log(ttrack(m-1)));
    dlnrdlnt1 += dlrdlt; // store the dln(r)/dln(t) derivative
  }
  if (ntr > 2) dlnrdlnt1 /= (ntr-2);

  // Once dlnrdlnt drops for the first time, we need to switch off resetting.
  if (dlnrdlnt1 < dlnrdlnt0) time_free = pm->time;

  vtrack = (vtrack <= 0.0) ? 0.0 : vtrack; // enforce expansion
  if (Globals::my_rank == 0)
    fprintf(stdout,"[UpdateGrid] vtrack=%13.5e rtrack=%13.5e xmax =%13.5e dlrndlnt=%13.5e time_free=%13.5e\n", vtrack,totradius,pm->GridData(3),dlnrdlnt1,time_free);

  vtrack *= boost;

  pm->GridData( 2) = vtrack;
  pm->GridData( 6) = vtrack;
  pm->GridData(10) = vtrack;

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
      stop_this();
    }
    EnrollCalcGridData(UpdateGridDataKnova);
    ttrack.NewAthenaArray(maxntrack); // for position tracking
    rtrack.NewAthenaArray(maxntrack);
    ivexp      = pin->GetOrAddInteger("problem","ivexp",1); // see UpdateGridData
    boost      = pin->GetOrAddReal("problem","boost",1.0); // enhancement of vtrack
 
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

  ambdens    = pin->GetReal("problem","d0");
  ambprss    = pin->GetReal("problem","p0");
  x1rat      = pin->GetOrAddReal("mesh","x1rat",1.0);
  iheatcool  = pin->GetOrAddInteger("problem","iheatcool",0);
  if (x1rat < 0.0)
    EnrollUserMeshGenerator(X1DIR,PowerGridX1);
  EnrollUserExplicitSourceFunction(HeatCool);
  EnrollUserTimeStepFunction(HeatCoolTimeStep);
    //tramp0  = pin->GetOrAddReal("problem","tramp0",0.0);
    //if (tramp0 <= 0.0) {
    //  EnrollUserTimeStepFunction(RampTimeStep);
    //  dtramp0 = pin->GetOrAddReal("problem","dtramp0",1e-15); // trial&error for ramping timestep
    //  dtramp1 = pin->GetOrAddReal("problem","dtramp1",1e-13);
    //}

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
        prim(IPR,k,j,ie+i)    = ambprss;  
        if (DUAL_ENERGY) prim(IGE,k,j,ie+i) = ambprss;
        prim(IVX,k,j,ie+i)    = 0.0;
        prim(IVY,k,j,ie+i)    = 0.0;
        prim(IVZ,k,j,ie+i)    = 0.0;
        prim(NHYDRO-NSCALARS,k,j,ie+i)   = 0.0;
        prim(NHYDRO-NSCALARS+1,k,j,ie+i) = 0.0;
        prim(NHYDRO-NSCALARS+2,k,j,ie+i) = 0.0;
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
        prim(IPR,k,je+j,i)    = ambprss;  
        if (DUAL_ENERGY) prim(IGE,k,je+j,i) = ambprss;
        prim(IVX,k,je+j,i)    = 0.0;
        prim(IVY,k,je+j,i)    = 0.0;
        prim(IVZ,k,je+j,i)    = 0.0;
        prim(NHYDRO-NSCALARS,k,je+j,i) = 0.0;
        prim(NHYDRO-NSCALARS+1,k,je+j,i) = 0.0;
        prim(NHYDRO-NSCALARS+2,k,je+j,i) = 0.0;
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
        prim(IPR,ke+k,j,i)    = ambprss;  
        if (DUAL_ENERGY) prim(IGE,ke+k,j,i) = ambprss;
        prim(IVX,ke+k,j,i)    = 0.0;
        prim(IVY,ke+k,j,i)    = 0.0;
        prim(IVZ,ke+k,j,i)    = 0.0;
        prim(NHYDRO-NSCALARS,ke+k,j,i) = 0.0;
        prim(NHYDRO-NSCALARS+1,ke+k,j,i) = 0.0;
        prim(NHYDRO-NSCALARS+2,ke+k,j,i) = 0.0;
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
        prim(IPR,k,j,is-i) = ambprss;  
        if (DUAL_ENERGY) prim(IGE,k,j,is-i) = ambprss;
        prim(IVX,k,j,is-i) = 0.0;
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;
        prim(NHYDRO-NSCALARS,k,j,is-i) = 0.0;
        prim(NHYDRO-NSCALARS+1,k,j,is-i) = 0.0;
        prim(NHYDRO-NSCALARS+2,k,j,is-i) = 0.0;
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
        prim(IPR,k,js-j,i)    = ambprss;  
        if (DUAL_ENERGY) prim(IGE,k,js-j,i) = ambprss;
        prim(IVX,k,js-j,i)    = 0.0;
        prim(IVY,k,js-j,i)    = 0.0;
        prim(IVZ,k,js-j,i)    = 0.0;
        prim(NHYDRO-NSCALARS,k,js-j,i) = 0.0;
        prim(NHYDRO-NSCALARS+1,k,js-j,i) = 0.0;
        prim(NHYDRO-NSCALARS+2,k,js-j,i) = 0.0;
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
        prim(IPR,ks-k,j,i)    = ambprss;  
        if (DUAL_ENERGY) prim(IGE,ks-k,j,i) = ambprss;
        prim(IVX,ks-k,j,i)    = 0.0;
        prim(IVY,ks-k,j,i)    = 0.0;
        prim(IVZ,ks-k,j,i)    = 0.0;
        prim(NHYDRO-NSCALARS,ks-k,j,i) = 0.0;
        prim(NHYDRO-NSCALARS+1,ks-k,j,i) = 0.0;
        prim(NHYDRO-NSCALARS+2,ks-k,j,i) = 0.0;
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
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief Called for individual meshblock
//========================================================================================

void Mesh::UserWorkInLoop(void) {

  MeshBlock *pmb=pblock;
  Real gamma = pmb->peos->GetGamma();
  Real gm1 = gamma - 1.0, eint;
  bool fail = false, allfail = false;
  int ifail = 0;

  const int nq = 7;
  Real qtot[nq]; // 0: vol, 1: dens, 2: vtot, 3: etot, 4: eint, 5: ekin, 6: emag
  Real qmin[nq];
  Real qmax[nq];
  Real ener[nq];
  Real trac[3];
  Real lengrat[2];
  for (int q=0; q<nq; q++) {
    qtot[q] = 0.0;
    qmin[q] = (HUGE_NUMBER);
    qmax[q] = -(HUGE_NUMBER);
    ener[q] = 0.0;
  }
  for (int q=0; q<3; q++) 
    trac[q] = 0.0;
  for (int q=0; q<2; q++)
    lengrat[q] = (HUGE_NUMBER);
  Real u[NHYDRO];

  while (pmb != NULL) { // collect results from individual pmbs
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      Real x3  = pmb->pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; j++) {
        Real x2  = pmb->pcoord->x2v(j);
        for (int i=pmb->is; i<=pmb->ie; i++) {
          Real x1 = pmb->pcoord->x1v(i);
#pragma omp simd
          for (int q=0; q<NHYDRO; ++q)
            u[q] = pmb->phydro->u(q,k,j,i);
          fail =    isnan(u[IEN])
                 || isnan(u[IDN])
                 || (u[IEN] <= 0.0)
                 || (u[IDN] <= 0.0);
          if (DUAL_ENERGY) {
            fail = fail || isnan(u[IIE]) || (u[IIE] <= 0.0);
          } else {
            eint =  u[IEN]-0.5*(SQR(u[IM1])+SQR(u[IM2])+SQR(u[IM3]))/u[IDN];
            fail = fail || isnan(eint);// || (eint <= 0.0);
          }
          if (fail) {
            if (DUAL_ENERGY) {
              std::cout << "[UserWorkInLoop]: Warning: i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
                        << " x1=" << std::scientific << std::setw(11) << std::setprecision(3) << x1
                        << " x2=" << std::scientific << std::setw(11) << std::setprecision(3) << x2
                        << " x3=" << std::scientific << std::setw(11) << std::setprecision(3) << x3
                        << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << u[IDN]
                        << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IM1]
                        << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IM2]
                        << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IM3]
                        << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IEN]
                        << " ei=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IIE]
                        << std::endl;
            } else {
              std::cout << "[UserWorkInLoop]: Warning: i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
                        << " x1=" << std::scientific << std::setw(11) << std::setprecision(3) << x1
                        << " x2=" << std::scientific << std::setw(11) << std::setprecision(3) << x2
                        << " x3=" << std::scientific << std::setw(11) << std::setprecision(3) << x3
                        << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << u[IDN]
                        << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IM1]
                        << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IM2]
                        << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IM3]
                        << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IEN]
                        << " ei=" << std::scientific << std::setw(11) << std::setprecision(3) << eint
                        << std::endl;
            }
            ifail++;
          }
          allfail = (fail||allfail);
          Real dx1  = pmb->pcoord->dx1f(i);
          Real dvol = pmb->pcoord->GetCellVolume(k,j,i);
          // energy
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
          // shell tracking: velocity and radius. Shell defined in ProblemGenerator
          Real rad = std::sqrt(x1*x1+x2*x2+x3*x3);
          trac[0] += u[NHYDRO-NSCALARS];                                              
          trac[1] += (u[NHYDRO-NSCALARS]/u[IDN])*(u[IM1]*x1+u[IM2]*x2+u[IM3]*x3)/rad; 
          trac[2] += u[NHYDRO-NSCALARS]*rad;                                          
          Real temp = gm1*ener[2]/u[IDN];
          //if (CoolingFunc != NULL) 
            //lengrat[0] = std::min(lengrat[0],temp*std::sqrt(temp)/(dx1*fabs(CoolingFunc(u[IDN],temp)))); // cooling length
          lengrat[1] = std::min(lengrat[1],std::sqrt(PI*temp/u[IDN])/dx1); // Jeans length
        }
      }
    }
    pmb = pmb->next;
  }
  if (allfail) {
    std::cout << "[UserWorkInLoop]: p =" <<std::setw(4)<<Globals::my_rank<<": failure in "<<std::setw(9)<<ifail<<" cells."<<std::endl;
  }

#ifdef MPI_PARALLEL
  int ierr;
  ierr = MPI_Allreduce(MPI_IN_PLACE,&allfail,1,MPI_C_BOOL,MPI_LOR,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qtot,nq,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qmin,nq,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qmax,nq,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&trac,3 ,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&lengrat,2,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
#endif
  for (int q=1; q<nq; q++) qtot[q] /= qtot[0];
  for (int q=1; q<3;  q++) trac[q] /= trac[0];

  if (Globals::my_rank==0) {
    if (EXPANDING_ENABLED) {
      std::cout << "[UserWorkInLoop]: rmax = " << std::scientific << std::setw(13) << std::setprecision(5) << mesh_size.x1max << std::endl;
    }
    std::cout << "[UserWorkInLoop]: lcool= " << std::scientific << std::setw(13) << std::setprecision(5) << lengrat[0]
              << " lgrv= "                   << std::scientific << std::setw(13) << std::setprecision(5) << lengrat[1]
              << std::endl;
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

  if (allfail) {
    if (Globals::my_rank == 0) {
      std::cout << "[UserWorkInLoop]: failure in " << std::setw(9) << ifail << " cells." << std::endl;
    }
    if (!(RECOVER_ENABLED)) {
      stop_this();
    }
  }
  if (dt < 1e-32) {
    std::cout << "[UserWorkInLoop]: Timestep dropped below 1e-40. Failure." << std::endl;
    stop_this();
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
  Real dang       = pin->GetReal("problem","dang"); //width of angular transition
  Real p0         = pin->GetReal("problem","p0"); // ambient pressure
  Real d0         = pin->GetReal("problem","d0"); // ambient density
  Real v0         = pin->GetReal("problem","v0"); // velocity of ejecta
  Real mwind      = pin->GetReal("problem","mwind"); // mass of wind ejecta
  Real mtidal     = pin->GetReal("problem","mtidal"); // mass of tidal ejecta
  Real tempwind   = pin->GetReal("problem","tempwind"); //temperature of wind ejecta
  Real temptidal  = pin->GetReal("problem","temptidal"); // temperature of tidal ejecta
  int  ihomol     = pin->GetOrAddInteger("problem","ihomol",0); // initial homologous expansion (no "ring")
  Real thopen     = pin->GetOrAddReal("problem","thopen",pi); // opening angle of disk ejecta. pi is full polar coverage
  Real rwindtidev = pin->GetOrAddReal("problem","rwindtidev",1.0); // ratio between wind and tidal ejecta speed.
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
  Real r03     = SQR(r0)*r0;
  Real voltidal= 4.0*PI*r03*std::sin(0.5*thopen)/3.0;
  Real volwind = 4.0*PI*r03*(1.0-std::sin(0.5*thopen))/3.0;
  Real dtidal  = mtidal/voltidal;
  Real dwind   = mwind/volwind;
  Real etidal  = temptidal*dtidal/gm1;
  Real ewind   = tempwind*dwind/gm1;
  Real e0      = p0/gm1; //seems to be ambient initial energy density
  ambdens      = d0; // global variables for boundaries
  ambprss      = p0;

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x,y,z,rad,v0rad,thet; //rad=radius from centre, thet=angle btw z-axis and rad
        if (COORDINATE_SYSTEM == "cartesian") {
          x   = pcoord->x1v(i);
          y   = pcoord->x2v(j);
          z   = pcoord->x3v(k);
          rad      = std::sqrt(SQR(x) + SQR(y) + SQR(z));
          thet   = std::acos(z/rad);
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          z = pcoord->x3v(k);
          rad    = std::sqrt(SQR(x) + SQR(y) + SQR(z));
          thet   = std::acos(z/rad);
        } else { // if (COORDINATE_SYSTEM == "spherical_polar")
          x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          rad    = std::sqrt(SQR(x) + SQR(y) + SQR(z));
          thet   = pcoord->x2v(j);
        }

        // factors for radial and polar dependence
        Real fm1rad            = 0.5*(1.0-std::tanh((rad-r0)/dr)); // radial drop off
        // this is 1 within the tidal ejecta region, 0 otherwise
        Real fm1theta          = 0.25*((1.0+std::tanh((thet-0.5*(pi-thopen))/dang))*(1.0-std::tanh((thet-0.5*(pi+thopen))/dang))); // polar drop off
        if (ihomol==1) { // ramp to make ring
          v0rad = v0*rad/r0;
        } else if (ihomol==2) { // trying exponential
          v0rad = v0*std::exp(-5.0*(r0-rad));
        } else {
          v0rad = std::min(1.5*v0*rad/r0,v0);
        }

        phydro->u(IDN,k,j,i) = d0 + (dwind-d0)*fm1rad + (dtidal-dwind)*fm1theta*fm1rad; // add the tidal ejecta on top
        Real mom0radthe = (v0rad*(1.0-rwindtidev)*fm1rad*fm1theta+v0rad*rwindtidev*fm1rad)*phydro->u(IDN,k,j,i);
        if (COORDINATE_SYSTEM == "cartesian") {
          phydro->u(IM1,k,j,i) = mom0radthe*(x/rad); //for projection on the x-axis
          phydro->u(IM2,k,j,i) = mom0radthe*(y/rad);
          phydro->u(IM3,k,j,i) = mom0radthe*(z/rad);
        } else if (COORDINATE_SYSTEM == "spherical_polar") {
          phydro->u(IM1,k,j,i) = mom0radthe;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        }
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = e0+(ewind-e0)*fm1rad + (etidal-ewind)*fm1theta*fm1rad
                                   +0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+SQR(phydro->u(IM3,k,j,i)))
                                       /phydro->u(IDN,k,j,i);
          if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
            phydro->u(IEN,k,j,i) += d0; //smh this looks like a density being added to an energy density?
        }
        if (DUAL_ENERGY) {
          phydro->u(IIE,k,j,i) = e0+(ewind-e0)*fm1rad + (etidal-ewind)*fm1theta*fm1rad; //IIE: internal energy
        }
        // tracer field: 0 is shell, 1 is tidal ejecta, 2 is wind
        phydro->u(NHYDRO-NSCALARS+1,k,j,i) = dtidal*fm1rad*fm1theta;       // tidal ejecta
        phydro->u(NHYDRO-NSCALARS+2,k,j,i) = dwind*fm1rad*(1.0-fm1theta); // wind ejecta
        phydro->u(NHYDRO-NSCALARS+0,k,j,i) = phydro->u(IDN,k,j,i)      // shell tracer
                                            * 0.25*(1.0+std::tanh((rad-0.9*r0)/(0.01*r0)))
                                                  *(1.0-std::tanh((rad-1.1*r0)/(0.01*r0)));
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


//======================================================================
//there were two diagnostic functions after this that idk if I should take them with rn
//figured I'd try to compile it without first - added below 
//======================================================================
//
//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief otf diagnostics (shell position, sphericity etc)
//========================================================================================
void MeshBlock::UserWorkInLoop(void) {
  return;
#ifdef MPI_PARALLEL
  int mpierr;
#endif
  if (NSCALARS == 3) {
    // Average shock position based on scalar. Assumes center at origin.
    int nscl=3,nsum=3; // number of scalars. 
    int nelt=nsum*(nscl+1); // quantities to calculate (rad, thet, mass)
    
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

//========================================================================================
//! \fn Real HeatCool(...)
//  \brief Applies energy change due to heating and cooling
//   Combined with brute-force resetting of ambient gas early on to prevent super-luminal precursor.
//========================================================================================
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons){

  if (iheatcool > 0) {
    // From Lippuner & Roberts 2015, Fig. 4. Assumed that M*epsilon = 10^40 erg/s, thus epsilon = 10^40/2e31.
    // We need edot in erg/(cm^3*s), therefore edot = epsilon*rho = epsilon*n*mu*matom. Assuming mu=150 (see L&R15). 
    // Since L&R15 provide heating rate at 1 day, we shift the exponential decay. We also assume a decay rate of 1/(50 days).
    Real edot = 2.73e18; // Lippuner & Robert 2015, Fig 4. edot = epsilon * rho, [epsilon] = erg (g*s)^(-1). edot is in code units.
    Real day  = 2.89e-11; // day in code units.
    Real t0   = 1.44e-9 ; // decay rate of 50 days in code units
    Real deps = edot*std::exp(-(time-day)/t0)*dt;
    Real w[NHYDRO];
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
#pragma omp simd
          for (int n=0; n<NHYDRO; ++n)
            w[n]=prim(n,k,j,i);
          // here we can get more detailed regarding different composition
          Real denergy = (w[NHYDRO-NSCALARS+1]+w[NHYDRO-NSCALARS+2])*w[IDN]*deps;
          cons(IEN,k,j,i) += denergy;
          if (DUAL_ENERGY) 
            cons(IIE,k,j,i) += denergy;
        }
      }
    }
  } 
  if (time < time_free) {
    Real gam = pmb->peos->GetGamma();
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real ceject = std::min(1.0,prim(NHYDRO-NSCALARS+1,k,j,i)+prim(NHYDRO-NSCALARS+2,k,j,i));
          if (ceject < 1e-6) {
            cons(IDN,k,j,i) = ambdens;
            cons(IM1,k,j,i) = 0.0;
            cons(IM2,k,j,i) = 0.0;
            cons(IM3,k,j,i) = 0.0;
            cons(IEN,k,j,i) = ambprss/(gam-1.0);
            cons(IIE,k,j,i) = ambprss/(gam-1.0);
            cons(NHYDRO-NSCALARS+0,k,j,i) = 0.0;
            cons(NHYDRO-NSCALARS+1,k,j,i) = 0.0;
            cons(NHYDRO-NSCALARS+2,k,j,i) = 0.0;
          }
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn Real HeatCoolTimestep(...)
//  \brief Modifies timestep to account for cooling time. 
//========================================================================================
Real HeatCoolTimeStep(MeshBlock *pmb){
  Real dtcool = HUGE_NUMBER;
  return dtcool; // short-circuit for now, since it's only heating. 

  Real edot = 1e5; //2.73e18; // Lippuner & Robert 2015, Fig 4. edot = epsilon * rho, [epsilon] = erg (g*s)^(-1). edot is in code units.
  Real day  = 0.0;//2.89e-11; // day in code units.
  Real t0   = 1.44e-9 ; // decay rate of 50 days in code units
  Real deps = edot*std::exp(-(pmb->pmy_mesh->time-day)/t0);
  AthenaArray<Real> prim;
  prim.InitWithShallowCopy(pmb->phydro->w);
  //Real dtcool = HUGE_NUMBER;
  Real temp, w[NHYDRO];
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
#pragma omp simd
        for (int n=0; n<NHYDRO; ++n)
          w[n]=prim(n,k,j,i);
        if (DUAL_ENERGY) {
          temp = std::max(1e4,w[IIE]/w[IDN]);
        } else {
          temp = std::max(1e4,w[IPR]/w[IDN]);
        }
        Real denergy = (w[NHYDRO-NSCALARS+1]+w[NHYDRO-NSCALARS+2])*w[IDN]*deps;
        Real dttmp = 0.3*temp/(fabs(denergy)+1e-60);
        dtcool = std::min(dtcool,dttmp);
      }
    }
  }
  return dtcool;
}

//========================================================================================
//! \fn Real RampTimeStep(...)
//  \brief Lowers timestep at start of simulation to account for extreme ICs. 
//========================================================================================
Real RampTimeStep(MeshBlock *pmb){
  Real ramp   = pmb->pmy_mesh->time/tramp0;
  ramp        = ramp > 1.0 ? HUGE_NUMBER : ramp;
  Real dtramp = dtramp0 + ramp*(dtramp1-dtramp0);
  return dtramp;
}



