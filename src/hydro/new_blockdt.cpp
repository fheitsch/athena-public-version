//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt.cpp
//  \brief computes timestep using CFL condition on a MEshBlock

// C/C++ headers
#include <algorithm>  // min()
#include <cfloat>     // FLT_MAX
#include <cmath>      // fabs(), sqrt()

// Athena++ headers
#include "hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../cless/cless.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
// \!fn Real Hydro::NewBlockTimeStep(void)
// \brief calculate the minimum timestep within a MeshBlock

Real Hydro::NewBlockTimeStep(void) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> w,bcc,b_x1f,b_x2f,b_x3f, wcl;
  w.InitWithShallowCopy(pmb->phydro->w);
  if (MAGNETIC_FIELDS_ENABLED) {
    bcc.InitWithShallowCopy(pmb->pfield->bcc);
    b_x1f.InitWithShallowCopy(pmb->pfield->b.x1f);
    b_x2f.InitWithShallowCopy(pmb->pfield->b.x2f);
    b_x3f.InitWithShallowCopy(pmb->pfield->b.x3f);
  }
  if (CLESS_ENABLED) {
    wcl.InitWithShallowCopy(pmb->pcless->w); 
  }
  // expanding grid
  AthenaArray<Real> &evel1 = pmb->pex->vv[(IVX-1)]; // need the volume velocities here
  AthenaArray<Real> &evel2 = pmb->pex->vv[(IVY-1)];
  AthenaArray<Real> &evel3 = pmb->pex->vv[(IVZ-1)];
  Real ev1=0.0, ev2=0.0, ev3=0.0;
  const Real safe = 8.0;

  AthenaArray<Real> dt1, dt2, dt3;
  dt1.InitWithShallowCopy(dt1_);
  dt2.InitWithShallowCopy(dt2_);
  dt3.InitWithShallowCopy(dt3_);
  //Real wi[(NWAVE+NINT)];
  Real wi[(NHYDRO+2)]; // all hydro variables and the two transverse field components
  Real wicl[(NWAVECL)]; 

  Real min_dt = (FLT_MAX);
  if (TIMESTEPINFO_ENABLED) {
    for (int l=0; l<pmb->ndt; l++) {
      pmb->all_min_dts(l) = (FLT_MAX); 
      for (int m=0; m<3; m++) {
        pmb->all_min_loc(l,m) = -1e60;
        pmb->all_min_ind(l,m) = -1;
      }
    }
  }

  for (int k=ks; k<=ke; ++k) {
    if (EXPANDING_ENABLED) {
      if (pmb->pex->x3Move) ev3 = safe*std::max(fabs(evel3(k)),fabs(evel3(k+1)));  
    }
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CenterWidth1(k,j,is,ie,dt1);
      pmb->pcoord->CenterWidth2(k,j,is,ie,dt2);
      pmb->pcoord->CenterWidth3(k,j,is,ie,dt3);
      if (EXPANDING_ENABLED) {
        if (pmb->pex->x2Move) ev2 = safe*std::max(fabs(evel2(j)),fabs(evel2(j+1)));
      }
      if (!RELATIVISTIC_DYNAMICS) {
#pragma ivdep
        for (int i=is; i<=ie; ++i) {
          wi[IDN]=w(IDN,k,j,i);
          wi[IVX]=w(IVX,k,j,i);
          wi[IVY]=w(IVY,k,j,i);
          wi[IVZ]=w(IVZ,k,j,i);
          if (NON_BAROTROPIC_EOS) {
            wi[IPR]=w(IPR,k,j,i);
            if (DUAL_ENERGY) wi[IGE]=w(IGE,k,j,i);
          }

          if (EXPANDING_ENABLED) {
            if (pmb->pex->x1Move) ev1 = safe*std::max(fabs(evel1(i)),fabs(evel1(i+1)));
          }
					
          if (CLESS_ENABLED) { // cless + hydro
            Real c1f, c2f, c3f; 
            wicl[IDN ] = wcl(IDN ,k,j,i);
            wicl[IVX ] = wcl(IVX ,k,j,i);
            wicl[IVY ] = wcl(IVY ,k,j,i);
            wicl[IVZ ] = wcl(IVZ ,k,j,i);
            wicl[IP11] = wcl(IP11,k,j,i);
            wicl[IP22] = wcl(IP22,k,j,i);
            wicl[IP33] = wcl(IP33,k,j,i);
            wicl[IP12] = wcl(IP12,k,j,i);
            wicl[IP13] = wcl(IP13,k,j,i);
            wicl[IP23] = wcl(IP23,k,j,i);

            pmb->peos->SoundSpeedsCL(wicl,&c1f,&c2f,&c3f); 

            Real cs = pmb->peos->SoundSpeed(wi); 

            dt1(i) /= std::max( fabs(wi[IVX] + cs), fabs(wicl[IVX] + c1f) );
            dt2(i) /= std::max( fabs(wi[IVY] + cs), fabs(wicl[IVY] + c2f) );
            dt3(i) /= std::max( fabs(wi[IVZ] + cs), fabs(wicl[IVZ] + c3f) ); 
          }
          else if (MAGNETIC_FIELDS_ENABLED) { // hydro + mhd

            Real bx = bcc(IB1,k,j,i) + fabs(b_x1f(k,j,i)-bcc(IB1,k,j,i));
            wi[IBY] = bcc(IB2,k,j,i);
            wi[IBZ] = bcc(IB3,k,j,i);
            Real cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
            dt1(i) /= (fabs(wi[IVX]) + cf + ev1);

            wi[IBY] = bcc(IB3,k,j,i);
            wi[IBZ] = bcc(IB1,k,j,i);
            bx = bcc(IB2,k,j,i) + fabs(b_x2f(k,j,i)-bcc(IB2,k,j,i));
            cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
            dt2(i) /= (fabs(wi[IVY]) + cf + ev2);

            wi[IBY] = bcc(IB1,k,j,i);
            wi[IBZ] = bcc(IB2,k,j,i);
            bx = bcc(IB3,k,j,i) + fabs(b_x3f(k,j,i)-bcc(IB3,k,j,i));
            cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
            dt3(i) /= (fabs(wi[IVZ]) + cf + ev3);

          } 
          else { // hydro only 

            Real cs = pmb->peos->SoundSpeed(wi);

            if (TIMESTEPINFO_ENABLED) {
              if (dt1(i)/cs < pmb->all_min_dts(0)) { // sound speed
                pmb->all_min_dts(0)   = dt1(i)/cs; 
                pmb->all_min_loc(0,0) = pmb->pcoord->x1v(i);
                pmb->all_min_loc(0,1) = pmb->pcoord->x2v(j);
                pmb->all_min_loc(0,2) = pmb->pcoord->x3v(k);
                pmb->all_min_ind(0,0) = i;
                pmb->all_min_ind(0,1) = j;
                pmb->all_min_ind(0,2) = k;
              }
              if (fabs(dt1(i)/wi[IVX]) < pmb->all_min_dts(1)) { // x1-velocity
                pmb->all_min_dts(1)   = fabs(dt1(i)/wi[IVX]); 
                pmb->all_min_loc(1,0) = pmb->pcoord->x1v(i);
                pmb->all_min_loc(1,1) = pmb->pcoord->x2v(j);
                pmb->all_min_loc(1,2) = pmb->pcoord->x3v(k);
                pmb->all_min_ind(1,0) = i;
                pmb->all_min_ind(1,1) = j;
                pmb->all_min_ind(1,2) = k;
              }
              if (fabs(dt2(i)/wi[IVY]) < pmb->all_min_dts(2)) { // x2-velocity
                pmb->all_min_dts(2)   = fabs(dt2(i)/wi[IVY]);
                pmb->all_min_loc(2,0) = pmb->pcoord->x1v(i);
                pmb->all_min_loc(2,1) = pmb->pcoord->x2v(j);
                pmb->all_min_loc(2,2) = pmb->pcoord->x3v(k);
                pmb->all_min_ind(2,0) = i;
                pmb->all_min_ind(2,1) = j;
                pmb->all_min_ind(2,2) = k;
              }
              if (fabs(dt3(i)/wi[IVZ]) < pmb->all_min_dts(3)) { // x3-velocity
                pmb->all_min_dts(3)   = fabs(dt3(i)/wi[IVZ]);
                pmb->all_min_loc(3,0) = pmb->pcoord->x1v(i);
                pmb->all_min_loc(3,1) = pmb->pcoord->x2v(j);
                pmb->all_min_loc(3,2) = pmb->pcoord->x3v(k);
                pmb->all_min_ind(3,0) = i;
                pmb->all_min_ind(3,1) = j;
                pmb->all_min_ind(3,2) = k;
              }
            }

            dt1(i) /= (fabs(wi[IVX]) + cs + ev1);
            dt2(i) /= (fabs(wi[IVY]) + cs + ev2);
            dt3(i) /= (fabs(wi[IVZ]) + cs + ev3);

          }
        }
      }

      // compute minimum of (v1 +/- C)
      for (int i=is; i<=ie; ++i) {
        Real& dt_1 = dt1(i);
        min_dt = std::min(min_dt,dt_1);
      }

      // if grid is 2D/3D, compute minimum of (v2 +/- C)
      if (pmb->block_size.nx2 > 1) {
        for (int i=is; i<=ie; ++i) {
          Real& dt_2 = dt2(i);
          min_dt = std::min(min_dt,dt_2);
        }
      }

      // if grid is 3D, compute minimum of (v3 +/- C)
      if (pmb->block_size.nx3 > 1) {
        for (int i=is; i<=ie; ++i) {
          Real& dt_3 = dt3(i);
          min_dt = std::min(min_dt,dt_3);
        }
      }

    }
  }

// calculate the timestep limited by the diffusion process
  if (phdif->hydro_diffusion_defined) {
    Real mindt_vis, mindt_cnd;
    phdif->NewHydroDiffusionDt(mindt_vis, mindt_cnd);
    min_dt = std::min(min_dt,mindt_vis);
    min_dt = std::min(min_dt,mindt_cnd);
  } // hydro diffusion

  if(MAGNETIC_FIELDS_ENABLED &&
     pmb->pfield->pfdif->field_diffusion_defined) {
    Real mindt_oa, mindt_h;
    pmb->pfield->pfdif->NewFieldDiffusionDt(mindt_oa, mindt_h);
    min_dt = std::min(min_dt,mindt_oa);
    min_dt = std::min(min_dt,mindt_h);
  } // field diffusion
  
  min_dt *= pmb->pmy_mesh->cfl_number;

  //fprintf(stdout,"[NewBlockTimeStep]: min_dt before = %13.5e\n",min_dt);

  if (UserTimeStep_!=NULL) {
    min_dt = std::min(min_dt, UserTimeStep_(pmb));
  }
  if (EXPANDING_ENABLED) {
    min_dt = std::min(min_dt, pmb->pex->GridTimeStep(pmb));
  }

  //fprintf(stdout,"[NewBlockTimeStep]: min_dt after  = %13.5e\n",min_dt);

  //fprintf(stdout,"[new_block_dt]: WARNING WARNING WARNING: timestep set to 1e-2\n");
  //min_dt = 1e-2;
  pmb->new_block_dt=min_dt;

  return min_dt;
}
