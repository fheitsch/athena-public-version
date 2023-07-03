//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sync_eint.cpp
//  \brief Syncs internal with total energy

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

// Athena++ headers
#include "hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::SyncEint
//  \brief Sync internal energy with total energy, if necessary 

void Hydro::SyncEint(AthenaArray<Real> &u) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Real i2 = pmb->peos->GetIeta2(); 
  Real eint, emax, ekin, emag=0.0;  
  int dim = 1; 

  // Get dimension
  if (pmb->block_size.nx2 > 1) dim = 2;
  if (pmb->block_size.nx3 > 1) dim = 3;  

  if (MAGNETIC_FIELDS_ENABLED) {
    pmb->pfield->CalculateCellCenteredField(pmb->pfield->b, pmb->pfield->bcc,
                                            pmb->pcoord, is, ie, js, je, ks, ke);
  }

  // 1D-case
  if ( dim == 1 ) {
//#pragma omp simd
    for (int i=is; i<=ie; ++i) {
      ekin = 0.5*(SQR(u(IVX,ks,js,i)))/u(IDN,ks,js,i);
      if (MAGNETIC_FIELDS_ENABLED) 
        emag = 0.5*( SQR(pmb->pfield->bcc(0,ks,js,i))
                    +SQR(pmb->pfield->bcc(1,ks,js,i))
                    +SQR(pmb->pfield->bcc(2,ks,js,i))); 
      eint = u(IEN,ks,js,i) - ekin - emag;
      emax = u(IEN,ks,js,i); 
      emax = std::max(emax,u(IEN,ks,js,std::max(is,i-1)));
      emax = std::max(emax,u(IEN,ks,js,std::min(ie,i+1)));

      if ((eint/emax > i2) && (eint > 0.0)) {
        u(IIE,ks,js,i) = eint; 
      }
    }
  }
  // 2D-case
  else if ( dim == 2 ) {
    for (int j=js; j<=je; ++j) {
//#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        ekin = 0.5*(  SQR(u(IM1,ks,j,i))
                    + SQR(u(IM2,ks,j,i)))/u(IDN,ks,j,i);
        if (MAGNETIC_FIELDS_ENABLED) 
          emag = 0.5*( SQR(pmb->pfield->bcc(0,ks,j,i))
                      +SQR(pmb->pfield->bcc(1,ks,j,i))
                      +SQR(pmb->pfield->bcc(2,ks,j,i))); 
        eint = u(IEN,ks,j,i) - ekin - emag;
        emax = u(IEN,ks,j,i); 
				
        // Can't do this with for-loop bc of pragma directive
        emax = std::max(emax,u(IEN,ks,j-1,i-1));
        emax = std::max(emax,u(IEN,ks,j-1,i  ));
        emax = std::max(emax,u(IEN,ks,j-1,i+1));
        emax = std::max(emax,u(IEN,ks,j  ,i-1));
        emax = std::max(emax,u(IEN,ks,j  ,i  ));
        emax = std::max(emax,u(IEN,ks,j  ,i+1));
        emax = std::max(emax,u(IEN,ks,j+1,i-1));
        emax = std::max(emax,u(IEN,ks,j+1,i  ));
        emax = std::max(emax,u(IEN,ks,j+1,i+1));
				
        if ((eint/emax > i2) && (eint > 0.0)) {
        //std::cout << "[SyncEint]: k " << ks << " j " << j  << " i "  << i << " eint=" << eint
        //          << " IE: " << u(IIE,ks,j,i) << " ieta2: " << i2 << std::endl;
          u(IIE,ks,j,i) = eint; 

        }
      }
    } 
  }
  // 3D-case
  else {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
//#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          ekin = 0.5*( SQR(u(IM1,k,j,i))
                      +SQR(u(IM2,k,j,i)) 
                      +SQR(u(IM3,k,j,i)))/u(IDN,k,j,i);
          if (MAGNETIC_FIELDS_ENABLED)
            emag = 0.5*( SQR(pmb->pfield->bcc(0,k,j,i))
                        +SQR(pmb->pfield->bcc(1,k,j,i))
                        +SQR(pmb->pfield->bcc(2,k,j,i)));
          eint = u(IEN,k,j,i) - ekin - emag;
          emax = u(IEN,k,j,i); 
          for (int kk=-1; kk<=1; kk++) {
            int k1 = k+kk;
            for (int jj=-1; jj<=1; jj++) {
              int j1 = j+jj;
              for (int ii=-1; ii<=1; ii++) {
                int i1 = i+ii;
                emax = std::max(emax,u(IEN,k1,j1,i1));
              }
            }				
          }
          if ((eint/emax > i2) && (eint > 0.0)) {
            u(IIE,k,j,i) = eint; 
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CheckEint
//  \brief Check if internal energy is necessary at start of each substep  

void Hydro::CheckEint(AthenaArray<Real> &u) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Real i1 = pmb->peos->GetIeta1(); 
  Real eint, etot, ekin, emag=0.0;  
  int dim = 1; 

  // Get dimension
  if (pmb->block_size.nx2 > 1) dim = 2;
  if (pmb->block_size.nx3 > 1) dim = 3;  


  // 1D-case
  if ( dim == 1 ) {
    if (MAGNETIC_FIELDS_ENABLED) 
      pmb->pfield->CalculateCellCenteredField(pmb->pfield->b, pmb->pfield->bcc,
                                              pmb->pcoord, is-NGHOST, ie+NGHOST, 
                                                           js       , je       , 
                                                           ks       , ke       );
#pragma omp simd
    for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {

      etot = u(IEN,ks,js,i);
      ekin = 0.5*(SQR(u(IM1,ks,js,i)))/u(IDN,ks,js,i);
      if (MAGNETIC_FIELDS_ENABLED)
        emag = 0.5*( SQR(pmb->pfield->bcc(0,ks,js,i))
                    +SQR(pmb->pfield->bcc(1,ks,js,i))
                    +SQR(pmb->pfield->bcc(2,ks,js,i)));
      eint = etot - ekin - emag;
      if ( ((eint/etot) < i1) || (etot <= 0.0) || (eint <= 0.0) ) {
        u(IEN,ks,js,i)  = u(IIE,ks,js,i);
        u(IEN,ks,js,i) += (ekin + emag); 
      }
    }
  }
  // 2D-case
  else if ( dim == 2 ) {
    if (MAGNETIC_FIELDS_ENABLED) 
      pmb->pfield->CalculateCellCenteredField(pmb->pfield->b, pmb->pfield->bcc,
                                              pmb->pcoord, is-NGHOST, ie+NGHOST, 
                                                           js-NGHOST, je+NGHOST, 
                                                           ks       , ke      );
    for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
#pragma omp simd
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        etot = u(IEN,ks,j,i);
        ekin = 0.5*(  SQR(u(IM1,ks,j,i))
                    + SQR(u(IM2,ks,j,i)) )/u(IDN,ks,j,i);
        if (MAGNETIC_FIELDS_ENABLED)
          emag = 0.5*( SQR(pmb->pfield->bcc(0,ks,j,i))
                      +SQR(pmb->pfield->bcc(1,ks,j,i))
                      +SQR(pmb->pfield->bcc(2,ks,j,i)));
        eint = etot - ekin - emag;
        if ( ((eint/etot) < i1) || (etot <= 0.0) || (eint <= 0.0) ) {
          u(IEN,ks,j,i)  = u(IIE,ks,j,i);
          u(IEN,ks,j,i) += (ekin + emag); 
        }
      }
    }
  }
  // 3D-case
  else {
    if (MAGNETIC_FIELDS_ENABLED) 
      pmb->pfield->CalculateCellCenteredField(pmb->pfield->b, pmb->pfield->bcc,
                                              pmb->pcoord, is-NGHOST, ie+NGHOST, 
                                                           js-NGHOST, je+NGHOST, 
                                                           ks-NGHOST, ke+NGHOST);
    for (int k=ks-NGHOST; k<=ke+NGHOST; ++k) {
      for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
#pragma omp simd
        for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
          etot = u(IEN,k,j,i); 
          ekin = 0.5*(  SQR(u(IM1,k,j,i))
                      + SQR(u(IM2,k,j,i)) 
                      + SQR(u(IM3,k,j,i)) )/u(IDN,k,j,i);
          if (MAGNETIC_FIELDS_ENABLED)
            emag = 0.5*( SQR(pmb->pfield->bcc(0,k,j,i))
                        +SQR(pmb->pfield->bcc(1,k,j,i))
                        +SQR(pmb->pfield->bcc(2,k,j,i)));
          eint = etot - ekin - emag; 
          if ( ((eint/etot) < i1) || (etot <= 0.0) || (eint <= 0.0) ) {
            u(IEN,k,j,i)  = u(IIE,k,j,i);
            u(IEN,k,j,i) += (ekin + emag);
          }
        }
      }
    }
  }
  return;
}
