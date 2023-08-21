// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
/*! \file bgk2.c
 *  \brief Computes 1D fluxes using the BGK scheme.
 *
 * PURPOSE: Computes 1D fluxes using the 2nd order BGK scheme.
 *   See von Karman lectures by Kun Xu. 
 *
 * REFERENCES:
 * - Prendergast & Xu 1993, Slyz et al. 1998, Xu 2001.
 *
 * HISTORY: Recoded from Proteus' dxeesplitgrv3d.F
     by Zachary Mitchell, April 2013.
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 * - fluxes() - all Riemann solvers in Athena must have this function name and
 *              use the same argument list as defined in rsolvers/prototypes.h*/
/*========================================================================================*/

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()
#include <iostream>


// Athena++ headers
#include "../../hydro.hpp"
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../eos/eos.hpp"

#define FLUXES_RECON
#define FLUXES_G0
#define FLUXES_ABAR
//#define DEBUG_ALL
//

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//! \brief The HLLC Riemann solver for adiabatic hydrodynamics (use HLLE for isothermal)

//private function
void dxe3d(Real axu, Real ayu, Real azu, Real al ,
           Real sd , Real sx , Real sy , Real sz , Real sl ,
           Real *qd, Real *qx, Real *qy, Real *qz, Real *ql,
           Real ck, Real ddim);



void Hydro::RiemannSolver(const int kl, const int ku, const int jl, const int ju,
  const int il, const int iu, const int ivx, const AthenaArray<Real> &bx,
  AthenaArray<Real> &wl, AthenaArray<Real> &wr, AthenaArray<Real> &flx,
  AthenaArray<Real> &ey, AthenaArray<Real> &ez) {

  MeshBlock *pmb = pmy_block;
  Mesh *pmesh = pmb->pmy_mesh;
  Real tau=pmesh->dt; 
  Real bgkc1 = pmb->peos->GetBgkC1();
  Real bgkc2 = pmb->peos->GetBgkC2();

  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real Gamma_1 = pmy_block->peos->GetGamma() - 1.0;
  Real iGamma_1 = 1.0/Gamma_1;

  int im1 = ivx;
  int im2 = IVX + ((im1-IM1)+1)%3;
  int im3 = IVX + ((im1-IM1)+2)%3;

  AthenaArray<Real> x123f, dx123v;

#ifdef DEBUG_ALL
  fprintf(stdout,"[bgk2]: called with bgkc1=%13.5e bgkc2=%13.5e\n",bgkc1,bgkc2);
#endif

  //Offsets in inactive coordinates introduced for conservative variables in cell walls.
  int ioff, joff, koff;
  if (ivx==1) {
    ioff=1;
    joff=0;
    koff=0;
    x123f.InitWithShallowCopy(pmb->pcoord->x1f);
    dx123v.InitWithShallowCopy(pmb->pcoord->dx1f);
  }
  else if (ivx==2) {
    ioff=0;
    joff=1;
    koff=0;
    x123f.InitWithShallowCopy(pmb->pcoord->x2f);
    dx123v.InitWithShallowCopy(pmb->pcoord->dx2f);
  }
  else if (ivx==3) {
    ioff=0;
    joff=0;
    koff=1;
    x123f.InitWithShallowCopy(pmb->pcoord->x3f);
    dx123v.InitWithShallowCopy(pmb->pcoord->dx3f);
  }

  Real x1f=0.0,dxx;

  for (int k=kl; k<=ku; ++k) {
    if (ivx==1) dxx = pmb->pcoord->dx3f(k);
  for (int j=jl; j<=ju; ++j) {
    if (ivx==2) dxx = pmb->pcoord->dx2f(j);
  for (int i=il; i<=iu; ++i) {
    if (ivx==1) dxx = pmb->pcoord->dx1f(i);

    int kdif=k-koff;
    int jdif=j-joff;
    int idif=i-ioff;

//--- Step 1.  Load L/R states into local variables

// Originally the following declarations were passed as arguments to the 
// program, but now they are contained, or can be obtained, from the 
// primitive variables w(i-1) and w(i). flx(i) is on the wall between i-1 and i. 
    Real ad[2],ax[2],ay[2],az[2],ap[2],ae[2],al[2];
    Real ade1,axm1,aym1,azm1,axu1,ayu1,azu1,apr1,aen1,ae1,rade1,rae1,sae1;
    Real ade2,axm2,aym2,azm2,axu2,ayu2,azu2,apr2,aen2,ae2,rade2,rae2,sae2;
    Real ck, dof, ddim = 3.0;
    Real pi = 4.0*atan(1.0);
    Real gra1 = 0.0, gra2 = 0.0;
    dof = (-3.0*(Gamma_1+1.0)+5.0)/Gamma_1;
    ck  = dof + 3.0-ddim;

    // Get the conserved quantities at the center of cell "i-1"
    ad[0] = pmb->phydro->w(IDN,kdif,jdif,idif);
    ax[0] = pmb->phydro->w(ivx,kdif,jdif,idif)*ad[0];
    ay[0] = pmb->phydro->w(ivy,kdif,jdif,idif)*ad[0];
    az[0] = pmb->phydro->w(ivz,kdif,jdif,idif)*ad[0];
    ap[0] = pmb->phydro->w(IPR,kdif,jdif,idif); // this is the pressure
    al[0] = 0.25*((ck+3.0)*ad[0])/(iGamma_1*ap[0]);
//    al[0] = 0.5*ad[0]/ap[0]; // lambda (inverse temperature)
    ae[0] = ap[0]/Gamma_1 + 0.5*(SQR(ax[0])+SQR(ay[0])+SQR(az[0]))/ad[0]; // total energy

    // Get the conserved quantities at the center of cell "i"
    ad[1] = pmb->phydro->w(IDN,k,j,i);
    ax[1] = pmb->phydro->w(ivx,k,j,i)*ad[1];
    ay[1] = pmb->phydro->w(ivy,k,j,i)*ad[1];
    az[1] = pmb->phydro->w(ivz,k,j,i)*ad[1];
    ap[1] = pmb->phydro->w(IPR,k,j,i); // this is the pressure
    al[1] = 0.25*((ck+3.0)*ad[1])/(iGamma_1*ap[1]);
//    al[1] = 0.5*ad[1]/ap[1]; // lambda (inverse temperature)
    ae[1] = ap[1]/Gamma_1 + 0.5*(SQR(ax[1])+SQR(ay[1])+SQR(az[1]))/ad[1]; // total energy

    // conservative variables at cell walls (left): 
    ade1  = wl(IDN,k,j,i);
    axu1  = wl(ivx,k,j,i);
    ayu1  = wl(ivy,k,j,i);
    azu1  = wl(ivz,k,j,i);
    apr1  = wl(IPR,k,j,i);
    axm1  = axu1*ade1;
    aym1  = ayu1*ade1;
    azm1  = azu1*ade1;
    aen1  = apr1/Gamma_1 + 0.5*ade1*(SQR(axu1) + SQR(ayu1) + SQR(azu1));
    ae1 = 0.25*((ck+3.0)*ade1)/(iGamma_1*apr1);
//    ae1   = 0.5*ade1/apr1;
    
    // conservative variables at cell walls (right):
    ade2  = wr(IDN,k,j,i);
    axu2  = wr(ivx,k,j,i);
    ayu2  = wr(ivy,k,j,i);
    azu2  = wr(ivz,k,j,i);
    apr2  = wr(IPR,k,j,i);
    axm2  = axu2*ade2;
    aym2  = ayu2*ade2;
    azm2  = azu2*ade2;
    aen2  = apr2/Gamma_1 + 0.5*ade2*(SQR(axu2) + SQR(ayu2) + SQR(azu2));
    ae2 = 0.25*((ck+3.0)*ade2)/(iGamma_1*apr2);
//    ae2   = 0.5*ade2/apr2;
   
#ifdef DEBUG_ALL
    fprintf(stdout,"    initcond : i=%4i x1f=%13.5e dens: ad[0]=%13.5e,ade1=%13.5e,ade2=%13.5e,ad[1]=%13.5e\n", i,x1f,ad[0],ade1,ade2,ad[1]);
    fprintf(stdout,"    initcond : i=%4i x1f=%13.5e etot: ae[0]=%13.5e,aen1=%13.5e,aen2=%13.5e,ae[1]=%13.5e\n", i,x1f,ae[0],aen1,aen2,ae[1]);
    fprintf(stdout,"    initcond : i=%4i x1f=%13.5e momx: ax[0]=%13.5e,axm1=%13.5e,axm2=%13.5e,ax[1]=%13.5e\n", i,x1f,ax[0],axm1,axm2,ax[1]);
    fprintf(stdout,"    initcond : i=%4i x1f=%13.5e lamb: al[0]=%13.5e,ae1 =%13.5e,ae2 =%13.5e,al[1]=%13.5e\n", i,x1f,al[0],ae1 ,ae2 ,al[1]);
    fprintf(stdout,"    initcond : i=%4i x1f=%13.5e lamb: ap[0]=%13.5e,apr1=%13.5e,apr2=%13.5e,ap[1]=%13.5e\n", i,x1f,ap[0],apr1,apr2,ap[1]);
#endif

#ifdef DEBUG_ALL
    fprintf(stdout," VAR CHECK!!! axu1=%13.5e, ayu1=%13.5e, azu1=%13.5e, al_l=%13.5e\n", axu1,ayu1,azu1,ae1);
    fprintf(stdout," VAR CHECK!!! axu2=%13.5e, ayu2=%13.5e, azu2=%13.5e, al_r=%13.5e\n", axu2,ayu2,azu2,ae2);
#endif
//------------------------------------------------------------------------------------------------------
// Declare the moments 
/* Convention:
   T   : Moment
   E/G : left/right side of the cell wall
   U   : denotes moment of velocity half-space
   #   : number denotes order of moment
   Y/Z : denotes full moment in an inactive direction
   I   : denotes full moment of the internal degrees of freedom
   Examples:
       TEU[2]: 2nd order half-moment of velocity along the active 
               direction on the left side of the cell wall.
       TGU1Y2: a combined moment (two moments multiplied together),
               first order velocity half-moment along the active direction
               on the right side and the second order full moment along the 
               inactive y-direction.
       EZ1   : full first order moment of the inactive velocity component 
               in the z-direction
*/


// Half-moments up to the sixth order
// Note: For consistency with FORTRAN version (and with order number), we
// define 7 elements per moment array, and use 1:6 (instead of 6, with 0:5).
  Real teu[7],tgu[7];

// Combined moments on the left side
  Real te1,
       ey1,ey2,ey3,ey4,ey5,
       ez0,ez1,ez2,ez3,ez4,ez5,
       ei2,ei4,
       te1y1,te1y2,te1y3,te1y4,te1y5,
       te1z1,te1z2,
       te1i2,te1i4,
       te1y1i2,te1y2i2,te1y1i4,te1y3i2,
       teu1y1,teu1y2,teu1y3,teu1y4,
       teu1z1,teu1z2,teu1z3,teu1z4,
       teu1i2,teu1i4,
       teu1y1i2,teu1y2i2,teu1y1i4,teu1y3i2,
       teu1z1i2,teu1z2i2,teu2z1i2,teu2z2i2,
       teu2y1,teu2y2,teu2y3,teu2y4,
       teu2z1,teu2z2,teu2z3,teu2z4,
       teu2i2,teu2i4,
       teu2y1i2,teu2y2i2,
       teu3y1,teu3y2,teu3y3,
       teu3z1,teu3z2,
       teu3i2,
       teu3y1i2,
       teu4y1,teu4y2,
       teu4z1,teu4z2,
       teu4i2,
       teu5y1,
       teu1y1z1,teu1y1z2,
       teu1y2z1,teu1y2z2,teu2y1z1,teu2y1z2,teu2y2z1,teu2y2z2;
    
// Combined moments on the right side
  Real tg1,
       gy1,gy2,gy3,gy4,gy5,
       gz0,gz1,gz2,gz3,gz4,gz5,
       gi2,gi4,
       tg1y1,tg1y2,tg1y3,tg1y4,tg1y5,
       tg1z1,tg1z2,
       tg1i2,tg1i4,
       tg1y1i2,tg1y2i2,tg1y1i4,tg1y3i2,
       tgu1y1,tgu1y2,tgu1y3,tgu1y4,
       tgu1z1,tgu1z2,tgu1z3,tgu1z4,
       tgu1i2,tgu1i4,
       tgu1y1i2,tgu1y2i2,tgu1y1i4,tgu1y3i2,
       tgu1z1i2,tgu1z2i2,tgu2z1i2,tgu2z2i2,
       tgu2y1,tgu2y2,tgu2y3,tgu2y4,
       tgu2z1,tgu2z2,tgu2z3,tgu2z4,
       tgu2i2,tgu2i4,
       tgu2y1i2,tgu2y2i2,
       tgu3y1,tgu3y2,
       tgu3z1,tgu3z2,
       tgu3i2,
       tgu3y1i2,
       tgu4y1,tgu4y2,
       tgu4z1,tgu4z2,
       tgu4i2,
       tgu5y1,
       tgu1y1z1,tgu1y1z2,
       tgu1y2z1,tgu1y2z2,tgu2y1z1,tgu2y1z2,tgu2y2z1,tgu2y2z2;

// moments for g0
  Real ade,axm,aym,azm,aen,rade,ae0,rae0,axu0,
       ayu0,azu0,te,st,sw,se1,set0,set1,set2,fmi1,fmi2,
       fpi1,fpi2,fpix1,fpix2,fpiz1,fpiz2,
       fei1,fei2,ami,api,apix,apiz,aei,fm1,fm2,fp1,fp2,
       fpx1,fpx2,fpz1,fpz2,fe1,fe2,t0,t1,t2,t3,t4,t5,
       t6,t02,t12,t22,t32,t42,t04,t14,t24,
       t1y1,t2y1,t3y1,t4y1,t5y1,
       t02y1,t12y1,t22y1,t32y1,t04y1,t14y1,t1y2,t2y2,
       t3y2,t4y2,t02y2,t12y2,t22y2,t1y3,t2y3,t3y3,t02y3,
       t12y3,t1y4,t2y4,t1y5,te2,awl,bxwl,bywl,bzwl,cwl,
       awr,bxwr,bywr,bzwr,cwr,triu,triuy1,triuz1,triu2,triu2y1,
       triu2z1,triu3,trieu,trieu2,as,bs,bsx,bsz,cs,atriu,atriuy1,
       atriuz1,atriu2,atrieu,a,b,d,afm,afp,afpx,afpz,afe,
       fmg0,fpg0,fpxg0,fpzg0,feg0,fmAbar,fpAbar,fpxAbar,fpzAbar,feAbar,
       a1,b1,c1,f1,d1,aw11,bxw11,byw11,bzw11,
       cw11,aw22,bxw22,byw22,bzw22,cw22,
       tg2,ey0,gy0,aa1,aa2,y0,y1,y2,y3,y4,y5;
  Real z0,z1,z2,z3,z4,z5,t1z1,t2z1,t3z1,t12z1,t1z2,t2z2,t3z2,
       t4z2,t02z2,t12z2,t22z2,t1z3,t2z3,t3z3,t1z4,t2z4,t1y1z1,
       t1y1z2,t1y2z1,t1y2z2;
  Real tlu1,tlu2,tlu3,tlu4,tlu5,tru1,tru2,tru3,tru4,tru5,
       tlu1z1,tlu1z2,tlu1z3,tlu1z4,
       tlu1z1i2,tlu1z2i2,tlu2z1,tlu2z2,tlu2z3,tlu2z4,tlu2z1i2,
       tlu2z2i2,tlu3z1,tlu3z2,tlu4z1,tlu4z2,tlu1y1z1,tlu1y1z2,
       tlu1y2z1,tlu1y2z2,tlu2y1z1,tlu2y1z2,tlu2y2z1,tlu2y2z2,
       tru1z1,tru1z2,tru1z3,tru1z4,
       tru1z1i2,tru1z2i2,tru2z1,tru2z2,tru2z3,tru2z4,tru2z1i2,
       tru2z2i2,tru3z1,tru3z2,tru4z1,tru4z2,tru1y1z1,tru1y1z2,
       tru1y2z1,tru1y2z2,tru2y1z1,tru2y1z2,tru2y2z1,tru2y2z2,
       tru3y1,tru3y2,tlu3y1,tlu3y2,tlu2y1,tru2y1,tlu2y2,tru2y2,
       tlu0,tru0,tlu1y1,tru1y1,tlu1y2,tru1y2,tlu1y3,tru1y3,
       tlu1y4,tru1y4;
  Real a11,b11,c11,d11,e11,a22,b22,c22,d22,e22,fmll,fmrr,
       fpxll,fpxrr,fpyll,fpyrr,fpzll,fpzrr,fell,ferr;
  Real aetri1u1,aetri1u2,aetri1u3,aetri1u4,aetri1eu1,
       agtri1u1,agtri1u2,agtri1u3,agtri1u4,agtri1eu1;
  Real a2,a3,a4,a5,b2,b3,b4,b5,axu1t,axu2t,ae1t,ae2t;
  Real precision_arg,ekin,prec;
  Real et,fmg,fpg,fpxg,fpzg,feg;
  Real gm1,gm2,ge1,ge2,gpx1,gpx2,gpy1,gpy2,gpz1,gpz2;
  Real x1, yx1, zx1;
  Real x2, yx2, zx2;

  Real   qrt = 0.25, haf = 0.5, one = 1.0;
  Real   mach_precision = 26.0;
  Real   roundoff = 1e-14;

/*
=====================================================================
Calculation of "f_0", the initial true distribution function evaluated
at the wall, from the first order Taylor series around two half
Maxwellians "g_l" and "g_r" constructed on each side of the wall
=====================================================================
*/

  rade1 = one/ade1;
  rade2 = one/ade2;

  axu1t = axu1;
  axu2t = axu2;
  ae1t  = ae1;
  ae2t  = ae2;

    
//    Thus we don't have to redo the limiters
//    This assumes a uniform grid.
  //pcoord will give us dx1/dx2/dx3
    
  //Real dxx = pmy_block->pcoord->dx1v(i);

#ifdef FLUXES_RECON
  aw11  = 2.0*(ade1-ad[0])/dxx;
  bxw11 = 2.0*(axm1-ax[0])/dxx;
  byw11 = 2.0*(aym1-ay[0])/dxx;
  bzw11 = 2.0*(azm1-az[0])/dxx;
  cw11  = 2.0*(aen1-ae[0])/dxx;

  aw22  = 2.0*(ad[1]-ade2)/dxx;
  bxw22 = 2.0*(ax[1]-axm2)/dxx;
  byw22 = 2.0*(ay[1]-aym2)/dxx;
  bzw22 = 2.0*(az[1]-azm2)/dxx;
  cw22  = 2.0*(ae[1]-aen2)/dxx;

    
#ifdef DEBUG_ALL
  fprintf(stdout,"    fluxesrec: i=%4i x1f=%13.5e aw11=%13.5e bxw11=%13.5e byw11=%13.5e bzw11=%13.5e cw11=%13.5e DXX=%13.5e\n",i,x1f,aw11,bxw11,byw11,bzw11,cw11,dxx);
  fprintf(stdout,"    fluxesrec: i=%4i x1f=%13.5e aw22=%13.5e bxw22=%13.5e byw22=%13.5e bzw22=%13.5e cw22=%13.5e DXX=%13.5e\n" ,i,x1f,aw22,bxw22,byw22,bzw22,cw22,dxx);
#endif

#else
  aw11  = 0.0;
  bxw11 = 0.0;
  byw11 = 0.0;
  bzw11 = 0.0;
  cw11  = 0.0;

  aw22  = 0.0;
  bxw22 = 0.0;
  byw22 = 0.0;
  bzw22 = 0.0;
  cw22  = 0.0;
#endif

/*
=====================================================================
Computation of the slopes (al1,al2,al3,al4,al5) and 
(ar1,ar2,ar3,ar4,ar5) of "f_0" (Pg. 66-69 in Kun Xu for 1D case). 

(al1,al2,al3,al4,al5) = (x1,y1,yx1,zx1,z1)
(ar1,ar2,ar3,ar4,ar5) = (x2,y2,yx2,zx2,z2)

dxe3d gives the algebraic equation for the inverted matrix M_alpha_beta
=====================================================================
*/

// WARNING: y1, y2, z1, z2 are being used as moments further down. Possible conflict.

  dxe3d(axu1,ayu1,azu1,ae1,aw11,bxw11,byw11,bzw11,cw11,&x1,&y1,&yx1,&zx1,&z1,ck,ddim);
  dxe3d(axu2,ayu2,azu2,ae2,aw22,bxw22,byw22,bzw22,cw22,&x2,&y2,&yx2,&zx2,&z2,ck,ddim);

  rae1 = one/ae1;
  rae2 = one/ae2;
#if DEBUG_ALL
  fprintf(stdout,"rae1=%13.5e, rae2=%13.5e\n",rae1,rae2);
#endif
  sae1 = std::sqrt(ae1);
  sae2 = std::sqrt(ae2);


/*
Compute the half-moments of the left side of velocity space.

For numerical precision reasons we can't call  ERFC or EXP in 
the hyper-machian flow case, otherwise the result is zero and 
we might divide by it later.  This could cause overflows so we 
have to limit it to the highest possible mach number. In any case, 
it doesn't matter for the final flux whether ERFC or EXP is 1e-20
or 0.
*/


  precision_arg = std::min(-axu1*std::sqrt(ae1),mach_precision);
  //precision_arg = -axu1*sqrt(ae1);
  te1 = 0.5*erfc(precision_arg);
  te2 = 0.5*std::exp(-SQR(precision_arg))/std::sqrt(ae1*pi);

#if DEBUG_ALL
  fprintf(stdout,"precision_arg=%13.5e\n",precision_arg);
#endif

  teu[1] = axu1*te1+te2;
  teu[2] = axu1*teu[1]+0.5*te1*rae1;
  teu[3] = axu1*teu[2]+1.0*teu[1]*rae1;
  teu[4] = axu1*teu[3]+1.5*teu[2]*rae1;
  teu[5] = axu1*teu[4]+2.0*teu[3]*rae1;
  teu[6] = axu1*teu[5]+2.5*teu[4]*rae1;


  precision_arg = std::min(axu2*std::sqrt(ae2),mach_precision);
  //precision_arg = axu2*sqrt(ae2);
  tg1 = 0.5*erfc(precision_arg);
  tg2 = -0.5*std::exp(-SQR(precision_arg))/std::sqrt(ae2*pi);


  tgu[1] = axu2*tg1+tg2;
  tgu[2] = axu2*tgu[1]+0.5*tg1*rae2;
  tgu[3] = axu2*tgu[2]+1.0*tgu[1]*rae2;
  tgu[4] = axu2*tgu[3]+1.5*tgu[2]*rae2;
  tgu[5] = axu2*tgu[4]+2.0*tgu[3]*rae2;
  tgu[6] = axu2*tgu[5]+2.5*tgu[4]*rae2;

#ifdef DEBUG_ALL
  fprintf(stdout,"axu2=%11.3e,tgu[5]=%11.3e,tgu[4]=%11.3e,tgu[6]=%11.3e,rae2=%11.3e\n",axu2,tgu[5],tgu[4],tgu[6],rae2);
#endif

// Extra moments for the Navier Stokes version

#ifdef UNAVSTOKES

  tlu0=1.0;
  tlu1=axu1;
  tlu2=axu1*tlu1+0.5*tlu0*rae1;
  tlu3=axu1*tlu2+1.0*tlu1*rae1;
  tlu4=axu1*tlu3+1.5*tlu2*rae1;
  tlu5=axu1*tlu4+2.0*tlu3*rae1;

  tru0=1.0;
  tru1=axu2;
  tru2=axu2*tru1+0.5*tru0*rae2;
  tru3=axu2*tru2+1.0*tru1*rae2;
  tru4=axu2*tru3+1.5*tru2*rae2;
  tru5=axu2*tru4+2.0*tru3*rae2;
// End of moments for the Navier Stokes version

#endif // UNAVSTOKES


// Full moments of the internal degrees of freedom

  ei2 = 0.5*ck*rae1;
  ei4 = 0.25*ck*(ck+2.0)*rae1*rae1;
  gi2 = 0.5*ck*rae2;
  gi4 = 0.25*ck*(ck+2.0)*rae2*rae2;

// Full moments of the inactive velocity in the y-direction.
// The last number indicates the order of the moment.

  ey0 = 1.0;    
  ey1 = ayu1;
  ey2 = ayu1*ey1+0.5*ey0*rae1;
  ey3 = ayu1*ey2+1.0*ey1*rae1;
  ey4 = ayu1*ey3+1.5*ey2*rae1;
  ey5 = ayu1*ey4+2.0*ey3*rae1;

  gy0 = 1.0;
  gy1 = ayu2;
  gy2 = ayu2*gy1+0.5*gy0*rae2;
  gy3 = ayu2*gy2+1.0*gy1*rae2;
  gy4 = ayu2*gy3+1.5*gy2*rae2;
  gy5 = ayu2*gy4+2.0*gy3*rae2;

// Full moments of the inactive velocity in the z-direction.
// The last number indicates the order of the moment.

  ez0 = 1.0;
  ez1 = azu1;
  ez2 = azu1*ez1+0.5*ez0*rae1;
  ez3 = azu1*ez2+1.0*ez1*rae1;
  ez4 = azu1*ez3+1.5*ez2*rae1;
  ez5 = azu1*ez4+2.0*ez3*rae1;
  
  gz0 = 1.0;
  gz1 = azu2;
  gz2 = azu2*gz1+0.5*gz0*rae2;
  gz3 = azu2*gz2+1.0*gz1*rae2;
  gz4 = azu2*gz3+1.5*gz2*rae2;
  gz5 = azu2*gz4+2.0*gz3*rae2;

// Combined moments of the velocity half-space on the left, 
// the full moments of the inactive velocity in the y-direction,
// and the internal degrees of freedom.

  te1y1   = te1*ey1;
  te1y2   = te1*ey2;
  te1y3   = te1*ey3;
  te1y4   = te1*ey4;
  te1y5   = te1*ey5;
  te1i2   = te1*ei2;
  te1i4   = te1*ei4;
  te1y1i2 = te1y1*ei2;
  te1y2i2 = te1y2*ei2;
  te1y1i4 = te1y1*ei4;
  te1y3i2 = te1y3*ei2;

  teu1y1   = teu[1]*ey1;
  teu1y2   = teu[1]*ey2;
  teu1y3   = teu[1]*ey3;
  teu1y4   = teu[1]*ey4;
  teu1i2   = teu[1]*ei2;
  teu1i4   = teu[1]*ei4;
  teu1y1i2 = teu1y1*ei2;
  teu1y2i2 = teu1y2*ei2;
  teu1y1i4 = teu1y1*ei4;
  teu1y3i2 = teu1y3*ei2;

  teu2y1   = teu[2]*ey1;
  teu2y2   = teu[2]*ey2;
  teu2y3   = teu[2]*ey3;
  teu2y4   = teu[2]*ey4;
  teu2i2   = teu[2]*ei2;
  teu2i4   = teu[2]*ei4;
  teu2y1i2 = teu2y1*ei2;
  teu2y2i2 = teu2y2*ei2;
      
  teu3y1   = teu[3]*ey1;
  teu3y2   = teu[3]*ey2;
  teu3y3   = teu[3]*ey3;
  teu3i2   = teu[3]*ei2;
  teu3y1i2 = teu3y1*ei2;
 
  teu4y1   = teu[4]*ey1;
  teu4y2   = teu[4]*ey2;
  teu4i2   = teu[4]*ei2;
  
  teu5y1 = teu[5]*ey1;

// Combined moments of the velocity half-space on the right, 
// the full moments of the inactive velocity in the y-direction,
// and the internal degrees of freedom.

  tg1y1   = tg1*gy1;
  tg1y2   = tg1*gy2;
  tg1y3   = tg1*gy3;
  tg1y4   = tg1*gy4;
  tg1y5   = tg1*gy5;
  tg1i2   = tg1*gi2;
  tg1i4   = tg1*gi4;
  tg1y1i2 = tg1y1*gi2;
  tg1y2i2 = tg1y2*gi2;
  tg1y1i4 = tg1y1*gi4;
  tg1y3i2 = tg1y3*gi2;
  
  tgu1y1   = tgu[1]*gy1;
  tgu1y2   = tgu[1]*gy2;
  tgu1y3   = tgu[1]*gy3;
  tgu1y4   = tgu[1]*gy4;
  tgu1i2   = tgu[1]*gi2;
  tgu1i4   = tgu[1]*gi4;
  tgu1y1i2 = tgu1y1*gi2;
  tgu1y2i2 = tgu1y2*gi2;
  tgu1y1i4 = tgu1y1*gi4;
  tgu1y3i2 = tgu1y3*gi2;
  
  tgu2y1   = tgu[2]*gy1;
  tgu2y2   = tgu[2]*gy2;
  tgu2y3   = tgu[2]*gy3;
  tgu2y4   = tgu[2]*gy4;
  tgu2i2   = tgu[2]*gi2;
  tgu2i4   = tgu[2]*gi4;
  tgu2y1i2 = tgu2y1*gi2;
  tgu2y2i2 = tgu2y2*gi2;
  
  tgu3y1   = tgu[3]*gy1;
  tgu3y2   = tgu[3]*gy2;
  tgu3i2   = tgu[3]*gi2;
  tgu3y1i2 = tgu3y1*gi2;
  
  tgu4y1   = tgu[4]*gy1;
  tgu4y2   = tgu[4]*gy2;
  tgu4i2   = tgu[4]*gi2;
  
  tgu5y1   = tgu[5]*gy1;
  
// Combined moments of the velocity half-space on the left, 
// the full moments of the inactive velocity in the z-direction,
// and the internal degrees of freedom.

  te1z1    = te1*ez1;
  te1z2    = te1*ez2;
  
  teu1z1   = teu[1]*ez1;
  teu1z2   = teu[1]*ez2;
  teu1z3   = teu[1]*ez3;
  teu1z4   = teu[1]*ez4;
  teu1z1i2 = teu1z1*ei2;
  teu1z2i2 = teu1z2*ei2;

  teu2z1   = teu[2]*ez1;
  teu2z2   = teu[2]*ez2;
  teu2z3   = teu[2]*ez3;
  teu2z4   = teu[2]*ez4;
  teu2z1i2 = teu2z1*ei2;
  teu2z2i2 = teu2z2*ei2;

  teu3z1=teu[3]*ez1;
  teu3z2=teu[3]*ez2;
  
  teu4z1=teu[4]*ez1;
  teu4z2=teu[4]*ez2;
  
// Combined moments of the velocity half-space on the right, 
// the full moments of the inactive velocity in the z-direction,
// and the internal degrees of freedom.

  tg1z1 = tg1*gz1;
  tg1z2 = tg1*gz2;

  tgu1z1=tgu[1]*gz1;
  tgu1z2=tgu[1]*gz2;
  tgu1z3=tgu[1]*gz3;
  tgu1z4=tgu[1]*gz4;
  tgu1z1i2=tgu1z1*gi2;
  tgu1z2i2=tgu1z2*gi2;
  
  tgu2z1=tgu[2]*gz1;
  tgu2z2=tgu[2]*gz2;
  tgu2z3=tgu[2]*gz3;
  tgu2z4=tgu[2]*gz4;
  tgu2z1i2=tgu2z1*gi2;
  tgu2z2i2=tgu2z2*gi2;
  
  tgu3z1=tgu[3]*gz1;
  tgu3z2=tgu[3]*gz2;
  
  tgu4z1=tgu[4]*gz1;
  tgu4z2=tgu[4]*gz2;
  
// Combined moments of the velocity half-space on the left, 
// the full moments of the inactive velocity in the y-direction,
// and the full moments of the inactive velocity in the z-direction.

  teu1y1z1=teu[1]*ey1*ez1;
  teu1y1z2=teu[1]*ey1*ez2;
  teu1y2z1=teu[1]*ey2*ez1;
  teu1y2z2=teu[1]*ey2*ez2;
  
  teu2y1z1=teu[2]*ey1*ez1;
  teu2y1z2=teu[2]*ey1*ez2;
  teu2y2z1=teu[2]*ey2*ez1;
  teu2y2z2=teu[2]*ey2*ez2;
  
// Combined moments of the velocity half-space on the right, 
// the full moments of the inactive velocity in the y-direction,
// and the full moments of the inactive velocity in the z-direction.

  tgu1y1z1=tgu[1]*gy1*gz1;
  tgu1y1z2=tgu[1]*gy1*gz2;
  tgu1y2z1=tgu[1]*gy2*gz1;
  tgu1y2z2=tgu[1]*gy2*gz2;
  
  tgu2y1z1=tgu[2]*gy1*gz1;
  tgu2y1z2=tgu[2]*gy1*gz2;
  tgu2y2z1=tgu[2]*gy2*gz1;
  tgu2y2z2=tgu[2]*gy2*gz2;


// Moments needed for the N.S. version


#ifdef UNAVSTOKES

  tlu1y1   = tlu1*ey1;
  tlu1y2   = tlu1*ey2;
  tlu1y3   = tlu1*ey3;
  tlu1y4   = tlu1*ey4;
  
  tru1y1   = tru1*gy1;
  tru1y2   = tru1*gy2;
  tru1y3   = tru1*gy3;
  tru1y4   = tru1*gy4;
  
  tlu2y1   = tlu2*ey1;
  tlu2y2   = tlu2*ey2;
  
  tru2y1   = tru2*gy1;
  tru2y2   = tru2*gy2;

  tlu3y1   = tlu3*ey1;
  tlu3y2   = tlu3*ey2;

  tru3y1   = tru3*gy1;
  tru3y2   = tru3*gy2;

  tlu1z1=tlu1*ez1;
  tlu1z2=tlu1*ez2;
  tlu1z3=tlu1*ez3;
  tlu1z4=tlu1*ez4;
  tlu1z1i2=tlu1z1*ei2;
  tlu1z2i2=tlu1z2*ei2;

  tru1z1=tru1*gz1;
  tru1z2=tru1*gz2;
  tru1z3=tru1*gz3;
  tru1z4=tru1*gz4;
  tru1z1i2=tru1z1*gi2;
  tru1z2i2=tru1z2*gi2;

  tlu2z1=tlu2*ez1;
  tlu2z2=tlu2*ez2;
  tlu2z3=tlu2*ez3;
  tlu2z4=tlu2*ez4;
  tlu2z1i2=tlu2z1*ei2;
  tlu2z2i2=tlu2z2*ei2;
  tru2z1=tru2*gz1;
  tru2z2=tru2*gz2;
  tru2z3=tru2*gz3;
  tru2z4=tru2*gz4;
  tru2z1i2=tru2z1*gi2;
  tru2z2i2=tru2z2*gi2;

  tlu3z1=tlu3*ez1;
  tlu3z2=tlu3*ez2;
  tru3z1=tru3*gz1;
  tru3z2=tru3*gz2;
  
  tlu4z1=tlu4*ez1;
  tlu4z2=tlu4*ez2;
  tru4z1=tru4*gz1;
  tru4z2=tru4*gz2;

  tlu1y1z1=tlu1*ey1*ez1;
  tlu1y1z2=tlu1*ey1*ez2;
  tlu1y2z1=tlu1*ey2*ez1;
  tlu1y2z2=tlu1*ey2*ez2;
  tru1y1z1=tru1*gy1*gz1;
  tru1y1z2=tru1*gy1*gz2;
  tru1y2z1=tru1*gy2*gz1;
  tru1y2z2=tru1*gy2*gz2;
  
  tlu2y1z1=tlu2*ey1*ez1;
  tlu2y1z2=tlu2*ey1*ez2;
  tlu2y2z1=tlu2*ey2*ez1;
  tlu2y2z2=tlu2*ey2*ez2;
  tru2y1z1=tru2*gy1*gz1;
  tru2y1z2=tru2*gy1*gz2;
  tru2y2z1=tru2*gy2*gz1;
  tru2y2z2=tru2*gy2*gz2;
  
// End of the extra moments for the Navier Stokes version

#endif // UNAVSTOKES


/*
=====================================================================
 
 This comes from eqn. (4.19) in Kun Xu pg. 69.  It simply says that 
 "g_0" at the wall at t=0 can be derived from "f_0" at the wall by
 definition.

=====================================================================
*/

#ifdef DEBUG_ALL
  fprintf(stdout,"ade1=%11.3e, te1=%11.3e, ade2=%11.3e, tg1=%11.3e, teu[1]=%11.3e, tgu[1]=%11.3e, te1y1=%11.3e, tg1y1=%11.3e, te1z1=%11.3e, tg1z1=%11.3e, teu[2]=%11.3e, te1y2=%11.3e, te1z2=%11.3e, te1i2=%11.3e, tgu[2]=%11.3e, tg1y2=%11.3e, tg1z2=%11.3e, tg1i2=%11.3e\n",ade1, te1, ade2, tg1, teu[1], tgu[1], te1y1, tg1y1, te1z1, tg1z1, teu[2], te1y2, te1z2, te1i2, tgu[2], tg1y2, tg1z2, tg1i2);
#endif

// conservative variables
  ade  = ade1*te1+ade2*tg1;
  axm  = ade1*teu[1]+ade2*tgu[1];
  aym  = ade1*te1y1+ade2*tg1y1;
  azm  = ade1*te1z1+ade2*tg1z1;
  aen  = 0.5*(ade1*(teu[2]+te1y2+te1z2+te1i2)+ade2*(tgu[2]+tg1y2+tg1z2+tg1i2));
  rade = 1.0/ade;

// primitive variables
  ekin = 0.5*(SQR(axm)+SQR(aym)+SQR(azm))*rade;
  if (aen < 0.0) perror("[bgk2]: aen < 0.\n");

  prec = fabs(aen-ekin)/aen;
  if (prec < roundoff) aen = (1.0+roundoff)*ekin;

#ifdef DEBUG_ALL
#ifdef FLUXES_G0
  fprintf(stdout,"    fluxesg0: ade=%13.5e,Gamma_1=%13.5e,aen=%13.5e,ekin=%13.5e\n",ade,Gamma_1,aen,ekin);
#endif
#endif
//  ae0 = 0.5*ade/(Gamma_1*(aen-ekin));
  ae0=0.25*((ck+3.0)*ade)/(Gamma_1*(aen-ekin));

  if ((ae0 < 0.0) || (ade < 0.0)) perror("[bgk2]: ae0 or ade < 0\n");

  axu0 = axm*rade;
  ayu0 = aym*rade;
  azu0 = azm*rade;
  rae0 = 1.0/ae0;

/*
=====================================================================

From the conserved quantities at the cell wall you can make the 
equilibrium Maxwellian "g_0".  We already have the velocity moments of
"f_0".  Now we compute the time integral of the second term of the
distribution function at the wall, "f".  This term is the "f_0" term 
in eqn. (4.34) of Kun Xu pg. 77.

 We now have g_0 and the velocity moments of f_0 at the wall.     C
 We compute here the integral over time of the moments of the     C
 second term entering the true distribution function at the wall, C
 f. This term is the f_0 term in eq (4-34) of Kun p77.            C

=====================================================================
*/
  aa1 = ade1*rae1;
  aa2 = ade2*rae2;
/*
Here we determine the collision time "tau" (called "te" here).
This comes from v.K. eq. 2.15 and remark 4.10/eq. 4.40. C_1 gives the number of mean free
paths per cell dxx (equivalently number of collision times per sound crossing time tau_s = dxx/c_s).
C_2 is an artificial viscosity coefficient, applied for discontinuities of the reconstruction.
Discontinuity will be spread out over C_2 cells. For e.g. p1=4p2, we get a second term of
c_2*dt*3/5. So for c_2=1, we reach dt/tau = 3/5, means "free transport".
Note that the 'min' clause means that at least you'll get 1 collision per timestep. More collisions
per timestep mean less diffusivity.
The last choice (ieuler .eq. 4) tries to weight the contributions from pressure and temperature
discontinuities.  
*/

//if(ieuler == 0) // Navier Stokes problem
//{
//    te = c_1*sqrt(ae0)/ade*dxx + tau*MIN(1,c_2*(fabs(aa1-aa2)/(aa1+aa2)));
//}else if(ieuler == 1) // Euler problem 
//{
 // Real BGK_c1 = pin->GetReal("hydro","BGK_c1");
 // Real BGK_c2 = pin->GetReal("hydro","BGK_c2");

  te = bgkc1*tau + tau*bgkc2*(fabs(aa1-aa2)/(aa1+aa2));

//}else if(ieuler == 2) // Navier Stokes with diffusivity nu=c_1
//{
//    te = 2.0*c_1*ae0 + c_2*tau*(fabs(aa1-aa2)/(aa1+aa2));
//}else if(ieuler == 3) // Kun's suggestion
//{
//te = bgkc1*sqrt(ae0)/ade*dxx + bgkc2*tau*(fabs(aa1-aa2)/(aa1+aa2));
//    te = bgkc1*tau*sqrt(ae0)/ade + bgkc2*tau*(fabs(aa1-aa2)/(aa1+aa2));
#if DEBUG_ALL
    fprintf(stdout,"te=%13.5e, tau=%13.5e, sqrt(ae0)=%13.5e, ade=%13.5e, aa1-aa2=%13.5e, aa1+aa2=%13.5e\n",te,tau,sqrt(ae0),ade,aa1-aa2,aa1+aa2);
#endif

//Force quit code at 1st order if criterion not met
    if(te>(10.0*tau)){
#if DEBUG_ALL
      fprintf(stdout,"***only 1st order fluxes used\n");
#endif
//TEST: no g0 or Abar fluxes yet      

      fm1 = ade1*teu[1]*tau;
      fp1 = ade1*teu[2]*tau;
      fpx1= ade1*teu1y1*tau;
      fpz1= ade1*teu1z1*tau;
      fe1 = ade1*0.5*(teu[3]+teu1i2+teu1y2+teu1z2)*tau;

      fm2 = ade2*tgu[1]*tau;
      fp2 = ade2*tgu[2]*tau;
      fpx2= ade2*tgu1y1*tau;
      fpz2= ade2*tgu1z1*tau;
      fe2 = ade2*0.5*(tgu[3]+tgu1i2+tgu1y2+tgu1z2)*tau;

      afm= fm1 + fm2;
      afp= fp1 + fp2;
      afpx= fpx1 + fpx2;
      afpz= fpz1 + fpz2;
      afe= fe1 + fe2;

      flx(IDN,k,j,i)  = afm /tau;
      flx(im1,k,j,i) = afp /tau;
      flx(im2,k,j,i) = afpx/tau;
      flx(im3,k,j,i) = afpz/tau;
      flx(IEN,k,j,i)  = afe /tau;    

#ifdef DEBUG_ALL
      fprintf(stdout,"idnflux=%13.5e, im1flux=%13.5e, im2flux=%13.5e, im3flux=%13.5e, ienflux=%13.5e\n",flx(IDN,k,j,i),flx(im1,k,j,i),flx(im2,k,j,i),flx(im3,k,j,i),flx(IEN,k,j,i));
#endif 

    }else{

    te = std::max(0.01*tau,std::min(1.0*te,10.0*tau));
 
#ifdef DEBUG_ALL
    fprintf(stdout,"te=%13.5e\n",te); 
#endif
//}else if(ieuler == 4) // Attempt to include heat conduction
//{
//    pfrac = fabs(aa1-aa2)/(aa1+aa2);
//    tfrac = fabs(rae1-rae2)/(rae1+rae2);
//    te = c_1*sqrt(ae0)/ade*dxx + c_2*tau*pfrac*fabs(1.0-tfrac) + kappa*tau*tfrac*fabs(1.0-pfrac);
//}

// Integrals over a timestep of exp(-t/tau) and t*exp(-t/tau)

  st   = -tau/te;
  sw   = std::exp(st);
  se1  = te*(1.0-sw);
  set0 = te*(-tau+se1);
  set1 = -te*tau*sw+te*se1;
  set2 = te*(-tau+se1)+set1;

// Integrals over time of the moments of the second term of eqn. (4.34)
// (without the "b_l" or "b_r"). FMI1, FPI1, etc are the terms with the 
// gamma_4 coefficient in (4.37). AMI, API, APIX, and AEI will be used to
// compute the A#_bar coefficients, which are needed to compute "g".

  fmi1=ade1*se1*te1-set1* 
      (x1*teu[1]+y1*teu[2]+yx1*teu1y1+zx1*teu1z1+ 
      z1*(teu[3]+teu1y2+teu1z2+teu1i2));
  fpi1=ade1*se1*teu[1]-set1* 
      (x1*teu[2]+y1*teu[3]+yx1*teu2y1+zx1*teu2z1+ 
      z1*(teu[4]+teu2y2+teu2z2+teu2i2));
  fpix1=ade1*se1*te1y1-set1* 
      (x1*teu1y1+y1*teu2y1 
      +yx1*teu1y2+zx1*teu1y1z1+ 
      z1*(teu3y1+teu1y3+teu1y1z2+teu1y1i2));
  fpiz1=ade1*se1*te1z1-set1* 
      (x1*teu1z1+y1*teu2z1+yx1*teu1y1z1+zx1*teu1z2+ 
      z1*(teu3z1+teu1y2z1+teu1z3+teu1z1i2));
  fei1=ade1*0.5*se1*(teu[2]+te1y2+te1z2+te1i2) 
       -0.5*set1*       
       (x1*(teu[3]+teu1y2+teu1z2+teu1i2)+ 
       y1*(teu[4]+teu2y2+teu2z2+teu2i2)+ 
       yx1*(teu3y1+teu1y3+teu1y1z2+teu1y1i2)+ 
       zx1*(teu3z1+teu1y2z1+teu1z3+teu1z1i2)+      
       z1*(teu[5]+teu1y4+teu1z4+teu1i4 
       +2.0*teu3y2+2.0*teu3z2+2.0*teu1y2z2+ 
       2.0*teu1z2i2+2.0*teu3i2+2.0*teu1y2i2));
  
  fmi2=ade2*se1*tg1-set1* 
       (x2*tgu[1]+y2*tgu[2]+yx2*tgu1y1+zx2*tgu1z1+ 
       z2*(tgu[3]+tgu1y2+tgu1z2+tgu1i2));
  fpi2=ade2*se1*tgu[1]-set1* 
       (x2*tgu[2]+y2*tgu[3]+yx2*tgu2y1+zx2*tgu2z1+ 
       z2*(tgu[4]+tgu2y2+tgu2z2+tgu2i2));
  fpix2=ade2*se1*tg1y1-set1* 
       (x2*tgu1y1+y2*tgu2y1 
       +yx2*tgu1y2+zx2*tgu1y1z1+ 
       z2*(tgu3y1+tgu1y3+tgu1y1z2+tgu1y1i2));
  fpiz2=ade2*se1*tg1z1-set1* 
       (x2*tgu1z1+y2*tgu2z1+yx2*tgu1y1z1+zx2*tgu1z2+ 
       z2*(tgu3z1+tgu1y2z1+tgu1z3+tgu1z1i2));
  fei2=ade2*0.5*se1*(tgu[2]+tg1y2+tg1z2+tg1i2) 
       -0.5*set1* 
       (x2*(tgu[3]+tgu1y2+tgu1z2+tgu1i2)+ 
       y2*(tgu[4]+tgu2y2+tgu2z2+tgu2i2)+ 
       yx2*(tgu3y1+tgu1y3+tgu1y1z2+tgu1y1i2)+ 
       zx2*(tgu3z1+tgu1y2z1+tgu1z3+tgu1z1i2)+ 
       z2*(tgu[5]+tgu1y4+tgu1z4+tgu1i4 
       +2.0*tgu3y2+2.0*tgu3z2+2.0*tgu1y2z2+ 
       2.0*tgu1z2i2 
       +2.0*tgu3i2+2.0*tgu1y2i2));
  
#ifdef UGRAV2ND
//  Gravitational potential stuff, which we don't need anymore
  fmg  = set1*2.0*gra1*ade1*ae1*(teu[1]-axu1*te1) 
       + set1*2.0*gra2*ade2*ae2*(tgu[1]-axu2*tg1);
  fpg  = set1*2.0*gra1*ade1*ae1*(teu[2]-axu1*teu[1]) 
       + set1*2.0*gra2*ade2*ae2*(tgu[2]-axu2*tgu[1]);
  fpxg = set1*2.0*gra1*ade1*ae1*(teu1y1-axu1*te1y1) 
       + set1*2.0*gra2*ade2*ae2*(tgu1y1-axu2*tg1y1);
  fpzg = set1*2.0*gra1*ade1*ae1*(teu1z1-axu1*te1z1) 
       + set1*2.0*gra2*ade2*ae2*(tgu1z1-axu2*tg1z1);
  feg  = set1*gra1*ade1*ae1*(teu[3]+teu1y2+teu1z2+teu1i2 
       - axu1*teu[2]-axu1*te1y2-axu1*te1z2-axu1*te1i2) 
       + set1*gra2*ade2*ae2*(tgu[3]+tgu1y2+tgu1z2+tgu1i2 
       - axu2*tgu[2]-axu2*tg1y2-axu2*tg1z2-axu2*tg1i2);    
#endif

  ami  = fmi1+fmi2;//+fmg;
  api  = fpi1+fpi2;//+fpg;
  apix = fpix1+fpix2;//+fpxg;
  apiz = fpiz1+fpiz2;//+fpzg;
  aei  = fei1+fei2;//+feg;

// Fluxes from "f_0" (the non-equilibrium state), but note that 
// now f_0 = g_l*(1+ a_l*x - te*(a_l*u + A_l)), but the "A_l" are
// not included yet because they haven't been computed yet.

//Changed so that specific contributions can be shut off.
  fm1 =   ade1*se1*teu[1]  
#ifdef FLUXES_RECON
        - (set1+se1*te)*(x1*teu[2]+y1*teu[3]+yx1*teu2y1+zx1*teu2z1+z1*(teu[4]+teu2y2+teu2z2+teu2i2)) 
#endif
#ifdef UGRAV2ND
        + set1*2.0*gra1*ade1*ae1*(teu[2]-axu1*teu[1])
#endif
        ;
#ifdef DEBUG_ALL
  fprintf(stdout,"ade1=%11.3e,se1=%11.3e,teu[1]=%11.3e,set1=%11.3e,te=%11.3e,teu[2]=%11.3e,y1=%11.3e,teu[3]=%11.3e,yx1=%11.3e,teu2y1=%11.3e,zx1=%11.3e,teu2z1=%11.3e,z1=%11.3e,teu[4]=%11.3e,teu2y2=%11.3e,teu2z2=%11.3e,teu2i2=%11.3e\n",ade1,se1,teu[1],set1,te,teu[2],y1,teu[3],yx1,teu2y1,zx1,teu2z1,z1,teu[4],teu2y2,teu2z2,teu2i2);
  fprintf(stdout,"ade2=%11.3e,tgu[1]=%11.3e,tgu[2]=%11.3e,tgu[3]=%11.3e,tgu2y1=%11.3e,tgu2z1=%11.3e,tgu[4]=%11.3e,tgu2y2=%11.3e,tgu2z2=%11.3e,tgu2i2=%11.3e,tgu1i2=%11.3e,tgu1y2=%11.3e,tgu1z2=%11.3e,x2=%11.3e,y2=%11.3e,yx2=%11.3e,zx2=%11.3e,z2=%11.3e\n",ade2,tgu[1],tgu[2],tgu[3],tgu2y1,tgu2z1,tgu[4],tgu2y2,tgu2z2,tgu2i2,tgu1i2,tgu1y2,tgu1z2,x2,y2,yx2,zx2,z2);
  fprintf(stdout,"tgu[4]=%11.3e,tgu[5]=%11.3e,tgu[6]=%11.3e,tgu3y2=%11.3e,tgu3z2=%11.3e,tgu3i2=%11.3e,tgu2y4=%11.3e,tgu2z4=%11.3e,tgu2i4=%11.3e,tgu4y2=%11.3e,tgu4z2=%11.3e,tgu2y2z2=%11.3e,tgu2z2i2=%11.3e,tgu4i2=%11.3e,tgu2y2i2=%11.3e\n",tgu[4],tgu[5],tgu[6],tgu3y2,tgu3z2,tgu3i2,tgu2y4,tgu2z4,tgu2i4,tgu4y2,tgu4z2,tgu2y2z2,tgu2z2i2,tgu4i2,tgu2y2i2);
#endif


  fp1 =   ade1*se1*teu[2]  
#ifdef FLUXES_RECON
        - (set1+se1*te)*(x1*teu[3]+y1*teu[4]+yx1*teu3y1+zx1*teu3z1+z1*(teu[5]+teu3y2+teu3z2+teu3i2)) 
#endif
#ifdef UGRAV2ND
        + set1*2.0*gra1*ade1*ae1*(teu[3]-axu1*teu[2])
#endif
        ;
  fpx1=   ade1*se1*teu1y1  
#ifdef FLUXES_RECON
        - (set1+se1*te)*(x1*teu2y1+y1*teu3y1+yx1*teu2y2+zx1*teu2y1z1+z1*(teu4y1+teu2y3+teu2y1z2+teu2y1i2)) 
#endif
#ifdef UGRAV2ND
        + set1*2.0*gra1*ade1*ae1*(teu2y1-axu1*teu1y1)
#endif
        ;
  fpz1=   ade1*se1*teu1z1  
#ifdef FLUXES_RECON
        - (set1+se1*te)*(x1*teu2z1+y1*teu3z1+yx1*teu2y1z1+zx1*teu2z2+z1*(teu4z1+teu2z3+teu2y2z1+teu2z1i2)) 
#endif
#ifdef UGRAV2ND
        + set1*2.0*gra1*ade1*ae1*(teu2z1-axu1*teu1z1)
#endif
        ;
  fe1 =   ade1*0.5*se1*(teu[3]+teu1i2+teu1y2+teu1z2)  
#ifdef FLUXES_RECON
        - 0.5*(set1+se1*te)*(  x1*(teu[4]+teu2y2+teu2z2+teu2i2) 
                             + y1*(teu[5]+teu3y2+teu3z2+teu3i2) 
                             +yx1*(teu4y1+teu2y3+teu2y1z2+teu2y1i2) 
                             +zx1*(teu4z1+teu2y2z1+teu2z3+teu2z1i2) 
                             + z1*(teu[6]+teu2y4+teu2z4+teu2i4+2.0*teu4y2 
                                   +2.0*teu4z2+2.0*teu2y2z2+2.0*teu2z2i2+2.0*teu4i2+2.0*teu2y2i2)) 
#endif
#ifdef UGRAV2ND
        + set1*gra1*ade1*ae1*(teu[4]+teu2y2+teu2z2+teu2i2-axu1*teu[3]-axu1*teu1y2-axu1*teu1z2-axu1*teu1i2)
#endif
        ;
 

  fm2 =   ade2*se1*tgu[1] 
#ifdef FLUXES_RECON
        - (set1+se1*te)*(x2*tgu[2]+y2*tgu[3]+yx2*tgu2y1+zx2*tgu2z1+z2*(tgu[4]+tgu2y2+tgu2z2+tgu2i2)) 
#endif
#ifdef UGRAV2ND
        + set1*2.0*gra2*ade2*ae2*(tgu[2]-axu2*tgu[1])
#endif
        ;
  fp2 =   ade2*se1*tgu[2]  
#ifdef FLUXES_RECON
        - (set1+se1*te)*(x2*tgu[3]+y2*tgu[4]+yx2*tgu3y1+zx2*tgu3z1+z2*(tgu[5]+tgu3y2+tgu3z2+tgu3i2)) 
#endif
#ifdef UGRAV2ND
        + set1*2.0*gra2*ade2*ae2*(tgu[3]-axu2*tgu[2])
#endif
        ;
  fpx2=   ade2*se1*tgu1y1  
#ifdef FLUXES_RECON
        - (set1+se1*te)*(x2*tgu2y1+y2*tgu3y1+yx2*tgu2y2+zx2*tgu2y1z1+z2*(tgu4y1+tgu2y3+tgu2y1z2+tgu2y1i2)) 
#endif
#ifdef UGRAV2ND
        + set1*2.0*gra2*ade2*ae2*(tgu2y1-axu2*tgu1y1)
#endif
        ;
  fpz2=   ade2*se1*tgu1z1 
#ifdef FLUXES_RECON
        - (set1+se1*te)*(x2*tgu2z1+y2*tgu3z1+yx2*tgu2y1z1+zx2*tgu2z2+z2*(tgu4z1+tgu2z3+tgu2y2z1+tgu2z1i2)) 
#endif
#ifdef UGRAV2ND
        + set1*2.0*gra2*ade2*ae2*(tgu2z1-axu2*tgu1z1)
#endif
        ;
  fe2 =   ade2*0.5*se1*(tgu[3]+tgu1i2+tgu1y2+tgu1z2) 
#ifdef FLUXES_RECON
        - 0.5*(set1+se1*te)*(  x2*(tgu[4]+tgu2y2+tgu2z2+tgu2i2) 
                             + y2*(tgu[5]+tgu3y2+tgu3z2+tgu3i2) 
                             +yx2*(tgu4y1+tgu2y3+tgu2y1z2+tgu2y1i2) 
                             +zx2*(tgu4z1+tgu2y2z1+tgu2z3+tgu2z1i2) 
                             + z2*(tgu[6]+tgu2y4+tgu2z4+tgu2i4+2.0*tgu4y2 
                                   +2.0*tgu4z2+2.0*tgu2y2z2+2.0*tgu2z2i2+2.0*tgu4i2+2.0*tgu2y2i2)) 
#endif
#ifdef UGRAV2ND
        + set1*gra2*ade2*ae2*(tgu[4]+tgu2y2+tgu2z2+tgu2i2-axu2*tgu[3]-axu2*tgu1y2-axu2*tgu1z2-axu2*tgu1i2)
#endif
        ;


#ifdef UNAVSTOKES

// Extra stuff for the Navier Stokes version
  a1=-(x1*tlu1+y1*tlu2+yx1*tlu1y1+zx1*tlu1z1+z1* 
      (tlu3+tlu1y2+tlu1z2+tlu1*ei2));

  a2=-(x1*tlu2+y1*tlu3+yx1*tlu2y1+zx1*tlu2z1+z1* 
      (tlu4+tlu2y2+tlu2z2+tlu2*ei2));
  a3=-(x1*tlu1y1+y1*tlu2y1+yx1*tlu1y2+zx1*tlu1y1z1+z1* 
      (tlu3y1+tlu1y3+tlu1y1z2+tlu1y1*ei2));
  a4=-(x1*tlu1z1+y1*tlu2z1+yx1*tlu1y1z1+zx1*tlu1z2+z1* 
      (tlu3z1+tlu1y2z1+tlu1z3+tlu1z1*ei2));
  a5=-0.5*(x1*(tlu3+tlu1y2+tlu1z2+tlu1*ei2)+ 
      y1*(tlu4+tlu2y2+tlu2z2+tlu2*ei2)+ 
       yx1*(tlu3y1+tlu1y3+tlu1y1z2+tlu1y1*ei2)+ 
       zx1*(tlu3z1+tlu1y2z1+tlu1z3+tlu1z1*ei2)+ 
       z1*(tlu5+tlu1y4+tlu1z4+tlu1*ei4+2.0*(tlu3y2+tlu3z2+ 
       tlu1y2z2+tlu1z2*ei2+tlu3*ei2+tlu1y2*ei2)));
    
  b1=-(x2*tru1+y2*tru2+yx2*tru1y1+zx2*tru1z1+z2* 
      (tru3+tru1y2+tru1z2+tru1*gi2));
  b2=-(x2*tru2+y2*tru3+yx2*tru2y1+zx2*tru2z1+z2* 
      (tru4+tru2y2+tru2z2+tru2*gi2));
  b3=-(x2*tru1y1+y2*tru2y1+yx2*tru1y2+zx2*tru1y1z1+z2* 
      (tru3y1+tru1y3+tru1y1z2+tru1y1*gi2));
  b4=-(x2*tru1z1+y2*tru2z1+yx2*tru1y1z1+zx2*tru1z2+z2* 
      (tru3z1+tru1y2z1+tru1z3+tru1z1*gi2));
  b5=-0.5*(x2*(tru3+tru1y2+tru1z2+tru1*gi2)+ 
      y2*(tru4+tru2y2+tru2z2+tru2*gi2)+ 
       yx2*(tru3y1+tru1y3+tru1y1z2+tru1y1*gi2)+ 
       zx2*(tru3z1+tru1y2z1+tru1z3+tru1z1*gi2)+ 
       z2*(tru5+tru1y4+tru1z4+tru1*gi4+2.0*(tru3y2+tru3z2+ 
       tru1y2z2+tru1z2*gi2+tru3*gi2+tru1y2*gi2)));
    
  dxe3d(axu1,ayu1,azu1,ae1,a1,a2,a3,a4,a5,&a11,&b11,&c11,&d11,&e11,ck,ddim);
  dxe3d(axu2,ayu2,azu2,ae2,b1,b2,b3,b4,b5,&a22,&b22,&c22,&d22,&e22,ck,ddim);
    
  aetri1u1=a11*teu[1]+b11*teu[2]+c11*teu1y1+d11*teu1z1+ 
       e11*(teu[3]+teu1y2+teu1z2+teu1i2);
  aetri1u2=a11*teu[2]+b11*teu[3]+c11*teu2y1+d11*teu2z1+ 
       e11*(teu[4]+teu2y2+teu2z2+teu2i2);
  aetri1u3=a11*teu1y1+b11*teu2y1 
       +c11*teu1y2+d11*teu1y1z1+ 
       e11*(teu3y1+teu1y3+teu1y1z2+teu1y1i2);
  aetri1u4=a11*teu1z1+b11*teu2z1+c11*teu1y1z1+d11*teu1z2+ 
       e11*(teu3z1+teu1y2z1+teu1z3+teu1z1i2);
  aetri1eu1=0.5* 
       (a11*(teu[3]+teu1y2+teu1z2+teu1i2)+ 
       b11*(teu[4]+teu2y2+teu2z2+teu2i2)+ 
       c11*(teu3y1+teu1y3+teu1y1z2+teu1y1i2)+ 
       d11*(teu3z1+teu1y2z1+teu1z3+teu1z1i2)+ 
       e11*(teu[5]+teu1y4+teu1z4+teu1i4 
       +2.0*teu3y2+2.0*teu3z2+2.0*teu1y2z2+ 
       2.0*teu1z2i2+2.0*teu3i2+2.0*teu1y2i2));
    
  agtri1u1=a22*tgu[1]+b22*tgu[2]+c22*tgu1y1+d22*tgu1z1+ 
       e22*(tgu[3]+tgu1y2+tgu1z2+tgu1i2);
  agtri1u2=a22*tgu[2]+b22*tgu[3]+c22*tgu2y1+d22*tgu2z1+ 
       e22*(tgu[4]+tgu2y2+tgu2z2+tgu2i2);
  agtri1u3=a22*tgu1y1+b22*tgu2y1 
       +c22*tgu1y2+d22*tgu1y1z1+ 
       e22*(tgu3y1+tgu1y3+tgu1y1z2+tgu1y1i2);
  agtri1u4=a22*tgu1z1+b22*tgu2z1+c22*tgu1y1z1+d22*tgu1z2+ 
       e22*(tgu3z1+tgu1y2z1+tgu1z3+tgu1z1i2);
  agtri1eu1=0.5* 
       (a22*(tgu[3]+tgu1y2+tgu1z2+tgu1i2)+ 
       b22*(tgu[4]+tgu2y2+tgu2z2+tgu2i2)+ 
       c22*(tgu3y1+tgu1y3+tgu1y1z2+tgu1y1i2)+ 
       d22*(tgu3z1+tgu1y2z1+tgu1z3+tgu1z1i2)+ 
       e22*(tgu[5]+tgu1y4+tgu1z4+tgu1i4 
       +2.0*tgu3y2+2.0*tgu3z2+2.0*tgu1y2z2+ 
       2.0*tgu1z2i2+2.0*tgu3i2+2.0*tgu1y2i2));
  
  fmll=(-se1*te)*aetri1u1;
  fpxll=(-se1*te)*aetri1u2;
  fpyll=(-se1*te)*aetri1u3;
  fpzll=(-se1*te)*aetri1u4;
  fell=(-se1*te)*aetri1eu1;
    
  fmrr=(-se1*te)*agtri1u1;
  fpxrr=(-se1*te)*agtri1u2;
  fpyrr=(-se1*te)*agtri1u3;
  fpzrr=(-se1*te)*agtri1u4;
  ferr=(-se1*te)*agtri1eu1;

//  End of extra Navier Stokes stuff

#endif


/*
=====================================================================
We can get "g" now since we've already computed "g_0" and the 
integrals above.  The state "g" is Taylor expanded in both space 
AND time.
=====================================================================
*/

// Slopes for "g" (Kun Xu pg. 69-70)

#ifdef DEBUG_ALL
  fprintf(stdout,"aen=%13.5e,ae[0]=%13.5e,ae[1]=%13.5e,aen-ae[0]=%13.5e,ae[1]-aen=%13.5e,dxx=%13.5e\n",aen,ae[0],ae[1],aen-ae[0],ae[1]-aen,dxx);
  fprintf(stdout,"ae[0]-ae[1]=%11.5e\n",ae[0]-ae[1]);
#endif

  awl  = 2.0*(ade-ad[0])/dxx;
  bxwl = 2.0*(axm-ax[0])/dxx;
  bywl = 2.0*(aym-ay[0])/dxx;
  bzwl = 2.0*(azm-az[0])/dxx;
  cwl  = 2.0*(aen-ae[0])/dxx;

  awr  = 2.0*(ad[1]-ade)/dxx;
  bxwr = 2.0*(ax[1]-axm)/dxx;
  bywr = 2.0*(ay[1]-aym)/dxx;
  bzwr = 2.0*(az[1]-azm)/dxx;
  cwr  = 2.0*(ae[1]-aen)/dxx;

#ifdef DEBUG_ALL
  fprintf(stdout,"GRAPE awl=%13.5e,bxwl=%13.5e,bywl=%13.5e,bzwl=%13.5e,cwl=%13.5e\n",awl,bxwl,bywl,bzwl,cwl);
  fprintf(stdout,"GRAPE awr=%13.5e,bxwr=%13.5e,bywr=%13.5e,bzwr=%13.5e,cwr=%13.5e\n",awr,bxwr,bywr,bzwr,cwr);
#endif

  dxe3d(axu0,ayu0,azu0,ae0,awl,bxwl,bywl,bzwl,cwl,&x1,&y1,&yx1,&zx1,&z1,ck,ddim);
  dxe3d(axu0,ayu0,azu0,ae0,awr,bxwr,bywr,bzwr,cwr,&x2,&y2,&yx2,&zx2,&z2,ck,ddim);

// Moments of the initial Maxwellian "g_0".  Here both sides of
// the wall are identical, only the velocity half-space which
// you are integrating differs, but we insist that the values of
// mass-density, velocity, and temperature at the wall are the same
// regardless of whether the interpolation is done from the left or
// from the right.  Whereas for "f_0" the values of mass-density, 
// velocity, and temperature could be different depending on whether
// they are interpolated from the left or the right.
// NOTE: teu and tgu are being recomputed here for axu0. Thus,
// we CANNOT use TEU/TGU for f0 fluxes after this point!

#ifdef DEBUG_ALL
  fprintf(stdout,"x1=%11.5e,y1=%11.5e,yx1=%11.5e,zx1=%11.5e,z1=%11.5e\n",x1,y1,yx1,zx1,z1);
  fprintf(stdout,"x2=%11.5e,y2=%11.5e,yx2=%11.5e,zx2=%11.5e,z2=%11.5e\n",x2,y2,yx2,zx2,z2);
#endif
 
  t0 = 1.0;
  t1 = axu0;
  t2 = axu0*t1+0.5*t0*rae0;
  t3 = axu0*t2+1.0*t1*rae0;
  t4 = axu0*t3+1.5*t2*rae0;
  t5 = axu0*t4+2.0*t3*rae0;
  t6 = axu0*t5+2.5*t4*rae0;

  ae1 = ae0;
  ae2 = ae0;
  rae1 = one/ae0;
  rae2 = rae1;

  axu1 = axu0;
  ayu1 = ayu0;
  azu1 = azu0;
  axu2 = axu0;
  ayu2 = ayu0;
  azu2 = azu0;

  rade1 = one/ade;
  rade2 = rade1;


  precision_arg = std::min(-axu1*std::sqrt(ae1),mach_precision);
//precision_arg = -axu1*sqrt(ae1);
  te1    = 0.5*erfc(precision_arg);
  te2    = 0.5*std::exp(-precision_arg*precision_arg)/std::sqrt(ae1*pi);

  teu[1] = axu1*te1+te2;
  teu[2] = axu1*teu[1]+0.5*te1*rae1;
  teu[3] = axu1*teu[2]+1.0*teu[1]*rae1;
  teu[4] = axu1*teu[3]+1.5*teu[2]*rae1;
  teu[5] = axu1*teu[4]+2.0*teu[3]*rae1;
  teu[6] = axu1*teu[5]+2.5*teu[4]*rae1;

// Try computing these moments the long way instead
// of taking the shortcut of the subtraction


precision_arg = std::min(axu2*std::sqrt(ae2),mach_precision);
//precision_arg = axu2*sqrt(ae2);
  tg1    = 0.5*erfc(precision_arg);
  tg2    = -0.5*std::exp(-precision_arg*precision_arg)/std::sqrt(ae2*pi);

  tgu[1] = axu2*tg1+tg2;
  tgu[2] = axu2*tgu[1]+0.5*tg1*rae2;
  tgu[3] = axu2*tgu[2]+1.0*tgu[1]*rae2;
  tgu[4] = axu2*tgu[3]+1.5*tgu[2]*rae2;
  tgu[5] = axu2*tgu[4]+2.0*tgu[3]*rae2;
  tgu[6] = axu2*tgu[5]+2.5*tgu[4]*rae2;

// Velocity moments of the y-direction on the left and right

  ey0 = 1.0;       
  ey1 = ayu1;
  ey2 = ayu1*ey1+0.5*ey0*rae1;
  ey3 = ayu1*ey2+1.0*ey1*rae1;
  ey4 = ayu1*ey3+1.5*ey2*rae1;
  ey5 = ayu1*ey4+2.0*ey3*rae1;
  ei2 = 0.5*ck*rae1;
  ei4 = 0.25*ck*(ck+2.0)*rae1*rae1;
  
  gy0 = ey0;
  gy1 = ey1;
  gy2 = ey2;
  gy3 = ey3;
  gy4 = ey4;
  gy5 = ey5;
  gi2 = ei2;
  gi4 = ei4;

// Velocity moments of the z-direction on the left and right

  ez0 = 1.0;         
  ez1 = azu1;
  ez2 = azu1*ez1+0.5*ez0*rae1;
  ez3 = azu1*ez2+1.0*ez1*rae1;
  ez4 = azu1*ez3+1.5*ez2*rae1;
  ez5 = azu1*ez4+2.0*ez3*rae1;

  gz0 = ez0;
  gz1 = ez1;
  gz2 = ez2;
  gz3 = ez3;
  gz4 = ez4;
  gz5 = ez5;

// Combined moments of u and v on the left

  te1y1   = te1*ey1;
  te1y2   = te1*ey2;
  te1y3   = te1*ey3;
  te1y4   = te1*ey4;
  te1y5   = te1*ey5;
  te1i2   = te1*ei2;
  te1y1i2 = te1y1*ei2;
  te1y2i2 = te1y2*ei2;
  te1y1i4 = te1y1*ei4;
  te1y3i2 = te1y3*ei2;
  
  teu1y1   = teu[1]*ey1;
  teu1y2   = teu[1]*ey2;
  teu1y3   = teu[1]*ey3;
  teu1y4   = teu[1]*ey4;
  teu1i2   = teu[1]*ei2;
  teu1i4   = teu[1]*ei4;
  teu1y1i2 = teu1y1*ei2;
  teu1y2i2 = teu1y2*ei2;
  teu1y1i4 = teu1y1*ei4;
  teu1y3i2 = teu1y3*ei2;
  
  teu2y1   = teu[2]*ey1;
  teu2y2   = teu[2]*ey2;
  teu2y3   = teu[2]*ey3;
  teu2y4   = teu[2]*ey4;
  teu2i2   = teu[2]*ei2;
  teu2i4   = teu[2]*ei4;
  teu2y1i2 = teu2y1*ei2;
  teu2y2i2 = teu2y2*ei2;
  
  teu3y1   = teu[3]*ey1;
  teu3y2   = teu[3]*ey2;
  teu3y3   = teu[3]*ey3;
  teu3i2   = teu[3]*ei2;
  teu3y1i2 = teu3y1*ei2;
  
  teu4y1 = teu[4]*ey1;
  teu4y2 = teu[4]*ey2;
  teu4i2 = teu[4]*ei2;
  
  teu5y1 = teu[5]*ey1;
  
// Combined moments of u and v on the right

  tg1y1   = tg1*gy1;
  tg1y2   = tg1*gy2;
  tg1y3   = tg1*gy3;
  tg1y4   = tg1*gy4;
  tg1y5   = tg1*gy5;
  tg1i2   = tg1*gi2;
  tg1y1i2 = tg1y1*gi2;
  tg1y2i2 = tg1y2*gi2;
  tg1y1i4 = tg1y1*gi4;
  tg1y3i2 = tg1y3*gi2;
  
  tgu1y1   = tgu[1]*gy1;
  tgu1y2   = tgu[1]*gy2;
  tgu1y3   = tgu[1]*gy3;
  tgu1y4   = tgu[1]*gy4;
  tgu1i2   = tgu[1]*gi2;
  tgu1i4   = tgu[1]*gi4;
  tgu1y1i2 = tgu1y1*gi2;
  tgu1y2i2 = tgu1y2*gi2;
  tgu1y1i4 = tgu1y1*gi4;
  tgu1y3i2 = tgu1y3*gi2;
  
  tgu2y1   = tgu[2]*gy1;
  tgu2y2   = tgu[2]*gy2;
  tgu2y3   = tgu[2]*gy3;
  tgu2y4   = tgu[2]*gy4;
  tgu2i2   = tgu[2]*gi2;
  tgu2i4   = tgu[2]*gi4;
  tgu2y1i2 = tgu2y1*gi2;
  tgu2y2i2 = tgu2y2*gi2;
  
  tgu3y1   = tgu[3]*gy1;
  tgu3y2   = tgu[3]*gy2;
  tgu3i2   = tgu[3]*gi2;
  tgu3y1i2 = tgu3y1*gi2;
  
  tgu4y1 = tgu[4]*gy1;
  tgu4y2 = tgu[4]*gy2;
  tgu4i2 = tgu[4]*gi2;
  
  tgu5y1 = tgu[5]*gy1;
  
// Combined moments of u and w on the left

  teu1z1   = teu[1]*ez1;
  teu1z2   = teu[1]*ez2;
  teu1z3   = teu[1]*ez3;
  teu1z4   = teu[1]*ez4;
  teu1z1i2 = teu1z1*ei2;
  teu1z2i2 = teu1z2*ei2;
  
  teu2z1   = teu[2]*ez1;
  teu2z2   = teu[2]*ez2;
  teu2z3   = teu[2]*ez3;
  teu2z4   = teu[2]*ez4;
  teu2z1i2 = teu2z1*ei2;
  teu2z2i2 = teu2z2*ei2;
  
  teu3z1   = teu[3]*ez1;
  teu3z2   = teu[3]*ez2;
  
  teu4z1 = teu[4]*ez1;
  teu4z2 = teu[4]*ez2;
  
  // Combined moments of u and w on the right
  
  tgu1z1   = tgu[1]*gz1;
  tgu1z2   = tgu[1]*gz2;
  tgu1z3   = tgu[1]*gz3;
  tgu1z4   = tgu[1]*gz4;
  tgu1z1i2 = tgu1z1*gi2;
  tgu1z2i2 = tgu1z2*gi2;
  
  tgu2z1   = tgu[2]*gz1;
  tgu2z2   = tgu[2]*gz2;
  tgu2z3   = tgu[2]*gz3;
  tgu2z4   = tgu[2]*gz4;
  tgu2z1i2 = tgu2z1*gi2;
  tgu2z2i2 = tgu2z2*gi2;
  
  tgu3z1   = tgu[3]*gz1;
  tgu3z2   = tgu[3]*gz2;
  
  tgu4z1 = tgu[4]*gz1;
  tgu4z2 = tgu[4]*gz2;
  
// Combined moments of u, v, and w on the left

  teu1y1z1   = teu[1]*ey1*ez1;
  teu1y1z2   = teu[1]*ey1*ez2;
  teu1y2z1   = teu[1]*ey2*ez1;
  teu1y2z2   = teu[1]*ey2*ez2;
  
  teu2y1z1   = teu[2]*ey1*ez1;
  teu2y1z2   = teu[2]*ey1*ez2;
  teu2y2z1   = teu[2]*ey2*ez1;
  teu2y2z2   = teu[2]*ey2*ez2;
  
// Combined moments of u, v, and w on the right

  tgu1y1z1   = tgu[1]*gy1*gz1;
  tgu1y1z2   = tgu[1]*gy1*gz2;
  tgu1y2z1   = tgu[1]*gy2*gz1;
  tgu1y2z2   = tgu[1]*gy2*gz2;
  
  tgu2y1z1   = tgu[2]*gy1*gz1;
  tgu2y1z2   = tgu[2]*gy1*gz2;
  tgu2y2z1   = tgu[2]*gy2*gz1;
  tgu2y2z2   = tgu[2]*gy2*gz2;

//    Combination of moments of "g_0" entering the calculation of 
//    the A_bar coefficients of "g". In eqn. (4.37) the second term
//    in the integral gives "triu", "triu2", "triuy1", "trieu".


#ifdef DEBUG_ALL
#ifdef FLUXES_G0
  fprintf(stdout,"    x1=%13.5e,teu[2]=%13.5e,teu[3]=%13.5e,teu2i2=%13.5e\n",x1,teu[2],teu[3],teu2i2);
  fprintf(stdout,"    x2=%13.5e,tgu[2]=%13.5e,tgu[3]=%13.5e,tgu2i2=%13.5e\n",x2,tgu[2],tgu[3],tgu2i2);
#endif
#endif
  

  triu   = (x1*teu[1]+y1*teu[2]+yx1*teu1y1+zx1*teu1z1+ 
           z1*(teu[3]+teu1y2+teu1z2+teu1i2))+ 
           (x2*tgu[1]+y2*tgu[2]+yx2*tgu1y1+zx2*tgu1z1+ 
           z2*(tgu[3]+tgu1y2+tgu1z2+tgu1i2));

  gm1    = (x1*teu[2]+y1*teu[3]+yx1*teu2y1+zx1*teu2z1+ 
           z1*(teu[4]+teu2y2+teu2z2+teu2i2));

  gm2    = (x2*tgu[2]+y2*tgu[3]+yx2*tgu2y1+zx2*tgu2z1+ 
           z2*(tgu[4]+tgu2y2+tgu2z2+tgu2i2));

  triu2  = gm1+gm2;

  triuy1 = (x1*teu1y1+y1*teu2y1+yx1*teu1y2+zx1*teu1y1z1+ 
           z1*(teu3y1+teu1y3+teu1y1z2+teu1y1i2))+ 
           (x2*tgu1y1+y2*tgu2y1 
           +yx2*tgu1y2+zx2*tgu1y1z1+ 
           z2*(tgu3y1+tgu1y3+tgu1y1z2+tgu1y1i2));

  triuz1 = x1*teu1z1+y1*teu2z1+ 
           yx1*teu1y1z1+zx1*teu1z2+z1*(teu3z1+teu1y2z1+teu1z1i2+teu1z3) 
           +x2*tgu1z1+y2*tgu2z1+ 
           yx2*tgu1y1z1+zx2*tgu1z2+z2*(tgu3z1+tgu1y2z1+tgu1z1i2+tgu1z3);

  trieu  = 0.5*(x1*(teu[3]+teu1y2+teu1z2+teu1i2)+ 
           y1*(teu[4]+teu2y2+teu2z2+teu2i2)+ 
           yx1*(teu3y1+teu1y3+teu1y1z2+teu1y1i2)+ 
           zx1*(teu3z1+teu1y2z1+teu1z3+teu1z1i2)+ 
           z1*(teu[5]+teu1y4+teu1z4+teu1i4 
           +2.0*teu3y2+2.0*teu3z2+2.0*teu3i2+2.0*teu1y2i2 
           +2.0*teu1y2z2 + 2.0*teu1z2i2)) 
           +0.5* 
           (x2*(tgu[3]+tgu1y2+tgu1z2+tgu1i2)+ 
           y2*(tgu[4]+tgu2y2+tgu2z2+tgu2i2)+ 
           yx2*(tgu3y1+tgu1y3+tgu1y1z2+tgu1y1i2)+ 
           zx2*(tgu3z1+tgu1y2z1+tgu1z3+tgu1z1i2) 
           +z2*(tgu[5]+tgu1y4+tgu1z4+tgu1i4 
           +2.0*tgu3y2+2.0*tgu3z2+2.0*tgu3i2+2.0*tgu1y2i2 
           +2.0*tgu1y2z2 + 2.0*tgu1z2i2));

//#ifdef DEBUG_ALL
//fprintf(stdout,"TANGERINE x1=%13.5e, y1=%13.5e, teu[4]=%13.5e, teu2y2=%13.5e, teu2z2=%13.5e, teu2i2=%13.5e, yx1=%13.5e, teu3y1=%13.5e, teu1y3=%13.5e, teu1y1z2=%13.5e, teu1y1i2=%13.5e, zx1=%13.5e, teu3z1=%13.5e, teu1y2z1=%13.5e, teu1z3=%13.5e, teu1z1i2=%13.5e, z1=%13.5e, teu[5]=%13.5e, teu1y4=%13.5e, teu1z4=%13.5e, teu1i4=%13.5e, teu3y2=%13.5e, teu3z2=%13.5e, teu3i2=%13.5e, teu1y2i2=%13.5e, teu1y2z2=%13.5e, teu1z2i2=%13.5e, x2=%13.5e, y2=%13.5e, tgu[4]=%13.5e, tgu2y2=%13.5e, tgu2z2=%13.5e, tgu2i2=%13.5e, yx2=%13.5e, tgu3y1=%13.5e, tgu1y3=%13.5e, tgu1y1z2=%13.5e, tgu1y1i2=%13.5e, zx2=%13.5e, tgu3z1=%13.5e, tgu1z3=%13.5e, tgu1y2z1=%13.5e, tgu1z1i2=%13.5e, z2=%13.5e, tgu[5]=%13.5e, tgu1y4=%13.5e, tgu1z4=%13.5e, tgu1i4=%13.5e, tgu3y2=%13.5e, tgu3z2=%13.5e, tgu3i2=%13.5e, tgu1y2i2=%13.5e, tgu1y2z2=%13.5e, tgu1z2i2=%13.5e\n",x1,y1,teu[4],teu2y2,teu2z2,teu2i2,yx1,teu3y1,teu1y3,teu1y1z2,teu1y1i2,zx1,teu3z1,teu1y2z1,teu1z3,teu1z1i2,z1,teu[5],teu1y4,teu1z4,teu1i4,teu3y2,teu3z2,teu3i2,teu1y2i2,teu1y2z2,teu1z2i2,x2,y2,tgu[4],tgu2y2,tgu2z2,tgu2i2,yx2,tgu3y1,tgu1y3,tgu1y1z2,tgu1y1i2,zx2,tgu3z1,tgu1z3,tgu1y2z1,tgu1z1i2,z2,tgu[5],tgu1y4,tgu1z4,tgu1i4,tgu3y2,tgu3z2,tgu3i2,tgu1y2i2,tgu1y2z2,tgu1z2i2); 
//#endif

  gpx1   = (x1*teu[3]+y1*teu[4]+yx1*teu3y1+zx1*teu3z1+ 
           z1*(teu[5]+teu3y2+teu3z2+teu3i2));

  gpx2   = (x2*tgu[3]+y2*tgu[4]+yx2*tgu3y1+zx2*tgu3z1 
           +z2*(tgu[5]+tgu3y2+tgu3z2+tgu3i2));

  triu3  = gpx1 + gpx2;

  gpy1   = (x1*teu2y1+y1*teu3y1+yx1*teu2y2+zx1*teu2y1z1+ 
           z1*(teu4y1+teu2y3+teu2y1z2+teu2y1i2));

  gpy2   = (x2*tgu2y1+y2*tgu3y1+yx2*tgu2y2+zx2*tgu2y1z1+ 
           z2*(tgu4y1+tgu2y3+tgu2y1z2+tgu2y1i2));

  triu2y1= gpy1 + gpy2;

  gpz1   = x1*teu2z1+y1*teu3z1+yx1*teu2y1z1+zx1*teu2z2+ 
           z1*(teu4z1+teu2z1i2+teu2z3+teu2y2z1);

  gpz2   = x2*tgu2z1+y2*tgu3z1+yx2*tgu2y1z1+zx2*tgu2z2+ 
           z2*(tgu4z1+tgu2z1i2+tgu2z3+tgu2y2z1);

  triu2z1= gpz1 + gpz2;

  ge1    = 0.5*(x1*(teu[4]+teu2y2+teu2z2+teu2i2) 
           +y1*(teu[5]+teu3y2+teu3z2+teu3i2)+ 
           yx1*(teu4y1+teu2y3+teu2y1z2+teu2y1i2)+ 
           zx1*(teu4z1+teu2z1i2+teu2z3+teu2y2z1)+ 
           z1*(teu[6]+teu2y4+teu2z4+teu2i4 
           +2.0*teu4y2+2.0*teu4z2+2.0*teu4i2+2.0*teu2y2i2 
           +2.0*teu2y2z2+2.0*teu2z2i2));

  ge2    = 0.5*(x2*(tgu[4]+tgu2y2+tgu2z2+tgu2i2) 
           +y2*(tgu[5]+tgu3y2+tgu3z2+tgu3i2)+ 
           yx2*(tgu4y1+tgu2y3+tgu2y1z2+tgu2y1i2) 
           +zx2*(tgu4z1+tgu2z1i2+tgu2z3+tgu2y2z1) 
           +z2*(tgu[6]+tgu2y4+tgu2z4+tgu2i4 
           +2.0*tgu4y2+2.0*tgu4z2+2.0*tgu4i2+2.0*tgu2y2i2 
           +2.0*tgu2y2z2+2.0*tgu2z2i2));

  trieu2 = ge1 + ge2;

// These are the 5 right hand side components of eqn. (4.37) of
// Kun Xu pg. 78
  et  = set2/set0;

  as  = (-triu*set2+se1*ade-ami)/set0 
      +  et*2.0*ade*ae0*( 
         gra1*(teu[1]-axu0*te1)+gra2*(tgu[1]-axu0*tg1));
  bs  = (-triu2*set2+se1*axm-api)/set0 
      +  et*2.0*ade*ae0*( 
         gra1*(teu[2]-axu0*teu[1])+gra2*(tgu[2]-axu0*tgu[1]));
  bsx = (-triuy1*set2+se1*aym-apix)/set0 
      +  et*2.0*ade*ae0*( 
         gra1*(teu1y1-axu0*te1y1)+gra2*(tgu1y1-axu0*tg1y1));
  bsz = (-(triuz1)*set2+se1*azm-apiz)/set0 
      +  et*2.0*ade*ae0*( 
         gra1*(teu1z1-axu0*te1z1)+gra2*(tgu1z1-axu0*tg1z1));
  cs  = (-trieu*set2+se1*aen-aei)/set0 
      +  et*ade*ae0*(gra1*(teu[3]+teu1y2+teu1z2+teu1i2 
      -  axu0*teu[2]-axu0*te1y2-axu0*te1z2-axu0*te1i2) 
      +  gra2*(tgu[3]+tgu1y2+tgu1z2+tgu1i2 
      -  axu0*tgu[2]-axu0*tg1y2-axu0*tg1z2-axu0*tg1i2));

//#ifdef DEBUG_ALL
//fprintf(stdout,"PEACH trieu=%13.5e, set2=%13.5e, se1=%13.5e,\n aen=%13.5e, aei=%13.5e, set0=%13.5e,\n et=%13.5e, ade=%13.5e,\n gra1=%13.5e, teu[3]=%13.5e, teu1y2=%13.5e,\n teu1z2=%13.5e, teu1i2=%13.5e, teu[2]=%13.5e,\n te1y2=%13.5e, te1z2=%13.5e, te1i2=%13.5e,\n gra2=%13.5e, tgu[3]=%13.5e, tgu1y2=%13.5e,\n tgu1z2=%13.5e, tgu1i2=%13.5e,\n tgu[2]=%13.5e, tg1y2=%13.5e, tg1z2=%13.5e, tg1i2=%13.5e\n", trieu, set2, se1, aen, aei, set0, et, ade, gra1, teu[3], teu1y2, teu1z2, teu1i2, teu[2], te1y2, te1z2, te1i2, gra2, tgu[3], tgu1y2, tgu1z2, tgu1i2, tgu[2], tg1y2, tg1z2,tg1i2);
//#endif


#ifdef DEBUG_ALL
  fprintf(stdout,"PAPAYA axu0=%13.5e,ayu0=%13.5e,azu0=%13.5e,ae0=%13.5e,as=%13.5e,bs=%13.5e,bsx=%13.5e,bsz=%13.5e,cs=%13.5e,ck=%13.5e,ddim=%13.5e\n",axu0,ayu0,azu0,ae0,as,bs,bsx,bsz,cs,ck,ddim);
#endif

// Calculate the A_bar (A1,B1,C1,F1,D1)
  dxe3d(axu0,ayu0,azu0,ae0,as,bs,bsx,bsz,cs,&a1,&b1,&c1,&f1,&d1,ck,ddim);

#ifdef DEBUG_ALL
  fprintf(stdout,"YEE! a1=%13.5e,b1=%13.5e,c1=%13.5e,f1=%13.5e,ck=%13.5e,ddim=%13.5e\n",a1,b1,c1,f1,ck,ddim);
#endif
/*
=====================================================================

We now have the contribution of "f_0" to "f" over a timestep, and 
"g", so we need to calculate the contribution of "g" to "f" over a 
time step.  This means we need to calculate the integral of the 
moments of the first term in eqn. (4.34) over a time step.

=====================================================================
*/

  t0 = 1.0;
  t1 = axu0;
  t2 = axu0*t1+0.5*t0*rae0;
  t3 = axu0*t2+1.0*t1*rae0;
  t4 = axu0*t3+1.5*t2*rae0;
  t5 = axu0*t4+2.0*t3*rae0;
  t6 = axu0*t5+2.5*t4*rae0;

  t02 = 0.5*ck*rae0;
  t12 = axu0*t02;
  t22 = t2*t02;
  t32 = t3*t02;
  t42 = t4*t02;
  t04 = (3.0*ck+ck*(ck-1.0))/(4.0*ae0*ae0);
  t14 = axu0*t04;
  t24 = t2*t04;

// The y-moments

  y0 = 1.0;
  y1 = ayu0;
  y2 = ayu0*y1+0.5*y0*rae0;
  y3 = ayu0*y2+1.0*y1*rae0;
  y4 = ayu0*y3+1.5*y2*rae0;
  y5 = ayu0*y4+2.0*y3*rae0;

  t1y1 = t1*y1;
  t2y1 = t2*y1;
  t3y1 = t3*y1;
  t4y1 = t4*y1;
  t5y1 = t5*y1;
      
  t02y1 = t02*y1;
  t12y1 = t12*y1;
  t22y1 = t22*y1;
  t32y1 = t32*y1;
  t04y1 = t04*y1;
  t14y1 = t14*y1;

  t1y2  = t1*y2;
  t2y2  = t2*y2;
  t3y2  = t3*y2;
  t4y2  = t4*y2;
  t02y2 = t02*y2;
  t12y2 = t12*y2;
  t22y2 = t22*y2;

  t1y3  = t1*y3;
  t2y3  = t2*y3;
  t3y3  = t3*y3;
  t02y3 = t02*y3;
  t12y3 = t12*y3;

  t1y4 = t1*y4;
  t2y4 = t2*y4;

  t1y5 = t1*y5;

// The z-moments

  z0 = 1.0;
  z1 = azu0;
  z2 = azu0*z1+0.5*z0*rae0;
  z3 = azu0*z2+1.0*z1*rae0;
  z4 = azu0*z3+1.5*z2*rae0;
  z5 = azu0*z4+2.0*z3*rae0;

  t1z1 = t1*z1;
  t2z1 = t2*z1;
  t3z1 = t3*z1;
      
  t12z1 = t12*z1;

  t1z2  = t1*z2;
  t2z2  = t2*z2;
  t3z2  = t3*z2;
  t4z2  = t4*z2;
  t02z2 = t02*z2;
  t12z2 = t12*z2;
  t22z2 = t22*z2;

  t1z3  = t1*z3;
  t2z3  = t2*z3;
  t3z3  = t3*z3;

  t1z4   = t1*z4;
  t2z4   = t2*z4;

  t1y1z1 = t1*y1*z1;
  t1y1z2 = t1*y1*z2;
  t1y2z1 = t1*y2*z1;
  t1y2z2 = t1*y2*z2;

// Other bits necessary for the contribution

  atriu   = a1*t1+b1*t2+c1*t1y1+f1*t1z1+d1*(t3+t12+t1y2+t1z2);
  atriuy1 = a1*t1y1+b1*t2y1+c1*t1y2+f1*t1y1z1+d1*(t3y1+t12y1+ 
            t1y3+t1y1z2);
  atriuz1 = a1*t1z1+b1*t2z1+c1*t1y1z1+f1*t1z2+ 
            d1*(t3z1+t12z1+t1z3+t1y2z1) ;
  atriu2  = a1*t2+b1*t3+c1*t2y1+f1*t2z1+d1*(t4+t22+t2y2+t2z2);
#ifdef DEBUG_ALL
  fprintf(stdout,"GUAVA a1=%13.5e,t2=%13.5e,b1=%13.5e,t3=%13.e,c1=%13.5e,t2y1=%13.5e,f1=%13.5e,t2z1=%13.5e,d1=%13.5e,t4=%13.5e,t22=%13.5e,t2y2=%13.5e,t2z2=%13.5e\n",a1,t2,b1,t3,c1,t2y1,f1,t2z1,d1,t4,t22,t2y2,t2z2);
#endif

  atrieu  = 0.5*(a1*(t3+t12+t1y2+t1z2)+b1*(t4+t22+t2y2+t2z2) 
            +c1*(t3y1+t12y1+t1y3+t1y1z2) 
            +f1*(t3z1+t12z1+t1z3+t1y2z1) 
            +d1*(t5+t1y4+t1z4+t14+2.0*t3y2+2.0*t3z2 
            +2.0*t32+2.0*t12y2+2.0*t1y2z2+2.0*t12z2));

// NEW: Here we calculate the fluxes arising from Abar, so that we can shut them off more easily.
// Once and for all: A*T1 + B*TRIU2 are the g0 fluxes. D*ATRIU is the capital A (time dependence)

  a = ade*(tau-se1);
  b = set2;
  d = 0.5*tau*tau-te*tau+te*te*(1.0-sw);

#ifdef DEBUG_ALL
fprintf(stdout,"t1=%13.5e,t2=%13.5e,t3=%13.5e,t1y1=%13.5e,t1z1=%13.5e,a=%13.5e,b=%13.5e,triu2=%13.5e,triu3=%13.5e,triu2y1=%13.5e,triu2z1=%13.5e,t12=%13.5e,t1y2=%13.5e,t1z2=%13.5e,trieu2=%13.5e\n", t1,t2,t3,t1y1,t1z1,a,b,triu2,triu3,triu2y1,triu2z1,t12,t1y2,t1z2,trieu2);
fprintf(stdout,"a=%13.5e,b=%13.5e,d=%13.5e\n",a,b,d);
#endif

#ifdef FLUXES_G0
  fmg0    = a*t1   + b*triu2;
  fpg0    = a*t2   + b*triu3;
  fpxg0   = a*t1y1 + b*triu2y1;
  fpzg0   = a*t1z1 + b*triu2z1;
  feg0    = 0.5*a*(t3+t12+t1y2+t1z2) + b*trieu2; 
#else
  fmg0    = 0.0;
  fpg0    = 0.0;
  fpxg0   = 0.0;
  fpzg0   = 0.0;
  feg0    = 0.0;
#endif

#ifdef FLUXES_ABAR
  fmAbar  = d*atriu;
  fpAbar  = d*atriu2;
  fpxAbar = d*atriuy1;
  fpzAbar = d*atriuz1;
  feAbar  = d*atrieu;
#else
  fmAbar  = 0.0;
  fpAbar  = 0.0;
  fpxAbar = 0.0;
  fpzAbar = 0.0;
  feAbar  = 0.0;
#endif

//     At last we combine all the bits to derive the fluxes: 
//     last equation in Kun p78 ... 

//    if cells on both sides of the wall have a divergent hyper-machian 
//    flow, then the equilibrium maxwellian at the wall is zero (to numerical
//    precision) and therefore the flux at that wall is also zero (hydrodynamics
//    cannot be applied anymore, have to use rarefied gas dynamics). The zero
//    flux "fix" here is actually the flux given by the beam scheme based on 
//    the collisionless Boltzmann equation and therefore should be the best 
//    approximation to rarefied gas dynamics ...

 
  if ( (-axu1t*std::sqrt(ae1t) <= mach_precision) ||  (axu2t*std::sqrt(ae2t) <= mach_precision)) {

    afm =   fmg0
          + fmAbar
#ifdef UNAVSTOKES
          + fmll + fmrr
#endif
#ifdef UGRAV2ND
          - b*2.0*ade*ae0*(gra1*(teu[2]-axu0*teu[1])+gra2*(tgu[2]-axu0*tgu[1])) 
#endif
          + fm1 + fm2; 

    afp =   fpg0
          + fpAbar
#ifdef UNAVSTOKES
          + fpxll + fpxrr 
#endif
#ifdef UGRAV2ND
          - b*2.0*ade*ae0*(gra1*(teu[3]-axu0*teu[2])+gra2*(tgu[3]-axu0*tgu[2])) 
#endif
          + fp1 + fp2;

    afpx =   fpxg0
           + fpxAbar
#ifdef UNAVSTOKES
           + FPYLL + FPYRR 
#endif
#ifdef UGRAV2ND
           - B*2.0*ADE*AE0*(GRA1*(TEU2Y1-AXU0*TEU1Y1)+GRA2*(TGU2Y1-AXU0*TGU1Y1)) 
#endif
           + fpx1 + fpx2; 

    afpz =   fpzg0
           + fpzAbar
#ifdef UNAVSTOKES
           + FPZLL + FPZRR 
#endif
#ifdef UGRAV2ND
           - B*2.0*ADE*AE0*(GRA1*(TEU2Z1-AXU0*TEU1Z1)+GRA2*(TGU2Z1-AXU0*TGU1Z1)) 
#endif
           + fpz1 + fpz2;

    afe =   feg0
          + feAbar
#ifdef UNAVSTOKES
          + FELL + FERR
#endif
#ifdef UGRAV2ND
          - B*ADE*AE0*(  GRA1*(TEU[4]+TEU2Y2+TEU2Z2+TEU2I2 - AXU0*TEU[3]-AXU0*TEU1Y2-AXU0*TEU1Z2-AXU0*TEU1I2)
          + GRA2*(TGU[4]+TGU2Y2+TGU2Z2+TGU2I2 - AXU0*TGU[3]-AXU0*TGU1Y2-AXU0*TGU1Z2-AXU0*TGU1I2))
#endif
          + fe1 + fe2; 
  }else{

#ifdef DEBUG_ALL
    fprintf(stdout,"    without flux i=%4i il=%4i iu=%4i\n",i,il,iu);
#endif

    afm  = 0.0;
    afp  = 0.0;
    afpx = 0.0;
    afpz = 0.0;
    afe  = 0.0;
  }

#ifdef DEBUG_ALL
  fprintf(stdout,"    allfluxes: Um,Up,Wl,Wr                  = %11.3e %11.3e %11.3e %11.3e \n",pmb->phydro->u(IDN,kdif,jdif,idif),pmb->phydro->u(IDN,k,j,i),wl(IDN,k,j,i),wr(IDN,k,j,i));
  fprintf(stdout,"    allfluxes: fmg0 ,fmAbar ,fm1 ,fm2 ,afm  = %11.3e %11.3e %11.3e %11.3e %11.3e\n",fmg0 /tau,fmAbar /tau,fm1 /tau,fm2 /tau,afm /tau);
  fprintf(stdout,"    allfluxes: fpg0 ,fpAbar ,fp1 ,fp2 ,afp  = %11.3e %11.3e %11.3e %11.3e %11.3e\n",fpg0 /tau,fpAbar /tau,fp1 /tau,fp2 /tau,afp /tau);
  fprintf(stdout,"    allfluxes: fpxg0,fpxAbar,fpx1,fpx2,afpx = %11.3e %11.3e %11.3e %11.3e %11.3e\n",fpxg0/tau,fpxAbar/tau,fpx1/tau,fpx2/tau,afpx/tau);
  fprintf(stdout,"    allfluxes: fpzg0,fpzAbar,fpz1,fpz2,afpz = %11.3e %11.3e %11.3e %11.3e %11.3e\n",fpzg0/tau,fpzAbar/tau,fpz1/tau,fpz2/tau,afpz/tau);
  fprintf(stdout,"    allfluxes: feg0 ,feAbar ,fe1 ,fe2 ,afe  = %11.3e %11.3e %11.3e %11.3e %11.3e\n",feg0 /tau,feAbar /tau,fe1 /tau,fe2 /tau,afe /tau);
  fprintf(stdout,"AFM=%13.5e, AFP=%13.5e, AFPX=%13.5e, AFPZ=%13.5e, AFE=%13.5e\n",afm,afp,afpx,afpz,afe);
  fprintf(stdout,"AFM/tau=%13.5e, AFP/tau=%13.5e, AFPX/tau=%13.5e, AFPZ/tau=%13.5e, AFE/tau=%13.5e\n",afm/tau,afp/tau,afpx/tau,afpz/tau,afe/tau);
#endif
 
// NOTE: Divison by tau (=pGrid->dt) to make consistent with Athena. 
// bgk2 calculates absolute changes (F(W)*dt), Athena expects F(W).
  flx(IDN,k,j,i)  = afm /tau;
  flx(im1,k,j,i) = afp /tau;
  flx(im2,k,j,i) = afpx/tau;
  flx(im3,k,j,i) = afpz/tau;
  flx(IEN,k,j,i)  = afe /tau; 
  
  //if (i>4){
   //exit(0);
  // }
  }
  }
 }
}
return;
}


/*

===================================================
 SUBROUTINE: dxe3d.F
 PURPOSE   : This subroutine does the algebra for the matrix inversion and
             multiplication to get the slopes in space and time (see for
             example von K bottom of page 69 to 70 in 1D.) 
             For MHD (this case) the dimension needs to be 3, but otherwise
             nothing is changed against hydro, as the mag. fields do not enter
             the Maxwellian.
 INPUT     : axu...al  : momenta, lambda
             sd, sx..sl: left or right limited slope of density, momentum, energy
 OUTPUT    : qd..ql    : coefficients a for density, momentum, scalar, energy.
===================================================

*/

void dxe3d(Real axu, Real ayu, Real azu, Real al ,
           Real sd , Real sx , Real sy , Real sz , Real sl ,
           Real *qd, Real *qx, Real *qy, Real *qz, Real *ql,
           Real ck, Real ddim) {

  Real haf = 0.5;
  Real two = 2.0;
  Real ckd = ck+ddim;
  Real cc, bb, aa, dd, ee;

  ee = (SQR(axu) + SQR(ayu) + SQR(azu) + haf*ckd/al);
  cc = two*sl-ee*sd;
  bb = sy-ayu*sd;
  aa = sx-axu*sd;
  dd = sz-azu*sd;
  *ql = two*al*al*(cc-two*(axu*aa+ayu*bb+azu*dd))/ckd;
  *qx = two*al*(aa-axu*(*ql)/al);
  *qy = two*al*(bb-ayu*(*ql)/al);
  *qz = two*al*(dd-azu*(*ql)/al);
  *qd = sd-(*qx)*axu-(*qy)*ayu-(*qz)*azu-(*ql)*ee;

}


