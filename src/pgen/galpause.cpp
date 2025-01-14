//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file galpause.cpp
//  \brief Expanding Galactic wind to find galactopause
//
//

#include <algorithm>
#include <cmath>
#include <cfloat>     // FLT_MAX
#include <stdexcept>
#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
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
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include <hdf5.h>  // H5[F|P|S|T]_*, H5[A|D|F|P|S|T]*(), hid_t
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if (NSCALARS != 3)
#error: Requires NSCALARS == 3
#endif

//#define DEBUG
#define WIND

//========
// compiling interrupt to check type
// Use:    show_type_abort(TYPE_TO_CHECK);
//========
//template<typename T>
//void show_type_abort_helper()
//{
//    return __PRETTY_FUNCTION__;
//}
//#define show_type_abort(x) show_type_abort_helper< decltype(x) >()

// ahead declarations

class CoolingFunction;
CoolingFunction* pcoolfunc;
class Parameters;
Parameters* pparam;

typedef Real (*ProfileFunc_t)(Real r);
ProfileFunc_t ProfileFunc;

Real BetaProfile(const Real r);
Real ConstProfile(const Real r);
Real WindProfile(const Real r);

void SpitzerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
int RefinementCondition(MeshBlock *pmb);

Real vtrack0 = 0.0;
int itrack = 0;
int maxntrack = 20;
int ncycold=-1;
AthenaArray<Real> ttrack,rtrack; // for tracking

//========================================================================================
// Parameters
//========================================================================================

class Parameters {

  private:
    const Real kpsc = 8.80388050e-03;
    const Real gnewt= 6.67408000e-08;
    const Real secs = 2.99200577e+15;
    const Real myrs = 9.48758804e+01;
    const Real cmsc = 9.08222098e+03;
    const Real kmsc = 9.08222098e-02;
    const Real leng = 2.71740575e+19;
    const Real prsc = 8.80388050e+00;
    const Real gram = 3.35851438e+34;
    const Real msol = 1.68888383e+01;
    const Real ergs = 2.77032895e+42;
    const Real mugs = 1.66168589e-02;
    const Real matom= 1.6737236e-24;
    const Real kboltz= 1.3806e-16;
    Real gam,vesc,temp,dens0,pigm,phi0,delphi,rho0,rscal,cpar,mvir,mdot,vwind,beta,rbeta,x1min,x1max,fwind;

  public:
    // This must be unit-less. Otherwise just a nightmare.
    Parameters(const Real vesc0, const Real T0, const Real mdot0, const Real vwind0, const Real n0, const Real x1min0, const Real x1max0, const Real gamma, const Real fwind0) {
      gam    = gamma;
      vesc   = vesc0;
      temp   = T0;
      dens0  = n0;
      pigm   = 40.0;
      phi0   = 0.5*SQR(vesc);
      delphi = phi0/temp;
      rho0   = 1.81e4*1.51e-29/matom; // Capelo+10
      rscal  = std::sqrt(phi0/(4.0*PI*rho0));
      cpar   = 10.0;
      mvir   = 4.0*PI*rho0*SQR(rscal)*rscal*(std::log(1.0+cpar)-cpar/(1.0+cpar));
      mdot   = mdot0;
      vwind  = vwind0;
      beta   = 0.51;
      rbeta  = std::pow(2.82e-2/dens0,1.0/(3.0*beta))/kpsc; //SM20
      x1min  = x1min0;
      x1max  = x1max0;
      fwind  = fwind0;
    }
    Real Vesc() {return vesc;}; 
    Real Temp() {return temp;};
    Real Mdot() {return mdot;};
    Real Vwind() {return vwind;};
    Real Dens0() {return dens0;};
    Real Phi0() {return phi0;};
    Real Delphi() {return delphi;};
    Real Beta() {return beta;};
    Real Rbeta() {return rbeta;};
    Real Rscal() {return rscal;};
    Real Kpc() {return kpsc;};
    Real Pigm() {return pigm;};
    Real X1min() { return x1min;};
    Real X1max() { return x1max;};
    Real Fwind() {return fwind;};
    Real Gamma() {return gam;};
    Real GaccBeta(const Real r) {
      return 3.0*beta*temp*r/(SQR(rbeta)+SQR(r));
    }
    Real Dwind(const Real r) {
      return Mdot()/(4.0*PI*fwind*SQR(r)*Vwind());
    }
};
//========================================================================================
// Radiative loss functions
//========================================================================================

int icool, iwiso;
Real gammaheat0, x1rat, x2rat, gam, gm1, acosfwind, csound2,rmin,vexp, turbcool, zmetwind, zmethalo;
Real tolcool, coolsafe = 0.2, lengthcool;
Real lodxref, lodxderef; // refinement bounds
AthenaArray<Real> k1_, k2_, k3_, k4_, y0_, y1_, ytemp_, dydx_; // for wind profile

void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);
//void GSSAWdydr(const Real r, const AthenaArray<Real> &y, AthenaArray<Real> &dydr);
void GetSteadyStateAdbWind(const Real r0, const Real dr, const AthenaArray<Real> &y0, AthenaArray<Real> &y);
void GetDyDxSteadyStateAdbWind(const Real r, const AthenaArray<Real> &y, AthenaArray<Real> &dydx);
Real BetaPotential(const Real x1, const Real x2, const Real x3, const Real time);
Real BetaProfile(const Real r);
Real ConstProfile(const Real r);
void DdensDr(const Real s,  const AthenaArray<Real> &y, AthenaArray<Real> &k);

static void stop_this();

//========================================================================================
// Time Dependent Grid Functions
//  \brief Functions for time dependent grid, including two example boundary conditions
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
void UpdateGridData(Mesh *pm);

//Global Variables for OuterX1
Real bx0,by0,bz0,boost;

void OuterX1_HydroStat(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void InnerX1_Wind(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

Real PowerGridX1(Real x, RegionSize rs);
Real QuadraticX2(Real x, RegionSize rs);
Real ExponentialX2(Real x, RegionSize rs);

void ShockDetector(AthenaArray<Real> data, AthenaArray<Real> grid, int outArr[], Real eps);


//========================================================================================
// Unit mapping
//========================================================================================
std::map <std::string, Real> units;

//========================================================================================
// short for debugging interrupt
//========================================================================================
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

Real BetaPotential(const Real x1, const Real x2, const Real x3, const Real time) {
  Real x   = x1/pparam->Rbeta();
  Real phi = 1.5*pparam->Beta()*pparam->Temp()*std::log(1.0+SQR(x));
  return phi;
}

// BetaProfile, page 15, 8/10/22
Real BetaProfile(const Real r) {
  return pparam->Dens0()*pow(1.0+SQR(r/pparam->Rbeta()),-1.5*pparam->Beta());
}

Real ConstProfile(const Real r) {
  return pparam->Dens0();
}

#ifdef WIND
void GetDyDxSteadyStateAdbWind(const Real r, const AthenaArray<Real> &y, AthenaArray<Real> &dydx) {
  Real cs2 = std::pow(pparam->Gamma()*pparam->Temp()*(y(0)/pparam->Dwind(pparam->X1min())),pparam->Gamma()-1.0);
  Real m2  = SQR(y(1))/cs2;
  dydx(1)  = 2.0*y(1)/((m2-1.0)*r) - (m2/(m2-1.0))*pparam->GaccBeta(r)/y(1); // velocity
  dydx(0)  = -y(0)*(2.0/r+dydx(1)/y(1)); // density
  return;
}

void GetSteadyStateAdbVwind(const Real r, const Real dr, const AthenaArray<Real> &y0, AthenaArray<Real> &y) {
  GetDyDxSteadyStateAdbWind(r        ,y0,dydx_);
  for (int q=0; q<2; ++q) {
    k1_(q)    = dr*dydx_(q);
    ytemp_(q) = y0(q)+0.5*k1_(q);
  }
  GetDyDxSteadyStateAdbWind(r+0.5*dr,ytemp_,dydx_);
  for (int q=0; q<2; ++q) {
    k2_(q)    = dr*dydx_(q);
    ytemp_(q) = y0(q)+0.5*k2_(q);
  }
  GetDyDxSteadyStateAdbWind(r+0.5*dr,ytemp_,dydx_);
  for (int q=0; q<2; ++q) {
    k3_(q)    = dr*dydx_(q);
    ytemp_(q) = y0(q)+    k3_(q);
  }
  GetDyDxSteadyStateAdbWind(r+    dr,ytemp_,dydx_);
  for (int q=0; q<2; ++q) {
    k4_(q)    = dr*dydx_(q);
    y(q)      = y0(q) + (k1_(q)+2.0*(k2_(q)+k3_(q))+k4_(q))/6.0;
  }
  return;
}

Real WindProfile(const Real r) {
  int nstep = 10;
  Real r0, x1, d0, d1, dr;
  r0      = pparam->X1min();
  dr      = (r-r0)/((Real) (nstep-1));
  y0_(0)  = pparam->Dwind(r0);
  y0_(1)  = pparam->Vwind(); 
  for (int s=0; s<nstep; ++s) {
    GetSteadyStateAdbVwind(r0,dr,y0_,y1_);
    r0     = r0+dr;
    y0_(0) = y1_(0); 
    y0_(1) = y1_(1);
  }
  return y1_(0);
}
#endif


//========================================================================================
// Class CoolingFunction
// see pyyacc.py
//========================================================================================
class CoolingFunction {

  Real EnergySrc_Direct(const Real w[NHYDRO], const Real r, const Real dr, const Real dt);
  Real EnergySrc_WS09(const Real w[NHYDRO], const Real r, const Real dr, const Real dt);
  Real GainLoss_Slyz05(const Real dens, const Real temp, const Real vtot, const Real zmet);
  Real GainLoss_WS09(const Real dens, const Real temp, const Real vtot, const Real zmet);

  private:
    // For direct integration (Slyz 05). This is actually Slyz extended to low T including C++
    const int nk = 11;
    const Real tempk0[12] = {1e2, 1.1e4, 1.78e4, 4.5e4, 1e5, 2.584e5, 3.732e5, 1.5e6, 4.5e6, 1.1e7, 3.0e7, 1.0e10};
    const Real alphk0[11] = {2.2  , 8.0, -0.2, 2.0, -0.2, -3.0, -0.22, -1.6, 0.33, -0.5, 0.5};
    const Real loss0 = 1.0308655552913232e-28;
    Real lambk0[12];
    Real tempk[12];
    Real lambk[12];
    Real lratk[12];
    Real tratk[12];
    Real almok[11];
    Real alphk[11];
    Real yk[12];
    int mode               = 0; // 0: piece-wise power law, 1: WSS09 table
    const Real fac         = 2.167177868e+31; // conversion of cooling rate from cgs to ISM units.
    Real _mintemp = 0.0;
    Real _suppress= 1.0;
    // Wiersma+09. "0" is metal-free, "z" is only metals, "s" is solar.
    AthenaArray<Real> xion_0, xion_s, netcool_0, netcool_z, densarr, temparr; 
    Real log10dens, log10temp;
    int ndens=-1, ntemp=-1;

  public:

    // Townsend 09
    CoolingFunction(const Real suppress, const Real mintemp) {
      mode      = 0; // piece-wise power law
      _suppress = suppress;
      _mintemp  = mintemp;
      lambk0[0] = loss0*_suppress;
      for (int k=1; k<=nk; ++k)
        lambk0[k] = lambk0[k-1]*std::pow(tempk0[k]/tempk0[k-1],alphk0[k-1]);
      for (int k=0; k<nk; ++k) { 
        alphk[k] = alphk0[k];
      }
      for (int k=0; k<=nk; ++k) {
        tempk[k] = tempk0[k];
        lambk[k] = lambk0[k];
      }
      return;
    };

    //WS09
    CoolingFunction(const char* path) {
      mode  = 1; 
      ndens = 41;
      ntemp = 176;
      xion_s.NewAthenaArray(ntemp,ndens);
      xion_0.NewAthenaArray(ntemp,ndens);
      netcool_z.NewAthenaArray(ntemp,ndens);
      netcool_0.NewAthenaArray(ntemp,ndens);
      densarr.NewAthenaArray(ndens);
      temparr.NewAthenaArray(ntemp);
      // Read in information from HDF5 file.
      hid_t fapl, file, obj;
      herr_t err;
      hsize_t size;
      H5O_info_t obj_info;
      H5G_info_t grp_info;
      if ((fapl = H5Pcreate(H5P_FILE_ACCESS)) == H5I_INVALID_HID) {
        std::cout << "Failure creating file access" << std::endl;
        stop_this();
      }
      std::string file_name = path;
      if ((file = H5Fopen(file_name.c_str(),H5F_ACC_RDONLY,fapl)) == H5I_INVALID_HID) {
        std::cout << "Failure opening file " << file_name.c_str() << std::endl;
        stop_this();
      }
      H5Fget_filesize(file,&size);
      if (size < 0) {
        std::cout << "Failure determining file size" << std::endl;
        stop_this();
      }
      //std::cout << "File size = " << size << " bytes" << std::endl;

      // Block Solar. Need density array, temperature array, and ne/nH for solar metallicity.
      std::string object_name = "/Solar"; // Solar, Metal_free, Total_Metals
      if (H5Oget_info_by_name(file, object_name.c_str(), &obj_info, H5P_DEFAULT) < 0) {
        std::cout << "Failure getting info of path " << object_name.c_str() << std::endl;
        stop_this();
      }
      if (obj_info.type != H5O_TYPE_GROUP) {
       std::cout << "Failure finding Group object." << std::endl;
       stop_this;
      }
      if (H5Gget_info_by_name(file, object_name.c_str(), &grp_info, H5P_DEFAULT) < 0) {
        std::cout << "Cannot get info about group " << object_name.c_str() << std::endl;
        stop_this();
      }
      //std::cout << "Group info:" << std::endl;
      if (grp_info.storage_type != H5G_STORAGE_TYPE_SYMBOL_TABLE) {
        std::cout << "Storage type must be Symbol Table" << std::endl;
        stop_this;
      }
      //std::cout << "Link count in HDF5 group " << object_name.c_str() << ": " << grp_info.nlinks << std::endl;
      // check for fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        //std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
        delete link_name;
      }
      // read fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        //std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
        if (   (strcmp(link_name,"Temperature_bins")==0) 
            || (strcmp(link_name,"Hydrogen_density_bins")==0)
            || (strcmp(link_name,"Electron_density_over_n_h")==0)) { 
          std::string path_to_object = object_name.c_str();
          path_to_object += '/';
          path_to_object += link_name;
          obj = H5Oopen(file,path_to_object.c_str(),H5P_DEFAULT);
          err = H5Oget_info(obj, &obj_info);
          if (obj_info.type != H5O_TYPE_DATASET) {
            std::cout << "Expected Dataset." << std::endl;
            stop_this();
          }
          hid_t dataset = H5Dopen(file,path_to_object.c_str(),H5P_DEFAULT);
          hid_t dataspace  = H5Dget_space(dataset);
          //std::cout << "        Dataset " << path_to_object.c_str() << " has space " << dataspace << std::endl;
          int datarank     = H5Sget_simple_extent_ndims(dataspace);
          //std::cout << "        Dataset " << path_to_object.c_str() << " has rank "  << datarank << std::endl;
          hsize_t dims[H5S_MAX_RANK], maxdim[H5S_MAX_RANK];
          H5Sget_simple_extent_dims(dataspace, dims, maxdim);
          hsize_t nelts = 1;
          for (int ir=0; ir<datarank; ir++) {
            //std::cout << "            dim[" << ir << "] = " << dims[ir] << std::endl;
            nelts *= dims[ir];
          }
          hid_t datatype   = H5Dget_type(dataset);
          //std::cout << "        Dataset " << path_to_object.c_str() << " has type "  << datatype << std::endl;
          hid_t dataclass  = H5Tget_class(datatype);
          //std::cout << "        Datatype " << path_to_object.c_str() << " has class "  << dataclass << std::endl;
          if (dataclass != H5T_FLOAT) {
            std::cout << "Expected dataclass H5T_FLOAT." << std::endl;
            stop_this();
          }
          hid_t datanative = H5Tget_native_type(datatype, H5T_DIR_DEFAULT);
          size_t datasize  = H5Tget_size(datanative);
          //std::cout << "        Datatype " << path_to_object.c_str() << " has size " << datasize << std::endl;
          float *fbuf = new float[datasize*nelts];
          H5Dread(dataset, datanative, H5S_ALL, H5S_ALL, H5P_DEFAULT, fbuf);
          //for (int elt=0; elt<nelts; elt++) {
          //  std::cout << "        Dataset " << path_to_object.c_str() << " elt " << elt << " = " << fbuf[elt] << std::endl;
          //}
          if (strcmp(link_name,"Temperature_bins")==0) {
            for (int ielt=0; ielt < nelts; ielt++) {
              temparr(ielt) = (Real) fbuf[ielt];
            }
          } else if (strcmp(link_name,"Hydrogen_density_bins")==0) {
            for (int ielt=0; ielt < nelts; ielt++) {
              densarr(ielt) = (Real) fbuf[ielt];
            }
          } else if (strcmp(link_name,"Electron_density_over_n_h")==0) {
            for (int ielt=0; ielt < nelts; ielt++) { 
              xion_s(ielt / ndens, ielt % ndens) = (Real) fbuf[ielt];
            }
          }
          delete fbuf;
          H5Tclose(datanative);
          H5Tclose(datatype);
          H5Sclose(dataspace);
          H5Dclose(dataset);
          H5Dclose(obj);
        }
        delete link_name;
      }

      // Block Metal_free. Need Net_Cooling and Electron_density_over_n_h, both at index 1 in first dimension (Y=0.248).
      object_name = "/Metal_free"; // Solar, Metal_free, Total_Metals
      if (H5Oget_info_by_name(file, object_name.c_str(), &obj_info, H5P_DEFAULT) < 0) {
        std::cout << "Failure getting info of path " << object_name.c_str() << std::endl;
        stop_this();
      }
      if (obj_info.type != H5O_TYPE_GROUP) {
       std::cout << "Failure finding Group object." << std::endl;
       stop_this;
      }
      if (H5Gget_info_by_name(file, object_name.c_str(), &grp_info, H5P_DEFAULT) < 0) {
        std::cout << "Cannot get info about group " << object_name.c_str() << std::endl;
        stop_this();
      }
      //std::cout << "Group info:" << std::endl;
      if (grp_info.storage_type != H5G_STORAGE_TYPE_SYMBOL_TABLE) {
        std::cout << "Storage type must be Symbol Table" << std::endl;
        stop_this;
      }
      //std::cout << "Link count in HDF5 group " << object_name.c_str() << ": " << grp_info.nlinks << std::endl;
      // check for fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        //std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
        delete link_name;
      }
      // read fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        //std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
        if (   (strcmp(link_name,"Net_Cooling")==0)
            || (strcmp(link_name,"Electron_density_over_n_h")==0)) {
          std::string path_to_object = object_name.c_str();
          path_to_object += '/';
          path_to_object += link_name;
          obj = H5Oopen(file,path_to_object.c_str(),H5P_DEFAULT);
          err = H5Oget_info(obj, &obj_info);
          if (obj_info.type != H5O_TYPE_DATASET) {
            std::cout << "Expected Dataset." << std::endl;
            stop_this();
          }
          hid_t dataset = H5Dopen(file,path_to_object.c_str(),H5P_DEFAULT);
          hid_t dataspace  = H5Dget_space(dataset);
          //std::cout << "        Dataset " << path_to_object.c_str() << " has space " << dataspace << std::endl;
          int datarank     = H5Sget_simple_extent_ndims(dataspace);
          //std::cout << "        Dataset " << path_to_object.c_str() << " has rank "  << datarank << std::endl;
          hsize_t dims[H5S_MAX_RANK], maxdim[H5S_MAX_RANK];
          H5Sget_simple_extent_dims(dataspace, dims, maxdim);
          hsize_t nelts = 1;
          for (int ir=0; ir<datarank; ir++) {
            //std::cout << "            dim[" << ir << "] = " << dims[ir] << std::endl;
            nelts *= dims[ir];
          }
          hid_t datatype   = H5Dget_type(dataset);
          //std::cout << "        Dataset " << path_to_object.c_str() << " has type "  << datatype << std::endl;
          hid_t dataclass  = H5Tget_class(datatype);
          //std::cout << "        Datatype " << path_to_object.c_str() << " has class "  << dataclass << std::endl;
          if (dataclass != H5T_FLOAT) {
            std::cout << "Expected dataclass H5T_FLOAT." << std::endl;
            stop_this();
          }
          hid_t datanative = H5Tget_native_type(datatype, H5T_DIR_DEFAULT);
          size_t datasize  = H5Tget_size(datanative);
          //std::cout << "        Datatype " << path_to_object.c_str() << " has size " << datasize << std::endl;
          float *fbuf = new float[datasize*nelts];
          H5Dread(dataset, datanative, H5S_ALL, H5S_ALL, H5P_DEFAULT, fbuf);
          //for (int elt=0; elt<nelts; elt++) {
          //  std::cout << "        Dataset " << path_to_object.c_str() << " elt " << elt << " = " << fbuf[elt] << std::endl;
          //}
          if (strcmp(link_name,"Net_Cooling")==0) {
            for (int ielt=0; ielt < nelts/7; ielt++) { // not nice, but read a 7th of the data set (just one He fraction)
              netcool_0(ielt / ndens, ielt % ndens) = (Real) fbuf[ntemp*ndens + ielt]; // index 1 in He fraction is offset by 1 (n,T) block.
            }
          } else if (strcmp(link_name,"Electron_density_over_n_h")==0) {
            for (int ielt=0; ielt < nelts/7; ielt++) {
              xion_0(ielt / ndens, ielt % ndens) = (Real) fbuf[ntemp*ndens + ielt];
            }
          }
          delete fbuf;
          H5Tclose(datanative);
          H5Tclose(datatype);
          H5Sclose(dataspace);
          H5Dclose(dataset);
          H5Dclose(dataclass);
          H5Dclose(obj);
        }
        delete link_name;
      }

      // Block Total_Metals. Need Net_cooling.
      object_name = "/Total_Metals"; // Solar, Metal_free, Total_Metals
      if (H5Oget_info_by_name(file, object_name.c_str(), &obj_info, H5P_DEFAULT) < 0) {
        std::cout << "Failure getting info of path " << object_name.c_str() << std::endl;
        stop_this();
      }
      if (obj_info.type != H5O_TYPE_GROUP) {
       std::cout << "Failure finding Group object." << std::endl;
       stop_this;
      }
      if (H5Gget_info_by_name(file, object_name.c_str(), &grp_info, H5P_DEFAULT) < 0) {
        std::cout << "Cannot get info about group " << object_name.c_str() << std::endl;
        stop_this();
      }
      //std::cout << "Group info:" << std::endl;
      if (grp_info.storage_type != H5G_STORAGE_TYPE_SYMBOL_TABLE) {
        std::cout << "Storage type must be Symbol Table" << std::endl;
        stop_this;
      }
      //std::cout << "Link count in HDF5 group " << object_name.c_str() << ": " << grp_info.nlinks << std::endl;
      // check for fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        //std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
        delete link_name;
      }
      // read fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        //std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
        if (strcmp(link_name,"Net_cooling")==0) {
          std::string path_to_object = object_name.c_str();
          path_to_object += '/';
          path_to_object += link_name;
          obj = H5Oopen(file,path_to_object.c_str(),H5P_DEFAULT);
          err = H5Oget_info(obj, &obj_info);
          if (obj_info.type != H5O_TYPE_DATASET) {
            std::cout << "Expected Dataset." << std::endl;
            stop_this();
          }
          hid_t dataset = H5Dopen(file,path_to_object.c_str(),H5P_DEFAULT);
          hid_t dataspace  = H5Dget_space(dataset);
          //std::cout << "        Dataset " << path_to_object.c_str() << " has space " << dataspace << std::endl;
          int datarank     = H5Sget_simple_extent_ndims(dataspace);
          //std::cout << "        Dataset " << path_to_object.c_str() << " has rank "  << datarank << std::endl;
          hsize_t dims[H5S_MAX_RANK], maxdim[H5S_MAX_RANK];
          H5Sget_simple_extent_dims(dataspace, dims, maxdim);
          hsize_t nelts = 1;
          for (int ir=0; ir<datarank; ir++) {
            //std::cout << "            dim[" << ir << "] = " << dims[ir] << std::endl;
            nelts *= dims[ir];
          }
          hid_t datatype   = H5Dget_type(dataset);
          //std::cout << "        Dataset " << path_to_object.c_str() << " has type "  << datatype << std::endl;
          hid_t dataclass  = H5Tget_class(datatype);
          //std::cout << "        Datatype " << path_to_object.c_str() << " has class "  << dataclass << std::endl;
          if (dataclass != H5T_FLOAT) {
            std::cout << "Expected dataclass H5T_FLOAT." << std::endl;
            stop_this();
          }
          hid_t datanative = H5Tget_native_type(datatype, H5T_DIR_DEFAULT);
          size_t datasize  = H5Tget_size(datanative);
          //std::cout << "        Datatype " << path_to_object.c_str() << " has size " << datasize << std::endl;
          //float *fbuf = (float *) malloc((size_t)(elements*datasize));
          float *fbuf = new float[datasize*nelts];
          H5Dread(dataset, datanative, H5S_ALL, H5S_ALL, H5P_DEFAULT, fbuf);
          //for (int elt=0; elt<nelts; elt++) {
          //  std::cout << "        Dataset " << path_to_object.c_str() << " elt " << elt << " = " << fbuf[elt] << std::endl;
          //}
          if (strcmp(link_name,"Net_cooling")==0) {
            for (int ielt=0; ielt < nelts; ielt++) {
              netcool_z(ielt / ndens, ielt % ndens) = (Real) fbuf[ielt];
            }
          }
          delete fbuf;
          H5Tclose(datanative);
          H5Tclose(datatype);
          H5Sclose(dataspace);
          H5Dclose(dataset);
          H5Dclose(obj);
        }
        delete link_name;
      }
      H5Fclose(file);
      H5Pclose(fapl);

      log10dens = std::log10(densarr(ndens-1)/densarr(0));
      log10temp = std::log10(temparr(ntemp-1)/temparr(0));
      if (Globals::my_rank == 0) {
        std::cout << "[CoolingFunction]: min(dens) = " << std::scientific << std::setw(11) << std::setprecision(3) << densarr(0) 
                  <<                   " max(dens) = " << std::scientific << std::setw(11) << std::setprecision(3) << densarr(ndens-1)
                  <<                   " min(temp) = " << std::scientific << std::setw(11) << std::setprecision(3) << temparr(0)
                  <<                   " max(temp) = " << std::scientific << std::setw(11) << std::setprecision(3) << temparr(ntemp-1)
                  << std::endl;
      }
    }
 
    ~CoolingFunction () {
      if (mode == 0) { 
        return;
      } else if (mode == 1) {
        xion_s.DeleteAthenaArray(); 
        xion_0.DeleteAthenaArray();
        netcool_z.DeleteAthenaArray();
        netcool_0.DeleteAthenaArray();
        densarr.DeleteAthenaArray();
        temparr.DeleteAthenaArray();
        return;
      }
    };

    Real EnergySrc(const Real w[NHYDRO], const Real r, const Real dr, const Real dt) {
      if (mode == 0) {
        return EnergySrc_Direct(w, r, dr, dt);
      } else if (mode == 1) {
        return EnergySrc_WS09(w, r, dr, dt);
      }
    }

    Real GainLoss(Real dens, const Real temp, const Real vtot, const Real zmet) {
      if (mode == 0) {
        return GainLoss_Slyz05(dens, temp, vtot, zmet);
      } else if (mode == 1) {
        return GainLoss_WS09(dens, temp, vtot, zmet);
      }
    }

    // This breaks the structure: does not work for WS09
    Real CoolingLength(const Real w[NHYDRO], const Real r, const Real dt) {
      //Real dener, ener, temp, csound, mintemp, mintemp0, alph0, zmet;
      //dener    = std::fabs(EnergySrc(w, r, dt));
      //zmet     = w[NHYDRO-NSCALARS];
      //alph0    = alphk0[0]/(zmet+1e-10);
      //mintemp  = 1.1e4*std::pow(lambk0[0]/lambk0[1],1.0/alph0);
      //mintemp0 = _mintemp <= 0.0 ? mintemp : _mintemp;
      //temp     = w[IPR]/w[IDN] < mintemp0 ? mintemp0 : w[IPR]/w[IDN];
      //ener     = w[IDN]*temp/gm1;
      //csound   = std::sqrt(temp);
      //return ener*csound/dener;
      Real temp   = w[IPR]/w[IDN];
      Real lambda = GetLambda(w, r, dt);
      Real csound = std::sqrt(temp);
      return temp*csound/(w[IDN]*lambda*fac);
    }

    Real GetLambda(const Real w[NHYDRO], const Real r, const Real dt) {
      Real lambda=0.0;
      Real temp = w[IPR]/w[IDN];
      Real zmet = w[NHYDRO-NSCALARS];
      alphk[0]  = alphk0[0]/(zmet+1e-10);
      alphk[2]  = alphk0[2]-1.5*(1.0-zmet);
      alphk[3]  = alphk0[3]-0.3*(1.0-zmet);
      alphk[4]  = alphk0[4]-(1.0-zmet)/(1.0+zmet);
      alphk[7]  = alphk0[7]+1.4*(1.0-zmet);
      alphk[9]  = alphk0[9]+SQR(1.0-zmet);
      tempk[0]  = 1.1e4*std::pow(lambk0[0]/lambk0[1],1.0/alphk[0]);
      lambk[0] = loss0*_suppress;
      for (int k=1; k<=nk; ++k) {
        lambk[k]  = lambk[k-1]*std::pow(tempk[k]/tempk[k-1],alphk[k-1]);
        almok[k-1]= alphk[k-1]-1.0;
      }
      if (temp < tempk[0]) {
        lambda = 0.0;
      } else {
        int k=0;
        while ((k < nk) && (tempk[k] <= temp)) k++;
        lambda = lambk[k-1]*std::pow(temp/tempk[k-1],alphk[k-1]);
      }
      return lambda;
    }

}; // class CoolingFunction

typedef Real (CoolingFunction::*GainLossFunc_t) (Real dens, const Real temp, const Real vtot, const Real zmet);
//using GainLossFunc_t = Real (CoolingFunction::*)(Real dens, const Real temp, const Real vtot, const Real zmet);

//========================================================================================
//! \fn void GainLoss_Slyz05(...)
//  \brief Heating and cooling (dedt in erg/s) for Slyz 05
//========================================================================================
Real CoolingFunction::GainLoss_Slyz05(const Real dens, const Real temp, const Real vtot, const Real zmet) {
  Real dedt, lambda;
  if (temp < tempk[0]) {
    lambda = 0.0;
  } else {
    int i=0;
    while ((i < nk) && (tempk[i] <= temp)) i++;
    lambda = lambk[i-1]*std::pow(temp-tempk[i-1],alphk[i-1]);
  }
  dedt   = -dens*lambda*fac;
  return dedt;
}

//========================================================================================
//! \fn void GainLoss_WS09(...)
//  \brief Heating and cooling (dedt in erg/s) for WS09
//========================================================================================
Real CoolingFunction::GainLoss_WS09(const Real dens, const Real temp, const Real vtot, const Real zmet) {
  int idens,itemp,idens1,itemp1;
  Real dedt,wd,wt,owd,owt,w1,w2,w3,w4,lambda,lambda_0,lambda_z,xe_0,xe_s;
  Real dd = dens;
  Real tt = temp;
  // need to interpolate in density and temperature first.
  idens = (int) std::floor(ndens*std::log10(dens/densarr(0))/log10dens);
  if (idens < 0) {
    idens = 0;
    dd    = densarr(0);
  }
  if (idens >= ndens-1) {
    idens = ndens-2; // enforce last interpolation bin, with wd = 1.0
    dd    = densarr(ndens-1);
  }
  idens1= idens+1;
  wd    = (dd-densarr(idens))/(densarr(idens1)-densarr(idens));
  owd   = 1.0-wd;
  itemp = (int) std::floor(ntemp*std::log10(temp/temparr(0))/log10temp);
  if (itemp < 0) {
    itemp = 0;
    tt    = temparr(0);
  }
  if (itemp >= ntemp-1) {
    itemp = ntemp-2; // enforce last interpolation bin, with wt = 1.0
    tt    = temparr(ntemp-1);
  }
  itemp1= itemp+1;
  wt    = (tt-temparr(itemp))/(temparr(itemp1)-temparr(itemp));
  owt   = 1.0-wt;
  w1    = owt*owd;
  w2    = owt* wd;
  w3    =  wt*owd;
  w4    =  wt* wd;
  // calculate lambda. 
  lambda_0 =   netcool_0(itemp ,idens )*w1
             + netcool_0(itemp ,idens1)*w2
             + netcool_0(itemp1,idens )*w3
             + netcool_0(itemp1,idens1)*w4;
  lambda_z =   netcool_z(itemp ,idens )*w1
             + netcool_z(itemp ,idens1)*w2
             + netcool_z(itemp1,idens )*w3
             + netcool_z(itemp1,idens1)*w4;
  xe_0 =       xion_0   (itemp ,idens )*w1
             + xion_0   (itemp ,idens1)*w2
             + xion_0   (itemp1,idens )*w3
             + xion_0   (itemp1,idens1)*w4;
  xe_s =       xion_s   (itemp ,idens )*w1
             + xion_s   (itemp ,idens1)*w2
             + xion_s   (itemp1,idens )*w3
             + xion_s   (itemp1,idens1)*w4;
  lambda = lambda_0 + lambda_z* (xe_0/xe_s)*zmet;
  // Cooling is positive in WSS09, and turbulent heating rate in code units
  // vtot should contain sound speed to prevent zeroing of temperature for quiescent flow.
  dedt   = -dd*lambda*fac + turbcool*dd*SQR(vtot)*vtot/lengthcool;
  return dedt;
}

//========================================================================================
//! \fn void EnergySrc_Slyz05(...)
//  \brief Heating and cooling for user-defined cooling function
//  Implicit solution of ODE for temperature change (see RootFunc)
//  (See HeatCoolTimeStep).
//========================================================================================
Real CoolingFunction::EnergySrc_Direct(const Real w[NHYDRO], const Real r, const Real dr, const Real dt) {
  Real mintemp0, pc, pr, dw, dens, cwind, temp0, temp1, vtot, zmet, xwh, xwl, wghthalo,wghtwind, dener; // general cooling variables
  wghthalo = 1.0; // Weights for suppressing cooling in halo and wind.
  wghtwind = 1.0; // 1.0 means 100% cooling, 0.0 no cooling
  dens     = w[IDN];
  if (DUAL_ENERGY) {
    pc    = w[IGE];
  } else {
    pc    = w[IPR];
  }
  temp0 = pc/dens;
  zmet = w[NHYDRO-NSCALARS]; // metallicity
  cwind= w[NHYDRO-1]; // wind tracer
  
  alphk[0] = alphk0[0]/(zmet+1e-10);
  alphk[2] = alphk0[2]-1.5*(1.0-zmet);
  alphk[3] = alphk0[3]-0.3*(1.0-zmet);
  alphk[4] = alphk0[4]-(1.0-zmet)/(1.0+zmet);
  alphk[7] = alphk0[7]+1.4*(1.0-zmet);
  alphk[9] = alphk0[9]+SQR(1.0-zmet);
  tempk[0] = 1.1e4*std::pow(lambk0[0]/lambk0[1],1.0/alphk[0]);
  mintemp0 = _mintemp <= 0.0 ? tempk[0] : _mintemp; 

  if (temp0 < tempk[0]) { //tempk[0]
    temp1    = tempk[0]; // put temperature back to minimum of cooling curve. 
    dener    = dens*(temp1-temp0)/gm1;
  } else {
    vtot     = std::sqrt(  SQR(w[IVX])+SQR(w[IVY])+SQR(w[IVZ])
                         + (temp0 < 1.0e4 ? 1.0e4 : temp0));
    pr       = ProfileFunc(r)*pparam->Temp();
    // Reduces cooling if pressure within 10% of profile pressure
    // page 120. 
    Real ftanhp = 0.3;
    xwh      = (pc/pr-(1.0+ftanhp))/0.03;
    xwl      = (pc/pr-(1.0-ftanhp))/0.03;
    wghthalo = 1.0+0.5*(std::tanh(xwh)-std::tanh(xwl));
#ifdef WIND
    Real ftanhn = 0.3;
    dw       = WindProfile(r);
    xwh      = (dens/dw-(1.0+ftanhn))/0.01;
    xwl      = (dens/dw-(1.0-ftanhn))/0.01;
    wghtwind = 1.0+0.5*(std::tanh(xwh)-std::tanh(xwl));
    wghtwind = (cwind > 0.99) ? wghtwind : 1.0; // apply only to wind region
#endif 
    // Townsend 09
    // step 0: calculate Tks, Yks
    lambk[0] = loss0*_suppress;
    for (int k=1; k<=nk; ++k) {
      lambk[k]  = lambk[k-1]*std::pow(tempk[k]/tempk[k-1],alphk[k-1]);
      almok[k-1]= alphk[k-1]-1.0;
    }
#pragma omp simd
    for (int k=0; k<=nk; ++k) {
      lratk[k] = lambk[k]/lambk[nk];
      tratk[k] = tempk[k]/tempk[nk];
    }
    yk[nk] = 0.0;
    for (int k=nk-1; k>=0; --k)
      yk[k] = yk[k+1] + (1.0/almok[k]) * tratk[k] / lratk[k]
                      * (1.0-std::pow(tempk[k]/tempk[k+1],almok[k]));
    // step 1: get k0 such that Tk<=T<Tk+1
    int k0=0;
    if (temp0 > tempk[nk]) {
       k0=nk;
    } else { 
      while (temp0 >= tempk[k0]) k0++;
    }
    k0--;
    // step 2: get y0 
    Real ttrat  = temp0/tempk[k0];
    Real y0     = yk[k0] - (1.0/almok[k0]) * tratk[k0] / lratk[k0]  
                         * (1.0-std::pow(ttrat,-almok[k0]));
    // step 3: integrate, i.e. evolve y in time
    Real lamb0  = lambk[k0]*std::pow(ttrat,alphk[k0]);
    Real tcool  = temp0/(dens*lamb0*fac);
    Real y1     = y0 + (temp0/tempk[nk]) * (lambk[nk]/lamb0) * (dt/tcool);
    // step 4: get k1 such that yk >= y > yk+1
    int k1=0;
    if (y1 > yk[0]) {
      k1=1;
    } else {
      while (y1 < yk[k1]) k1++;
    }
    k1--;
    temp1 = tempk[k1]*std::pow(1.0+almok[k1]*lratk[k1]/tratk[k1]*(y1-yk[k1]),-1.0/almok[k1]); 
    //if (isnan(temp1) || (temp1 < tempk[0])) temp1 = tempk[0];
    temp1 = (isnan(temp1) || (temp1 < tempk[0])) ? tempk[0] : temp1;
    temp1 = temp1 < mintemp0 ? mintemp0 : temp1;
    // just an experiment: iPad 11/19/24: use temperature limit depending on resolution
    // But does not work if the wind is allowed to cool. 
    //Real tempdr = std::pow(dens*GetLambda(w,r,dt)*fac*dr,2.0/3.0);
    //temp1 = temp1 < tempdr ? tempdr : temp1;
    //temp1 *= (1.0+0.05*(1.0+std::tanh((std::log10(dens)+1.0)/0.1)));
    dener = wghthalo*wghtwind * dens*(temp1-temp0)/gm1;
  }
  // At this point we have the dener coming from cooling curve or temperature floor.
  // Check whether we try to keep wind isothermal.
  //Real deneriso = dens*(pparam->Temp()-temp0)/gm1;
  //Real deneriso = zmet*SQR(dens)*gammaheat0*fac*dt/gm1;
  //dener = ((Real) iwiso) * (cwind*deneriso + (1.0-cwind)*dener) + (1.0-(Real) iwiso) * dener;
  //dener += zmet*SQR(dens)*gammaheat0*fac*dt/gm1;
  //dener += 0.5*(1.0+std::tanh((dens-1.0)/0.1))*dens*gammaheat0*fac*dt/gm1;
  return dener;
}

//========================================================================================
//! \fn void EnergySrc_WS09(...)
//  \brief Heating and cooling for user-defined cooling function
//  Implicit solution of ODE for temperature change (see RootFunc)
//  (See HeatCoolTimeStep).
//========================================================================================
Real CoolingFunction::EnergySrc_WS09(const Real w[NHYDRO], const Real r, const Real dr, const Real dt) {

  auto lRootFunc = [](Real dens, Real temp0, Real temp1, Real vtot, Real zmet, Real dt, CoolingFunction* pc) -> Real {
    //show_type_abort(glf);
    return temp0 + dt*gm1*pc->GainLoss(dens,temp1,vtot,zmet) - temp1;
  };

  Real pc, pr, dens, temp0, temp1, temp2, vtot, zmet, xwh, xwl, wght, dener; // general cooling variables
  Real rf, sig, fac, T[3], L[2];
  int nit;
  dens = w[IDN];
  if (DUAL_ENERGY) {
    pc    = w[IGE];
  } else {
    pc    = w[IPR];
  }
  temp0 = pc/dens;
  if (temp0 < temparr(0)) {
    temp2 = temparr(0);
  } else {
    vtot = std::sqrt(  SQR(w[IVX])+SQR(w[IVY])+SQR(w[IVZ])
                     + (temp0 < 1.0e4 ? 1.0e4 : temp0));
    zmet = w[NHYDRO-NSCALARS];
    pr   = ProfileFunc(r)*pparam->Temp();
    // Reduces cooling if pressure within 10% of profile pressure
    xwh  = 3e1*(pc-1.3*pr)/pr;
    xwl  = 3e1*(pc-0.7*pr)/pr;
    wght = 0.5*(1.0+std::tanh(xwh)) + 0.5*(1.0-std::tanh(xwl));
    // Find the temperature change that corresponds to the amount of energy change based on cooling curve at given dt.
    // Cannot be parallelized because of function calls.
    temp1 = temp0;
    rf    = lRootFunc(dens,temp0,temp1,vtot,zmet,dt,this);
    sig   = (Real) ((rf > 0) - (rf < 0));
    fac   = 1.0 + sig*0.1;
    while (rf*lRootFunc(dens,temp0,temp1,vtot,zmet,dt,this) > 0)
      temp1 *= fac;
    nit = (int) (log(std::fabs((temp1-temp0)/(temp1+temp0))/tolcool)/log(2.0));
    if (GainLoss(dens,temp0,vtot,zmet) == 0.0) return temp0; // Nothing to do for thermal equilibrium
    // Otherwise, temp1 and temp0 bracket the temperature down to which we should integrate.
    T[0]         = temp0;
    T[1]         = temp1;
    T[2]         = 0.5*(T[0]+T[1]);
    L[0]         = lRootFunc(dens,temp0,T[0],vtot,zmet,dt,this);
    L[1]         = lRootFunc(dens,temp0,T[2],vtot,zmet,dt,this);
    for (int i=0; i<nit; i++) {
      int w = (L[0]*L[1] < 0); // 0 if >0, 1 if <= 0
      T[w]  = T[2];
      L[w]  = L[1];
      T[2]  = 0.5*(T[0]+T[1]);
      L[1]  = lRootFunc(dens,temp0,T[2],vtot,zmet,dt,this);
    }
    temp2 = T[2];
  }
  dener = wght * dens*(temp2-temp0)/gm1;
  return dener;
}



//========================================================================================
//! \fn void heatcool(...)
//  \brief Heating and cooling for user-defined cooling function
//  Implicit solution of ODE for temperature change (see RootFunc)
//  (See HeatCoolTimeStep).
//========================================================================================
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  Real w[NHYDRO];

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real dr= pmb->pcoord->dx1f(i);
#pragma omp simd
        for (int n=0; n<NHYDRO; ++n) 
          w[n] = prim(n,k,j,i);
        Real dener = pcoolfunc->EnergySrc(w,r,dr,dt);
        cons(IEN,k,j,i) += dener;
        if (DUAL_ENERGY)
          cons(IIE,k,j,i) += dener;
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void HeatCoolTimeStep(...)
//  \brief Calculates cooling timestep and sends it to new_blockdt
//    Parameter coolsafe works as "CFL" (safety) factor for cooling: Timestep 
//    reduced to ensure only coolsafe*100 % of internal energy is removed. 
//========================================================================================
Real HeatCoolTimeStep(MeshBlock *pmb)
{
  AthenaArray<Real> prim;
  prim.InitWithShallowCopy(pmb->phydro->w);
  Real w[NHYDRO];
  Real dtcool = HUGE_NUMBER;// allows it to grow
  Real temp;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real dr= pmb->pcoord->dx1f(i);
        for (int n=0; n<NHYDRO; ++n)
          w[n] = prim(n,k,j,i);
        if (DUAL_ENERGY) {
          Real tt = w[IIE]/w[IDN];
          temp = (tt < 1e3 ? 1e3 : tt); 
        } else {
          Real tt = w[IPR]/w[IDN];
          temp = (tt < 1e3 ? 1e3 : tt);
        }
        Real dener  = pcoolfunc->EnergySrc(w,r,dr,pmb->pmy_mesh->dt);
        Real dttemp = coolsafe*(w[IDN]*temp/gm1)/(std::fabs(dener)+1e-60);
        if (TIMESTEPINFO_ENABLED) {
          if (dttemp < dtcool) {
            pmb->all_min_dts(8)   = dttemp;
            pmb->all_min_loc(8,0) = pmb->pcoord->x1v(i);
            pmb->all_min_loc(8,1) = pmb->pcoord->x2v(j);
            pmb->all_min_loc(8,2) = pmb->pcoord->x3v(k);
            pmb->all_min_ind(8,0) = i;
            pmb->all_min_ind(8,1) = j;
            pmb->all_min_ind(8,2) = k;
          }
        }
        dtcool   = std::min(dtcool,dttemp); 
      }
    }
  }
  return dtcool;
}

//#############################################################
// Conduction
//=============================================================
void SpitzerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        phdif->kappa(ISO,k,j,i) = phdif->kappa_iso * std::pow(1e-8*prim(IPR,k,j,i)/prim(IDN,k,j,i),2.5); // Spitzer conductivity
      }
    }
  }
  return;
}

//========================================================================================
//! \fn int RefinementCondition(MeshBlock *pmb)
//  \brief Refinement condition on the cooling length;
//  Only 1D for now.

int RefinementCondition(MeshBlock *pmb)
{
  Real w[NHYDRO];
  Real lodx, dx, r, dm, dm0, minlodx = HUGE_NUMBER, maxdm = TINY_NUMBER; 
  Real dt = pmb->pmy_mesh->dt;
  int k=pmb->ks; 
  int j=pmb->js;
  for(int i=pmb->is; i<=pmb->ie; i++) {
#pragma omp simd
    for (int n=0; n<NHYDRO; ++n) 
      w[n] = pmb->phydro->w(n,k,j,i);
    dx      = pmb->pcoord->dx1f(i);
    r       = pmb->pcoord->x1v(i);
    lodx    = pcoolfunc->CoolingLength(w, r, dt)/dx;
    minlodx = std::min(lodx,minlodx); 
  }
  if (minlodx < lodxref) return 1;   // lodx <= 2*lodxderef. refinement means division by 2,
  if (minlodx > lodxderef) return -1; // so these conditions should be exclusive (page 131).

  return 0;
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
 
  //if (dir != gridData(1)){
  //  retval = 0.0;
  //} else if (xf<=gridData(0)){
  //  retval = 0.0;
  //} else if (xf > gridData(0)){ 
  //  if (gridData(2)==0.0) retval = 0.0;
  //  else retval = gridData(2) * (xf-gridData(0))/(gridData(3)-gridData(0));
  //} 
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
      //fprintf(stdout,"[WallVel]: i=%3i xf=%17.9e x=%17.9e vexp=%17.9e rmin=%17.9e rmax=%17.9e retval=%17.9e\n",i,xf,x,gridData(2),gridData(0),gridData(3),retval);

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
//========================================================================================
void UpdateGridData(Mesh *pm) {

  pm->GridData(3) = pm->mesh_size.x1max;
  MeshBlock *pmb = pm->pblock;

  if (itrack == 0) { // analytic fit, see page 121 and findexpansion.py 
    pm->GridData(2) = vtrack0;
    return;
  } else if (itrack == 1) {
    Real myVel = 0.0;
    Real cellsize = pm->mesh_size.x1max/pm->mesh_size.nx1;
    Real posUp = 0.5*pm->mesh_size.x1max;
    Real posLow = 0.4*pm->mesh_size.x1max;// - 15.0*cellsize;
    Real velAve=0.0, vol=0.0;
    while (pmb != NULL) {
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            Real pos  = pmb->pcoord->x1v(i);
            Real dvol = pmb->pcoord->GetCellVolume(k,j,i);
            Real w = (Real) ((pos<=posUp) && (pos>=posLow));
            Real d  = pmb->phydro->u(IDN,k,j,i);
            Real c2 = pmb->phydro->u(NHYDRO-NSCALARS+2,k,j,i)/d;
            velAve += pmb->phydro->u(IM1,k,j,i)/d * dvol * c2 * w;
            vol    += dvol * c2 * w;
          }
        }
      }
      pmb = pmb->next;
    }
#ifdef MPI_PARALLEL
    Real arr[2];
    arr[0] = velAve;
    arr[1] = vol;
    MPI_Allreduce(MPI_IN_PLACE,&arr,2,MPI_ATHENA_REAL,MPI_SUM,
                  MPI_COMM_WORLD);
    velAve = arr[0];
    vol    = arr[1];
#endif
    velAve = velAve/(vol+1e-30);
    myVel = boost*velAve;
    if ((myVel <=0.0)) {
      myVel = 0.0;
    }
    // different attempt:
    //Real tref = 0.05*xMax/pparam->Vwind();
    //myVel = std::min(pm->time*pparam->Vwind()/tref,pparam->Vwind());
    pm->GridData(2) = myVel;
    //fprintf(stdout,"[UpdateGridData]: vel = %17.9e xmax = %17.9e\n",pm->GridData(2),pm->GridData(3));
    return;
  } else { // itrack > 1
    Real vtrack = 0.0;
    Real gamma  = pmb->peos->GetGamma();
    int ntr=0;
    AthenaArray<Real> weight, quant, radius;
    Real totweight = 0.0, totquant = 0.0, totradius = 0.0;
    while (pmb != NULL) {
      int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
      if (pm->dimension == 1) {
        weight.NewAthenaArray(1,1,pmb->block_size.nx1+2*NGHOST);
        quant.NewAthenaArray(1,1,pmb->block_size.nx1+2*NGHOST);
        radius.NewAthenaArray(1,1,pmb->block_size.nx1+2*NGHOST);
      } else if (pm->dimension == 2) {
        weight.NewAthenaArray(1,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        quant.NewAthenaArray(1,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        radius.NewAthenaArray(1,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
      } else {
        weight.NewAthenaArray(pmb->block_size.nx3+2*NGHOST,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        quant.NewAthenaArray(pmb->block_size.nx3+2*NGHOST,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        radius.NewAthenaArray(pmb->block_size.nx3+2*NGHOST,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
      }
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            Real rad = pmb->pcoord->x1v(i);
            quant(k,j,i)  = rad;
            radius(k,j,i) = rad;
          }
        }
      }
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
      if (pm->dimension == 1) {
#pragma omp simd
        for (int i=is+1; i<=ie-1; ++i) {
          Real gr    =   (weight(ks ,js, i+1)-weight(ks ,js, i-1))
                        /(pmb->pcoord->x1v(i+1)-pmb->pcoord->x1v(i-1));
          Real q     = quant(ks,js,i);
          Real r     = radius(ks,js,i);
          Real w     = std::fabs(gr);
          q         *= w;
          r         *= w;
          totquant  += q;
          totweight += w;
          totradius += r;
        }
      } else if (pm->dimension == 2) {
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
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) { // only radial gradient here, hence use whole j range
#pragma omp simd
            for (int i=is+1; i<=ie-1; ++i) {
              Real gr    =   (weight(k,j, i+1)-weight(k,j, i-1))
                            /(pmb->pcoord->x1v(i+1)-pmb->pcoord->x1v(i-1));
              Real q     = quant(k,j,i);
              Real r     = radius(k,j,i);
              Real w     = std::fabs(gr);
              q         *= w;
              r         *= w;
              totquant  += q;
              totweight += w;
              totradius += r;
            }
          }
        }
      }
      weight.DeleteAthenaArray();
      quant.DeleteAthenaArray();
      radius.DeleteAthenaArray();
      pmb = pmb->next;
    }  // while (pmb != NULL)

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
    vtrack = (vtrack <= 0.0) ? 0.0 : vtrack; // enforce expansion
    vtrack *= boost;
    //if (Globals::my_rank == 0)
    //  fprintf(stdout,"[UpdateGrid]: time=%13.5e vtrack=%13.5e xmax =%13.5e\n",pm->time,vtrack,pm->GridData(3));
    pm->GridData(2) = vtrack;
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

  //if (COORDINATE_SYSTEM != "spherical_polar") {
  //  std::stringstream msg;
  //  msg << "### FATAL ERROR in galpause.cpp: coordinate system must be spherical-polar" << std::endl;
  //  throw std::runtime_error(msg.str().c_str());
  //}

  //if (!DUAL_ENERGY) {
  //  std::stringstream msg;
  //  msg << "### FATAL ERROR in galpause.cpp: requires DUAL_ENERGY" << std::endl;
  //  throw std::runtime_error(msg.str().c_str());
  //}
      
  if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) 
    EnrollUserBoundaryFunction(INNER_X1,InnerX1_Wind);
  if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) 
    EnrollUserBoundaryFunction(OUTER_X1,OuterX1_HydroStat);

  x1rat      = pin->GetOrAddReal("mesh","x1rat",1.0);
  x2rat      = pin->GetOrAddReal("mesh","x2rat",1.0);
  if (x1rat < 0.0) {
    EnrollUserMeshGenerator(X1DIR,PowerGridX1);
    // Check mesh
    if (mesh_size.nx2 > 1) {
      Real nx2exp = (PI*std::pow(mesh_size.x1max/mesh_size.x1min,0.5/mesh_size.nx1))
                   /(std::pow(mesh_size.x1max/mesh_size.x1min,1.0/mesh_size.nx1)-1.0);
      if (std::max((Real) mesh_size.nx2,nx2exp) > 2.0*std::min((Real) mesh_size.nx2,nx2exp)) {
        std::stringstream msg;
        msg << "### FATAL ERROR in galpause.cpp: Aspect ratio > 2: " 
            << " nx2 =" << std::scientific << std::setw(10) << std::setprecision(2) << (Real) mesh_size.nx2
            << " nx2exp =" << std::scientific << std::setw(10) << std::setprecision(2) << nx2exp
            << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
    }
  }
  if (x2rat == -2.0) {
    EnrollUserMeshGenerator(X2DIR,QuadraticX2);
  } else if (x2rat == -1.0) {
    EnrollUserMeshGenerator(X2DIR,ExponentialX2);
  }

  if (EXPANDING_ENABLED) {
    SetGridData(4);
    EnrollGridDiffEq(WallVel);
    EnrollCalcGridData(UpdateGridData);
    ttrack.NewAthenaArray(maxntrack); // for position tracking
    rtrack.NewAthenaArray(maxntrack);
    
    GridData(0) = mesh_size.x1min;
    GridData(1) = 1; 
    GridData(2) = 0.0;
    GridData(3) = mesh_size.x1max; 
    itrack      = pin->GetOrAddInteger("problem","itrack",0); // 0: old galpause (ring) tracking, 1: pressure tracking
    vtrack0     = pin->GetOrAddReal("problem","vtrack0",0.0); // tracking velocity for (here) constant expansion
  }

  gam      = pin->GetReal("hydro","gamma");
  gm1      = gam-1.0;
  icool    = pin->GetOrAddReal("problem","icool",0);
  iwiso    = pin->GetOrAddReal("problem","iwiso",0); // keep wind isothermal. Only works with icool>0.
  coolsafe = pin->GetOrAddReal("problem","coolsafe",0.05);
  zmetwind = pin->GetOrAddReal("problem","zmetwind",1.0);
  zmethalo = pin->GetOrAddReal("problem","zmethalo",0.1);
  gammaheat0= pin->GetOrAddReal("problem","gammaheat0",0.0);
  lodxref  = pin->GetOrAddReal("problem","lodxref",1.0);
  lodxderef= pin->GetOrAddReal("problem","lodxderef",2.0);
  int ikappa = pin->GetOrAddInteger("problem","ikappa",0); // ikappa == 1: Spitzer Conduction
  if (icool > 0) {
    EnrollUserExplicitSourceFunction(HeatCool);
    EnrollUserTimeStepFunction(HeatCoolTimeStep);
  }
#ifdef WIND
  k1_.NewAthenaArray(2);
  k2_.NewAthenaArray(2);
  k3_.NewAthenaArray(2);
  k4_.NewAthenaArray(2); 
  y0_.NewAthenaArray(2);
  y1_.NewAthenaArray(2);
  ytemp_.NewAthenaArray(2);
  dydx_.NewAthenaArray(2);
#endif

  if (ikappa == 1) 
    EnrollConductionCoefficient(SpitzerConduction);

  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  int iprof = pin->GetOrAddInteger("problem","iprof",1);
  if (iprof == 0) {
    ProfileFunc = ConstProfile;
  } else if (iprof == 1) {
    ProfileFunc = BetaProfile;
    EnrollStaticGravPotFunction(BetaPotential);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in galpause.cpp: invalid value for iprof." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  return;
}

//========================================================================================
//! \fn void OuterX1_HydroStat(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for outer boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================
void OuterX1_HydroStat(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd 
      for (int i=1; i<=ngh; ++i) {
        Real r = pco->x1v(ie+i);
        //Real d = pparam->Dens0()*pow(1.0+SQR(r/pparam->Rbeta()),-1.5*pparam->Beta());
        Real d = ProfileFunc(r);
        prim(IDN,k,j,ie+i) = d;
        prim(IPR,k,j,ie+i) = d*pparam->Temp();
        if (DUAL_ENERGY) prim(IGE,k,j,ie+i) = d*pparam->Temp();
        prim(IVX,k,j,ie+i) = 0.0;
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = 0.0;
        prim(NHYDRO-NSCALARS  ,k,j,ie+i) = 0.1; // metalicity
        prim(NHYDRO-NSCALARS+1,k,j,ie+i) = 1.0; // ambient 
        prim(NHYDRO-NSCALARS+2,k,j,ie+i) = 0.0; // wind
      }
    }
  } 

  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(ie+i)) = bx0;  
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(ie+i)) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(ie+i)) = bz0;
        }
      }
    }
  }
  return;

}

//========================================================================================
//! \fn void InnerX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for inner boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================

void InnerX1_Wind(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      Real x2 = pco->x2v(j);
#pragma omp simd 
      for (int i=1; i<=ngh; ++i) {
        Real r   = pco->x1v(is-i);
        Real dw  = pparam->Dwind(r); //pparam->Mdot()/(4.0*PI*fwind*SQR(r)*pparam->Vwind());
        //Real da = pparam->Dens0()*pow(1.0+SQR(r/pparam->Rbeta()),-1.5*pparam->Beta());
        Real da = ProfileFunc(r);
        Real fjp = 1.0-0.25*(1+std::tanh((x2-acosfwind)/0.02))*(1.0-std::tanh((x2-(PI-acosfwind))/0.02));
        fjp = ((je==js) || (pparam->Fwind()==1.0)) ? 1.0 : fjp;
        Real d   = da + (dw-da)*fjp;
        prim(IDN,k,j,is-i) = d;
        prim(IPR,k,j,is-i) = d*pparam->Temp();
        if (DUAL_ENERGY) prim(IGE,k,j,is-i) = d*pparam->Temp();
        prim(IVX,k,j,is-i) = pparam->Vwind()*fjp;
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;
        prim(NHYDRO-NSCALARS  ,k,j,is-i) = zmethalo+(zmetwind-zmethalo)*fjp; // metalicity
        prim(NHYDRO-NSCALARS+1,k,j,is-i) = 1.0-fjp; // ambient 
        prim(NHYDRO-NSCALARS+2,k,j,is-i) = fjp; // wind
      }
    }
  }

  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(is-i)) = bx0;  
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(is-i)) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(is-i)) = bz0;
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn Real PowerGridX1(Real x, RegionSize rs)
//  \brief Generates grid following r_i = r_0*(r1/r0)**i
//========================================================================================
Real PowerGridX1(Real x, RegionSize rs) {
  //Real delta = pow((rs.x1max/rs.x1min),1.0/((Real)nx1));
  //Real r     = rs.x1min*pow(delta,x*((Real)nx1));
  Real delta = rs.x1max/rs.x1min;
  Real r     = rs.x1min*pow(delta,x);
  //fprintf(stdout,"[PowerGridX1]: x=%17.9e delta=%17.9e delta^(1/n)=%17.9e\n",x,delta,pow(delta,1.0/((Real) nx1)));
  return r;
}

//========================================================================================
//! \fn Real CompressedX2(Real x, RegionSize rs)
//  \brief Generates grid following theta_i = 3*psi^2+1 (see Athena++ documentation)
//========================================================================================

Real QuadraticX2(Real x, RegionSize rs)
{
  Real a = 3.0;
  return rs.x2min + (3.0/(a+3.0))*x*(a*x*(x/3.0-1.0)+a+1.0)*(rs.x2max-rs.x2min);
}

Real ExponentialX2(Real x, RegionSize rs)
{
  Real a = 3.0;
  return rs.x2min + (rs.x2max-rs.x2min)*(1.0-std::exp(-a*x))/(1.0-std::exp(-a));
}

//========================================================================================
//! \fn void DdensDr(const Real r, const Real dens)
//  \brief Derivative for RHS of RK4.
//    Constants hardwired;
//========================================================================================

#ifdef WURST
void GSSAWdydr(const Real r, const AthenaArray<Real> &y, AthenaArray<Real> &dydr) {
  Real cs2    = gam*0.1*std::pow(y(0)/1.0,gm1);
  Real m2     = SQR(y(1))/cs2;
  dydr(1)     = 2.0*y(1)/((m2-1.0)*r);  // velocity 
  dydr(0)     = -y(0)*(2.0/r+dydr(1)/y(1)); // density
  return;
}

void GetSteadyStateAdbWind(const Real r0, const AthenaArray<Real> &y0, const Real dr, AthenaArray<Real> &y) { 
  GSSAWdydr(r0,y0,k1);
  for (int i=0; i<2; ++i) {
    k1(i) *= dr;
    ytemp_(i) = y0(i) + 0.5*k1(i);
  }
  GSSAWdydr(r0+0.5*dr,ytemp_,k2); 
  for (int i=0; i<2; ++i) {
    k2(i) *= dr;
    ytemp_(i) = y0(i) + 0.5*k2(i);
  }
  GSSAWdydr(r0+0.5*dr,ytemp_,k3);
  for (int i=0; i<2; ++i) {
    k3(i) *= dr;
    ytemp_(i) = y0(i) +     k3(i);
  }
  GSSAWdydr(r0+    dr,ytemp_,k4);
  for (int i=0; i<2; ++i) {
    k4(i) *= dr;
    y(i)   = y0(i) + (k1(i)+2.0*(k2(i)+k3(i))+k4(i))/6.0;
  }

  y(2) = gam*0.1*std::pow(y(0)/1.0,gm1)*y(0);
  return;
}
#endif

void Rk4Profile(const Real s, const AthenaArray<Real> &y, AthenaArray<Real> &k) {
  const Real kpc  = 8.80388050e-3;
  const Real kmsc = 9.08222098e-02;
  const Real rs   = 2.06e1/kpc;
  const Real vesc = 230.0/kmsc;
  const Real rmin = 0.1/kpc;
  const Real rmax = 300.0/kpc;

  //  a     = y[0]/p._rs
  //  dd    = np.zeros(5)
  //  dd[0] = p._rmin*(p._rmax/p._rmin)**x * np.log(p._rmax/p._rmin) # radius
  //  dd[1] = y[1]*(p._vesc/p._cs)**2/(2*np.log(2))*(a/(1+a)-np.log(1+a))/a**2 # density
  //  dd[2] = 1.5*p._T0*y[1]*p._kb*4*np.pi*y[0]**2 # internal energy
  //  dd[3] = 1.5*p._mdot*p._vw # wind energy
  //  phi   = (p._vesc**2/(2*np.log(2)))*np.log(1+a)/a
  //  dd[4] = 2.0*np.pi*y[0]**2*y[1]*p._matom*phi # gravitational energy

  Real a     = y(0)/rs;
  Real vrat  = SQR(vesc)/csound2;
  k(0)      = rmin*std::pow(rmax/rmin,s) * std::log(rmax/rmin); // radius
  k(1)      = y(1) * (0.5*vrat/std::log(2.0)) * (a/(1.0+a) - std::log(1.0+a))/SQR(a);
  //fprintf(stdout,"s=%14.6e r=%14.6e rmin=%14.6e rmax=%14.6e a=%14.6e vrat=%14.6e rs=%14.6e dens=%14.6e dd=%14.6e\n",s,y(0),rmin,rmax,a,vrat,rs,y(1),k(1));
  //fprintf(stdout,"r=%13.5e dr=%13.5e dens=%13.5e ddens=%13.5e\n",y(0),k(0),y(1),k(1));
  return;
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real gamma;

  icool     = pin->GetOrAddInteger("problem","icool",0); // 0: no cooling, 1: power-law cooling, 2: WSS09 cooling
  iwiso     = pin->GetOrAddInteger("problem","iwiso",0); // 0: keep wind adiabatic, 1: keep wind isothermal
  turbcool  = pin->GetOrAddReal("problem","turbcool",0.0); // add turbulent heating
  boost     = pin->GetReal("problem","boost"); // 2.0 works ok for full angle. 
  // potential parameters
  Real vesc = pin->GetReal("problem","vesc"); // km/s 
  Real temp = pin->GetReal("problem","temp0");
  Real mdot = pin->GetReal("problem","mdot"); // Msol/yr
  Real vwind= pin->GetReal("problem","vwind"); // km/s
  Real n0   = pin->GetReal("problem","n0"); // cm^(-3)
  rmin = pin->GetReal("mesh","x1min"); 
  Real rmax = pin->GetReal("mesh","x1max"); 
  gamma   = peos->GetGamma();
  //pparam    = new Parameters(vesc,temp,mdot,vwind,n0,rmin,rmax,gamma);


  // wind parameters
  Real fwind   = pin->GetOrAddReal("problem","fwind",0.1);
  zmetwind= pin->GetOrAddReal("problem","zmetwind",1.0);
  zmethalo= pin->GetOrAddReal("problem","zmethalo",0.1);
  gm1     = gamma - 1.0;
  csound2 = temp;
  acosfwind = std::acos(1.0-fwind);

  if (icool == 1) {
    Real suppress = pin->GetOrAddReal("problem","suppress",1.0);
    Real mintemp  = pin->GetOrAddReal("problem","mintemp",-1.0);
    pcoolfunc = new CoolingFunction(suppress,mintemp);
  } else if (icool == 2) {
    pcoolfunc = new CoolingFunction("/nas/longleaf/home/fheitsch/cooling/hm12/data/z_0.000.hdf5");
  }
  if (icool ==2) { // allocate scratch arrays
    lengthcool = pin->GetOrAddReal("problem","lengthcool",1.13584e3); // set to 10kpc by default
    tolcool    = pin->GetOrAddReal("problem","tolcool",1.0e-12); // tolerance for implicit temperature equation
  }

  pparam    = new Parameters(vesc,temp,mdot,vwind,n0,rmin,rmax,gamma,fwind);



  // only spherical coordinates
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real r     = pcoord->x1v(i); 
        //Real den   = pparam->Dens0()*pow(1.0+SQR(r/pparam->Rbeta()),-1.5*pparam->Beta());
        Real den   = ProfileFunc(r);
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = den*pparam->Temp()/gm1 + 0.5*(  SQR(phydro->u(IM1,k,j,i))
                                                               + SQR(phydro->u(IM2,k,j,i))
                                                               + SQR(phydro->u(IM3,k,j,i)))
                                                             / phydro->u(IDN,k,j,i);
          if (DUAL_ENERGY) phydro->u(IIE,k,j,i) = den*pparam->Temp()/gm1;
        }
        phydro->u(NHYDRO-NSCALARS  ,k,j,i) = zmethalo*den; // metallicity
        phydro->u(NHYDRO-NSCALARS+1,k,j,i) = den; // ambient
        phydro->u(NHYDRO-NSCALARS+2,k,j,i) = 0.0; // wind
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
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
    if (NON_BAROTROPIC_EOS) { // careful here: mix of volume and face positions...
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            phydro->u(IEN,k,j,i) += 0.5*(  SQR(pfield->b.x1f(k,j,i))
                                       + SQR(pfield->b.x2f(k,j,i))
                                       + SQR(pfield->b.x3f(k,j,i)));
          }
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

void MeshBlock::UserWorkInLoop(void) {
  return; 
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief Called for individual meshblock
//========================================================================================

void Mesh::UserWorkInLoop(void) {

  if (ncycle_out != 0) {
    if (ncycle % ncycle_out != 0) {
      return;
    }
  }

  MeshBlock *pmb=pblock;

  if (EXPANDING_ENABLED) {
    if (Globals::my_rank==0) {
      if ((x1rat < 0.0) && (mesh_size.nx2 > 1)) {
        Real nx2exp = (PI*std::pow(mesh_size.x1max/mesh_size.x1min,0.5/mesh_size.nx1))
                     /(std::pow(mesh_size.x1max/mesh_size.x1min,1.0/mesh_size.nx1)-1.0);
        fprintf(stdout,"[UserWorkInLoop]: rmax = %13.5e nx2exp/nx2 = %13.5e\n",mesh_size.x1max,nx2exp/((Real) mesh_size.nx2));
      } else {
        fprintf(stdout,"[UserWorkInLoop]: rmax = %13.5e\n",mesh_size.x1max);
      }
    }
  }

  const int nq = 11;
  Real qtot[nq]; // 0: vol, 1: dens, 2: vtot, 3: etot, 4: eint, 5: ekin, 6: emag, 7-9: v1-v3
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

  bool allfail = false, fail = false;
  int ifail = 0;

  while (pmb != NULL) { // collect results from individual pmbs
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      Real x3  = pmb->pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; j++) {
        Real x2  = pmb->pcoord->x2v(j);
        Real xtest = pmb->pcoord->x1f(pmb->is);
        Real eint;
        for (int i=pmb->is; i<=pmb->ie; i++) {
          Real dx1  = pmb->pcoord->dx1f(i);
          Real x1  = pmb->pcoord->x1v(i);
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
            fail = fail || isnan(eint) || (eint <= 0.0);
          }
          if (fail) {
            if (DUAL_ENERGY) {
              std::cout << "[UserWorkInLoop]: Warning: i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
                        << " r =" << std::scientific << std::setw(11) << std::setprecision(3) << x1
                        << " dr=" << std::scientific << std::setw(11) << std::setprecision(3) << dx1
                        << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << u[IDN]
                        << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IM1]
                        << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IM2]
                        << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IM3]
                        << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IEN]
                        << " ei=" << std::scientific << std::setw(11) << std::setprecision(3) << u[IIE]
                        << std::endl;
            } else {
              std::cout << "[UserWorkInLoop]: Warning: i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
                        << " r =" << std::scientific << std::setw(11) << std::setprecision(3) << x1
                        << " dr=" << std::scientific << std::setw(11) << std::setprecision(3) << dx1
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
          //fprintf(stdout,"[UserWorkInLoop]: i=%4i xrat=%17.9e dxrat=%17.9e xtest=%17.9e\n",
          //        i,pmb->pcoord->x1f(i+1)/pmb->pcoord->x1f(i),pmb->pcoord->dx1f(i+1)/pmb->pcoord->dx1f(i),(xtest+dx1)/xtest);
          //xtest += dx1;
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
          if (DUAL_ENERGY) ener[6] = u[IIE];
          ener[7] = u[IM1]/u[IDN];
          ener[8] = u[IM2]/u[IDN];
          ener[9] = u[IM3]/u[IDN];
          ener[3] = ener[2]-ener[4]-ener[5];
          if (DUAL_ENERGY) {
            ener[10] = ener[6]*gm1/ener[1];
          } else {
            ener[10] = ener[3]*gm1/ener[1];
          }
          for (int q=1; q<nq; q++) {
            qtot[q] += ener[q]*dvol;
            if (ener[q] < qmin[q]) qmin[q] = ener[q];
            if (ener[q] > qmax[q]) qmax[q] = ener[q];
          } 
          // shell tracking
          Real rad;
          if (COORDINATE_SYSTEM=="spherical_polar") {
            rad = x1; 
          } else {
            rad = std::sqrt(x1*x1+x2*x2+x3*x3);
          }
          trac[0] += u[IDN];
          if (COORDINATE_SYSTEM=="spherical_polar") {
            trac[1] += u[IM1];
          } else {
            trac[1] += (u[IM1]*x1+u[IM2]*x2+u[IM3]*x3)/rad;
          }
          trac[2] += u[IDN]*rad;
          Real vt   = std::sqrt(SQR(u[IM1])+SQR(u[IM2])+SQR(u[IM3]))/SQR(u[IDN]);
          Real temp = gm1*ener[2]/u[IDN];
          Real zmet = u[NHYDRO-NSCALARS]/u[IDN];
          if (icool > 0) {
            Real w[NHYDRO];
            for (int n=0; n<NHYDRO; ++n)
              w[n] = u[n];
            w[IVX] /= w[IDN];
            w[IVY] /= w[IDN];
            w[IVZ] /= w[IDN];
            for (int n=NHYDRO-NSCALARS; n<NHYDRO; ++n) 
              w[n] /= w[IDN];
            w[IPR]     = temp*w[IDN];
            Real lc    = pcoolfunc->CoolingLength(w,rad,pmb->pmy_mesh->dt);
            lengrat[0] = std::min(lengrat[0],lc/dx1); // cooling length
            //if (pmb->phydro->phdif->kappa_iso > 0.0) 
            //  lengrat[1] = std::min(lengrat[1],std::sqrt(pmb->phydro->phdif->kappa(ISO,k,j,i)*temp/(u[IDN]*cf))/dx1); // Field length
          }
          //fprintf(stdout,"[UserWorkInLoop]: i=%2i d=%13.5e v=%13.5e e=%13.5e\n",i,u[IDN],u[IM1]/u[IDN],u[IEN]);
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
  int ierr;
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qtot,nq,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qmin,nq,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qmax,nq,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&trac,3 ,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&lengrat,2,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&allfail,1,MPI_C_BOOL,MPI_LOR,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&ifail,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
  for (int q=1; q<nq; q++) qtot[q] /= qtot[0];
  for (int q=1; q<3;  q++) trac[q] /= trac[0];

  if (Globals::my_rank==0) {
    std::cout << "[UserWorkInLoop]: lcool= " << std::scientific << std::setw(13) << std::setprecision(5) << lengrat[0]
              << " lfield= "                   << std::scientific << std::setw(13) << std::setprecision(5) << lengrat[1]
              << std::endl;
    std::cout << "[UserWorkInLoop]: vrad = " << std::scientific << std::setw(13) << std::setprecision(5) << trac[1]
              << " rad = "                   << std::scientific << std::setw(13) << std::setprecision(5) << trac[2]
              << std::endl;
    std::cout << "[UserWorkInLoop]: dens = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[1]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[1]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[1]
              << std::endl;
    std::cout << "[UserWorkInLoop]: temp = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[10]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[10]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[10]
              << std::endl;
    std::cout << "[UserWorkInLoop]: vel1 = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[7]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[7]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[7]
              << std::endl;
    std::cout << "[UserWorkInLoop]: vel2 = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[8]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[8]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[8]
              << std::endl;
    std::cout << "[UserWorkInLoop]: vel3 = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[9]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[9]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[9]
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
  if (dt < 1e-20) {
    std::cout << "[UserWorkInLoop]: Timestep dropped below 1e-12. Failure." << std::endl;
    stop_this();
  }

  return;
}
