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
#include <hdf5.h>  // H5[F|P|S|T]_*, H5[A|D|F|P|S|T]*(), hid_t
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// ahead declarations

class CoolingFunction;
CoolingFunction* pcoolfunc;
class Parameters;
Parameters* pparam;

typedef Real (*TimeStepFunc_t)(MeshBlock *pmb);

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
    Real vesc,temp,dens0,pigm,phi0,delphi,rho0,rscal,cpar,mvir,mdot,vwind,beta,rbeta,x1min,x1max;

  public:
    // This must be unit-less. Otherwise just a nightmare.
    Parameters(const Real vesc0, const Real T0, const Real mdot0, const Real vwind0, const Real n0, const Real x1min0, const Real x1max0) {
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
};
//========================================================================================
// Radiative loss functions
//========================================================================================

//typedef Real (*CoolingFunc_t)(const Real dens, const Real temp); // For generic cooling functions

int nx1, icool, iprof;
Real x1rat, gam,gm1, fwind, csound2,rmin,vexp;
Real coolsafe = 0.05;
AthenaArray<Real> k1, k2, k3, k4, yarr0, yarr1, ytemp_;

void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);
//CoolingFunc_t CoolingFunc;
void GSSAWdydr(const Real r, const AthenaArray<Real> &y, AthenaArray<Real> &dydr);
void GetSteadyStateAdbWind(const Real r0, const AthenaArray<Real> &y0, const Real dr, AthenaArray<Real> &y);
Real HaloProfile(const Real r);
Real BetaProfile(const Real r); // Sets ambient density for beta model
Real TestProfile(const Real r);
Real ConstProfile(const Real r);
Real LinProfile(const Real r);
Real NFWProfile(const Real r);
Real WindProfile(const Real r); // Sets wind model.
Real BetaPotential(const Real x1, const Real x2, const Real x3, const Real time);
Real LinearPotential(const Real x1, const Real x2, const Real x3, const Real time);
Real NFWPotential(const Real x1, const Real x2, const Real x3, const Real time);
void DdensDr(const Real s,  const AthenaArray<Real> &y, AthenaArray<Real> &k);

static void stop_this();

//========================================================================================
// Time Dependent Grid Functions
//  \brief Functions for time dependent grid, including two example boundary conditions
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
void UpdateGridData(Mesh *pm);

//Global Variables for OuterX1
Real bx0,by0,bz0;

void OuterX1_HydroStat(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void InnerX1_Wind(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

Real PowerGridX1(Real x, RegionSize rs);

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

//========================================================================================
// Returns the density for the beta model at beta = 0.5
//========================================================================================

Real HaloProfile(const Real r) {
  if ((iprof == -2) || (iprof == -1)) {
    return TestProfile(r);
  } else if (iprof == -4) {
    return LinProfile(r);
  } else if (iprof == 0) {
    return ConstProfile(r);
  } else if (iprof == 1) {
    return BetaProfile(r);
  } else if (iprof == 2) {
    return NFWProfile(r);
  }
}

Real BetaProfile(const Real r) {
  Real n00 = pparam->Dens0();
  Real nb = n00*pow(1.0+SQR(r/pparam->Rbeta()),-1.5*pparam->Beta());
  //fprintf(stdout,"[BetaProfile]: r=%13.5e n=%13.5e\n",r,nb);
  return nb;
}

Real NFWProfile(const Real r) {
  Real x  = r/pparam->Rscal();
  Real nr = pparam->Dens0()*std::exp(-pparam->Delphi()*(1.0-std::log(1.0+x)/x));
  //fprintf(stdout,"[NFWProfile]: r=%13.5e n=%13.5e\n",r,nr);
  return nr;
}

Real TestProfile(const Real r) {
  return pparam->Dens0();
}

Real LinProfile(const Real r) {
  Real n;
  n = pparam->Dens0()*(pparam->X1min()/r);
  return n;
}

Real ConstProfile(const Real r) {
  Real n;
  n = pparam->Pigm()/pparam->Temp();
  return n;
}

Real WindProfile(const Real r) {
  Real nw;
  if (iprof == -2) {
    nw = pparam->Dens0();
    //nw = pparam->Mdot()/(4.0*PI*fwind*SQR(r)*pparam->Vwind());
  } else {
    nw = pparam->Mdot()/(4.0*PI*fwind*SQR(r)*pparam->Vwind());
  }
  //fprintf(stdout,"[WindProfile]: r=%13.5e n=%13.5e\n",r,nw);
  return nw;
}

Real BetaPotential(const Real x1, const Real x2, const Real x3, const Real time) {
  Real x   = x1/pparam->Rbeta();
  Real phi = 1.5*pparam->Beta()*pparam->Temp()*std::log(1.0+SQR(x));
  return phi;
}

Real NFWPotential(const Real x1, const Real x2, const Real x3, const Real time) {
  Real  x  = x1/pparam->Rscal(); // assuming spherical coordinates
  Real phi = pparam->Phi0()*std::log(1.0+x)/x;
  return phi;
}

//========================================================================================
// Class CoolingFunction
// see pyyacc.py
//========================================================================================
class CoolingFunction {

  private:
    const int nccoeff      = 13;
    const int nhcoeff      = 3;
    int mode               = 0; // 0: piece-wise power law, 1: WSS09 table
    const Real fac         = 2.167177868e+31;
    const Real trangec[13] = {3.0,  5e1,  1e3,   8e3,  2.0e4,  4e4,    8e4, 1.13e5, 1.33e5,   1.86e5,   6.6e5,1.0e6,  1.33e7};
    const Real cexpo[13]   = {3.0,  1.1,  0.1,   5.0,    0.8, -1.0,   -2.0,   -3.0,   -1.8,    -0.09,  4.8965,  3.5,     3.5};
    const Real loss0       = 4.0e-30;
    const Real gain0       = 1.2e-24;
    const Real trangeh[3]  = {3.0,5e3,1e10};
    const Real hexpo[3]    = {0.0,-1.0,0.0};
    Real lhcoeff[3];
    Real ltrangeh[3];
    Real lccoeff[13];
    Real ltrangec[13];
    // Wiersma+09. "0" is metal-free, "z" is only metals, "s" is solar.
    AthenaArray<Real> xion_0, xion_s, netcool_0, netcool_z, densarr, temparr; 
    int ndens=-1, ntemp=-1;

  public:

    CoolingFunction() {
      mode = 0; // piece-wise power law
      lccoeff[0] = std::log(loss0);
      for (int i=0; i<nccoeff; ++i) ltrangec[i] = std::log(trangec[i]);
      for (int i=1; i<nccoeff; ++i) lccoeff[i]  = lccoeff[i-1] + cexpo[i-1]*(ltrangec[i]-ltrangec[i-1]);
      lhcoeff[0] = std::log(gain0);
      for (int i=0; i<nhcoeff; ++i) ltrangeh[i] = std::log(trangeh[i]);
      for (int i=1; i<nhcoeff; ++i) lhcoeff[i]  = lhcoeff[i-1] + hexpo[i-1]*(ltrangeh[i]-ltrangeh[i-1]);
      return;
    };

    CoolingFunction(const char* path) {
      mode  = 1; // WSS09
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
      std::cout << "File size = " << size << " bytes" << std::endl;

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
      std::cout << "Group info:" << std::endl;
      if (grp_info.storage_type != H5G_STORAGE_TYPE_SYMBOL_TABLE) {
        std::cout << "Storage type must be Symbol Table" << std::endl;
        stop_this;
      }
      std::cout << "Link count in HDF5 group " << object_name.c_str() << ": " << grp_info.nlinks << std::endl;
      // check for fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
        delete link_name;
      }
      // read fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
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
          std::cout << "        Dataset " << path_to_object.c_str() << " has space " << dataspace << std::endl;
          int datarank     = H5Sget_simple_extent_ndims(dataspace);
          std::cout << "        Dataset " << path_to_object.c_str() << " has rank "  << datarank << std::endl;
          hsize_t dims[H5S_MAX_RANK], maxdim[H5S_MAX_RANK];
          H5Sget_simple_extent_dims(dataspace, dims, maxdim);
          hsize_t nelts = 1;
          for (int ir=0; ir<datarank; ir++) {
            std::cout << "            dim[" << ir << "] = " << dims[ir] << std::endl;
            nelts *= dims[ir];
          }
          hid_t datatype   = H5Dget_type(dataset);
          std::cout << "        Dataset " << path_to_object.c_str() << " has type "  << datatype << std::endl;
          hid_t dataclass  = H5Tget_class(datatype);
          std::cout << "        Datatype " << path_to_object.c_str() << " has class "  << dataclass << std::endl;
          if (dataclass != H5T_FLOAT) {
            std::cout << "Expected dataclass H5T_FLOAT." << std::endl;
            stop_this();
          }
          hid_t datanative = H5Tget_native_type(datatype, H5T_DIR_DEFAULT);
          size_t datasize  = H5Tget_size(datanative);
          std::cout << "        Datatype " << path_to_object.c_str() << " has size " << datasize << std::endl;
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
      std::cout << "Group info:" << std::endl;
      if (grp_info.storage_type != H5G_STORAGE_TYPE_SYMBOL_TABLE) {
        std::cout << "Storage type must be Symbol Table" << std::endl;
        stop_this;
      }
      std::cout << "Link count in HDF5 group " << object_name.c_str() << ": " << grp_info.nlinks << std::endl;
      // check for fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
        delete link_name;
      }
      // read fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
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
          std::cout << "        Dataset " << path_to_object.c_str() << " has space " << dataspace << std::endl;
          int datarank     = H5Sget_simple_extent_ndims(dataspace);
          std::cout << "        Dataset " << path_to_object.c_str() << " has rank "  << datarank << std::endl;
          hsize_t dims[H5S_MAX_RANK], maxdim[H5S_MAX_RANK];
          H5Sget_simple_extent_dims(dataspace, dims, maxdim);
          hsize_t nelts = 1;
          for (int ir=0; ir<datarank; ir++) {
            std::cout << "            dim[" << ir << "] = " << dims[ir] << std::endl;
            nelts *= dims[ir];
          }
          hid_t datatype   = H5Dget_type(dataset);
          std::cout << "        Dataset " << path_to_object.c_str() << " has type "  << datatype << std::endl;
          hid_t dataclass  = H5Tget_class(datatype);
          std::cout << "        Datatype " << path_to_object.c_str() << " has class "  << dataclass << std::endl;
          if (dataclass != H5T_FLOAT) {
            std::cout << "Expected dataclass H5T_FLOAT." << std::endl;
            stop_this();
          }
          hid_t datanative = H5Tget_native_type(datatype, H5T_DIR_DEFAULT);
          size_t datasize  = H5Tget_size(datanative);
          std::cout << "        Datatype " << path_to_object.c_str() << " has size " << datasize << std::endl;
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
      std::cout << "Group info:" << std::endl;
      if (grp_info.storage_type != H5G_STORAGE_TYPE_SYMBOL_TABLE) {
        std::cout << "Storage type must be Symbol Table" << std::endl;
        stop_this;
      }
      std::cout << "Link count in HDF5 group " << object_name.c_str() << ": " << grp_info.nlinks << std::endl;
      // check for fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
        delete link_name;
      }
      // read fields
      for (int ilink=0; ilink<grp_info.nlinks; ilink++) {
        ssize_t size_l = H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, 0, 0, H5P_DEFAULT);
        char *link_name = new char[++size_l];
        H5Lget_name_by_idx(file, object_name.c_str(), H5_INDEX_NAME, H5_ITER_INC, ilink, link_name, size_l, H5P_DEFAULT);
        std::cout << "    Link " << ilink << ": name   = " << link_name <<  std::endl;
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
          std::cout << "        Dataset " << path_to_object.c_str() << " has space " << dataspace << std::endl;
          int datarank     = H5Sget_simple_extent_ndims(dataspace);
          std::cout << "        Dataset " << path_to_object.c_str() << " has rank "  << datarank << std::endl;
          hsize_t dims[H5S_MAX_RANK], maxdim[H5S_MAX_RANK];
          H5Sget_simple_extent_dims(dataspace, dims, maxdim);
          hsize_t nelts = 1;
          for (int ir=0; ir<datarank; ir++) {
            std::cout << "            dim[" << ir << "] = " << dims[ir] << std::endl;
            nelts *= dims[ir];
          }
          hid_t datatype   = H5Dget_type(dataset);
          std::cout << "        Dataset " << path_to_object.c_str() << " has type "  << datatype << std::endl;
          hid_t dataclass  = H5Tget_class(datatype);
          std::cout << "        Datatype " << path_to_object.c_str() << " has class "  << dataclass << std::endl;
          if (dataclass != H5T_FLOAT) {
            std::cout << "Expected dataclass H5T_FLOAT." << std::endl;
            stop_this();
          }
          hid_t datanative = H5Tget_native_type(datatype, H5T_DIR_DEFAULT);
          size_t datasize  = H5Tget_size(datanative);
          std::cout << "        Datatype " << path_to_object.c_str() << " has size " << datasize << std::endl;
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

    Real EvalLoss(const Real temp) {
      Real lt = std::log(temp);
      if (lt < ltrangec[0]) {
        return 0.0;
      } else {
        int i=0;
        while ((i < nccoeff) && (ltrangec[i] <= lt)) i++;
        return std::exp(lccoeff[i-1]+cexpo[i-1]*(lt-ltrangec[i-1]));
      }
    };

    Real EvalGain(const Real temp) {
      Real lt = std::log(temp);
      if (lt < ltrangeh[0]) {
        return std::exp(lhcoeff[0]);
      } else {
        int i=0;
        while ((i < nhcoeff) && (ltrangeh[i] <= lt)) i++;
        return std::exp(lhcoeff[i-1]+hexpo[i-1]*(lt-ltrangeh[i-1]));
      }
    };

    Real HeatCoolFunc(const Real dens, const Real temp, const Real zmet) {
      if (mode == 0) { // piece-wise power law
        Real gamma  = EvalGain(temp);
        Real lambda = EvalLoss(temp);
        Real dedt = (gamma-dens*lambda)*fac;
        return dedt;
      } else if (mode == 1) { // WSS09
        int idens,itemp;
        Real dedt,wd,wt,owd,owt,lambda,lambda_0,lambda_z,xe_0,xe_s;
        // need to interpolate in density and temperature first.
        idens = (int) std::floor(ndens*std::log10(dens/densarr(0))/std::log10(densarr(ndens-1)/densarr(0))); 
        idens = (idens < 0 ? 0 : idens);
        idens = (idens > ndens-1 ? ndens-1 : idens);
        wd    = (dens-densarr(idens))/(densarr(idens+1)-densarr(idens));
        owd   = 1.0-wd;
        itemp = (int) std::floor(ntemp*std::log10(temp/temparr(0))/std::log10(temparr(ntemp-1)/temparr(0)));
        itemp = (itemp < 0 ? 0 : itemp);
        itemp = (itemp > ntemp-1 ? ntemp-1 : itemp);
        wt    = (temp-temparr(itemp))/(temparr(itemp+1)-temparr(itemp));
        owt   = 1.0-wt;
        // calculate lambda. 
        lambda_0 =   netcool_0(itemp  ,idens  )*owt*owd
                   + netcool_0(itemp  ,idens+1)*owt* wd
                   + netcool_0(itemp+1,idens  )* wt*owd
                   + netcool_0(itemp+1,idens+1)* wt* wd;
        lambda_z =   netcool_z(itemp  ,idens  )*owt*owd
                   + netcool_z(itemp  ,idens+1)*owt* wd
                   + netcool_z(itemp+1,idens  )* wt*owd
                   + netcool_z(itemp+1,idens+1)* wt* wd;
        xe_0 =       xion_0   (itemp  ,idens  )*owt*owd
                   + xion_0   (itemp  ,idens+1)*owt* wd
                   + xion_0   (itemp+1,idens  )* wt*owd
                   + xion_0   (itemp+1,idens+1)* wt* wd;
        xe_s =       xion_s   (itemp  ,idens  )*owt*owd
                   + xion_s   (itemp  ,idens+1)*owt* wd
                   + xion_s   (itemp+1,idens  )* wt*owd
                   + xion_s   (itemp+1,idens+1)* wt* wd;
        lambda = lambda_0 + lambda_z* (xe_0/xe_s)*zmet;
        dedt   = -dens*lambda*fac; // cooling is positive in WSS09
        return dedt;
      }
    };

    //========================================================================================
    // Real RootFunc(const Real dens, const Real temp0, const Real, temp1, const Real dt)
    // This is not the thermal equilibrium, but the implicit update for the thermal ODE
    //   dT = dt*(gamma-1)/kB * (Gamma(T)-n*Lambda(T)).
    //   As long as dt is limited to a fraction of dtcool (see HeatCool), the implicit solution
    //   prevents under- or over-shoots.
    //========================================================================================
    Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real dt, const Real zmet) {
      return temp0 + dt*gm1*HeatCoolFunc(dens,temp1,zmet) - temp1;
    };

    //========================================================================================
    // Real BracketRoot(const Real temp0, const Real dt)
    //========================================================================================
    Real BracketRoot(const Real dens, const Real temp0, const Real dt, const Real zmet) {
      Real rf    = RootFunc(dens,temp0,temp0,dt,zmet);
      Real sig   = (Real) ((rf > 0) - (rf < 0));
      Real fac   = 1.0 + sig*0.1;
      Real temp1 = temp0;
      while (rf*RootFunc(dens,temp0,temp1,dt,zmet) > 0)
        temp1 *= fac;
      return temp1;
    };

    //========================================================================================
    // Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt)
    // \brief Finds root for RootFunc via bisection. Version without if-statements.
    //========================================================================================
    Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt, const Real zmet) {
      if (HeatCoolFunc(dens,temp0,zmet) == 0.0) return temp0; // Nothing to do for thermal equilibrium
      // Otherwise, temp1 and temp0 bracket the temperature down to which we should integrate.
      const Real tol = 1e-6;
      int nit = (int) (log(fabs(temp1-temp0)/tol)/log(2.0));
      Real T[3], L[2];
      T[0]         = temp0;
      T[1]         = temp1;
      T[2]         = 0.5*(T[0]+T[1]);
      L[0]         = RootFunc(dens,temp0,T[0],dt,zmet);
      L[1]         = RootFunc(dens,temp0,T[2],dt,zmet);
      for (int i=0; i<nit; i++) {
        int w = (L[0]*L[1] < 0); // 0 if >0, 1 if <= 0
        T[w]  = T[2];
        L[w]  = L[1];
        T[2]  = 0.5*(T[0]+T[1]);
        L[1]  = RootFunc(dens,temp0,T[2],dt,zmet);
      }
      return T[2];
    };

    //========================================================================================
    // Real GetEquiTemp(const Real dens, const Real temp0)
    // \brief Finds equilibrium temperature for cooling curve with equilibrium.
    //   Contains bracketing step as well. Assumes that new temperature < temp0.
    //========================================================================================
    Real GetEquiTemp(const Real dens, const Real temp0, const Real zmet) {
      Real T[3], L[2];
      Real fac, sig;
      const Real tol = 1e-6;
      int nit;
      L[0] = HeatCoolFunc(dens,temp0,zmet);
      if (L[0] == 0.0) return temp0;
      // bracket
      sig  = SIGN(L[0]); 
      fac  = 1.0 + sig*0.1;
      T[0] = temp0;
      T[1] = fac*T[0];
      L[1] = HeatCoolFunc(dens,T[1],zmet);
      while (L[0]*L[1] > 0.0) {
        T[0] = T[1];
        T[1] = fac*T[0];
        L[0] = L[1];
        L[1] = HeatCoolFunc(dens,T[1],zmet);
      }
      // root
      nit = (int) (log(fabs(T[1]-T[0])/tol)/log(2.0));
      T[2] = 0.5*(T[0]+T[1]);
      L[0] = HeatCoolFunc(dens,T[0],zmet);
      L[1] = HeatCoolFunc(dens,T[2],zmet);
      for (int i=0; i<nit; i++) {
        int w = (L[0]*L[1] < 0); // 0 if >0, 1 if <= 0
        T[w]  = T[2];
        L[w]  = L[1];
        T[2]  = 0.5*(T[0]+T[1]);
        L[1]  = HeatCoolFunc(dens,T[2],zmet);
      }
      return T[2];
    };
}; // class CoolingFunction

//========================================================================================
//! \fn void heatcool(...)
//  \brief Heating and cooling for user-defined cooling function
//  Implicit solution of ODE for temperature change (see RootFunc)
//  (See HeatCoolTimeStep).
//========================================================================================
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  Real g1  = pmb->peos->GetGamma()-1.0;

  AthenaArray<Real> dens, temp0, temp1, temp2, zmet, dener, dtcool, edot, sign; 
  dens.NewAthenaArray(prim.GetDim1());
  temp0.NewAthenaArray(prim.GetDim1());
  temp1.NewAthenaArray(prim.GetDim1());
  temp2.NewAthenaArray(prim.GetDim1()); 
  zmet.NewAthenaArray(prim.GetDim1());
  dener.NewAthenaArray(prim.GetDim1()); // Delta E by which to change total energy
  dtcool.NewAthenaArray(prim.GetDim1());
  edot.NewAthenaArray(prim.GetDim1());
  sign.NewAthenaArray(prim.GetDim1());

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real x32 = SQR(pmb->pcoord->x3v(k));
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real x22 = SQR(pmb->pcoord->x2v(j));
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        dens(i)  = prim(IDN,k,j,i); 
        if (DUAL_ENERGY) {
          temp0(i) = prim(IGE,k,j,i)/dens(i); // IGE is pressure
        } else {
          temp0(i) = prim(IPR,k,j,i)/dens(i);
        }
        zmet(i)  = prim(NHYDRO-NSCALARS,k,j,i);
      }
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        if (temp0(i) <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in galpause.cpp: HeatCool: temp0 <=0" << std::endl
              << "    p=" << std::setw(5) << Globals::my_rank << " i=" << std::setw(5) << i << " j=" << std::setw(5) << j << " k=" << std::setw(5) << k << std::endl
              << "    temp0=" << std::scientific << std::setw(13) << std::setprecision(5) << temp0(i)
              << "    dens=" << std::scientific << std::setw(13) << std::setprecision(5) << dens (i)<< std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
      }
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        temp1(i) = pcoolfunc->BracketRoot(dens(i),temp0(i),dt,zmet(i));
        temp2(i) = pcoolfunc->FindRoot(dens(i),temp0(i),temp1(i),dt,zmet(i));
      }
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        dener(i)         = dens(i)*(temp2(i)-temp0(i))/g1;
        cons(IEN,k,j,i) += dener(i);
        if (DUAL_ENERGY)
          cons(IIE,k,j,i) += dener(i);
      }
    }
  }
  dens.DeleteAthenaArray();
  temp0.DeleteAthenaArray();
  temp1.DeleteAthenaArray();
  temp2.DeleteAthenaArray();
  zmet.DeleteAthenaArray();
  dener.DeleteAthenaArray();
  dtcool.DeleteAthenaArray();
  edot.DeleteAthenaArray();
  sign.DeleteAthenaArray();

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
  Real g1     = pmb->peos->GetGamma()-1.0;
  Real dtcool = HUGE_NUMBER;// allows it to grow

  AthenaArray<Real> dens, temp0, zmet, w;
  w.InitWithShallowCopy(pmb->phydro->w);
  dens.NewAthenaArray(w.GetDim1());
  temp0.NewAthenaArray(w.GetDim1());
  zmet.NewAthenaArray(w.GetDim1());

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        dens(i)  = w(IDN,k,j,i); 
        if (DUAL_ENERGY) 
          temp0(i) = w(IGE,k,j,i)/dens(i); // IGE is pressure
        zmet(i)  = w(NHYDRO-NSCALARS,k,j,i);
      } 
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real dttemp   = coolsafe*temp0(i)/(fabs(pcoolfunc->HeatCoolFunc(dens(i),temp0(i),zmet(i)))+1e-60);
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
        dtcool   = std::min(dtcool,dttemp); // for next, not current timestep. Hence, has to be aggressive.
      }
    }
  }
  dens.DeleteAthenaArray();
  temp0.DeleteAthenaArray();
  zmet.DeleteAthenaArray();

  return dtcool;
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
  Real myVel = 0.0;
  Real pos  = 0.0;
  Real cellsize = pm->mesh_size.x1max/pm->mesh_size.nx1;
  Real posUp = 0.2*pm->mesh_size.x1max;
  Real posLow = 0.1*pm->mesh_size.x1max;// - 15.0*cellsize;
  Real velMax = 0.0, velAve=0.0, vol=0.0;
  Real pVelMax = 0.0;   
  while (pmb != NULL) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          pos = pmb->pcoord->x1v(i);
          if ((pos<=posUp) && (pos>=posLow)) {

            velMax = std::max( pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i), velMax);
            
            //pVelMax = std::max(std::sqrt( pmb->phydro->u(IEN,k,j,i)/pmb->phydro->u(IDN,k,j,i)), velMax);
            velAve += pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i)*pmb->pcoord->GetCellVolume(k,j,i);
            vol += pmb->pcoord->GetCellVolume(k,j,i);
          }
                      
        }
      }
    }
    pmb = pmb->next;
  }
#ifdef MPI_PARALLEL
  Real arr[2];
  arr[0] = velAve;
  arr[1] = vol;
  MPI_Allreduce(MPI_IN_PLACE,&velMax,1,MPI_ATHENA_REAL,MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&arr,2,MPI_ATHENA_REAL,MPI_SUM,
                MPI_COMM_WORLD);
  velAve = arr[0];
  vol    = arr[1];
#endif

  velAve = velAve/vol;
    
  //myVel = std::max(velMax,pVelMax)*(pm->GridData(3)/(posLow));
  myVel = velAve;
  if ((myVel <=0.0)) {
    myVel = 0.0;
  }

  // different attempt:
  //Real tref = 0.05*xMax/pparam->Vwind();
  //myVel = std::min(pm->time*pparam->Vwind()/tref,pparam->Vwind());

  if (iprof <= -2) {
    pm->GridData(2) = vexp;
  } else {
    pm->GridData(2) = 2.0*myVel;
  }
  //fprintf(stdout,"[UpdateGridData]: vel = %17.9e xmax = %17.9e\n",pm->GridData(2),pm->GridData(3));
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
 // }
      
  if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) 
    EnrollUserBoundaryFunction(INNER_X1,InnerX1_Wind);
  if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) 
    EnrollUserBoundaryFunction(OUTER_X1,OuterX1_HydroStat);

  x1rat      = pin->GetOrAddReal("mesh","x1rat",1.0);
  nx1        = pin->GetInteger("mesh","nx1");
  if (x1rat < 0.0) 
    EnrollUserMeshGenerator(X1DIR,PowerGridX1);

  if (EXPANDING_ENABLED) {
    SetGridData(4);
    EnrollGridDiffEq(WallVel);
    EnrollCalcGridData(UpdateGridData);
    
    GridData(0) = mesh_size.x1min;
    GridData(1) = 1; 
    GridData(2) = 0.0;
    GridData(3) = mesh_size.x1max; 
  }

  gam      = pin->GetReal("hydro","gamma");
  gm1      = gam-1.0;
  icool    = pin->GetOrAddReal("problem","icool",0);
  coolsafe = pin->GetOrAddReal("problem","coolsafe",0.05);
  if (icool > 0) {
    EnrollUserExplicitSourceFunction(HeatCool);
    EnrollUserTimeStepFunction(HeatCoolTimeStep);
  }

  iprof   = pin->GetOrAddReal("problem","iprof",0); // 0: isothermal, 1: beta-model
  if (iprof <= -2) {
    vexp = pin->GetReal("problem","vexp");
  } else if (iprof == 1) {
    EnrollStaticGravPotFunction(BetaPotential);
  } else if (iprof == 2) {
    EnrollStaticGravPotFunction(NFWPotential);
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


  if (iprof == -3) {
    // need to integrate this all the way from the start. Can use  :-( 
    yarr0(0) = 1.0; // density
    yarr0(1) = pparam->Vwind(); // velocity
    yarr0(2) = 0.1; // pressure
    Real dr;
    Real r;
    for (int i=is; i<=ie; ++i) {
      r  = pco->x1f(i);
      dr = pco->x1f(i+1)-r;
      GetSteadyStateAdbWind(r,yarr0,dr,yarr1);
      for (int l=0; l<3; ++l) yarr0(l) = yarr1(l);
    } // This ends up with the analytic profile on the outer boundary wall
    dr  = pco->x1v(ie+1)-pco->x1f(ie+1);
    r   = pco->x1f(ie+1);
    GetSteadyStateAdbWind(r,yarr0,dr,yarr1);
    for (int i=1; i<=ngh; i++) {
      prim(IDN,ks,js,ie+i) = yarr1(0);
      prim(IVX,ks,js,ie+i) = yarr1(1);
      prim(IVY,ks,js,ie+i) = 0.0;
      prim(IVZ,ks,js,ie+i) = 0.0;
      prim(IPR,ks,js,ie+i) = yarr1(2);
      if (DUAL_ENERGY)
        prim(IGE,ks,js,ie+i) = yarr1(2);
      if (NSCALARS == 2) {
        prim(NHYDRO-NSCALARS  ,ks,js,ie+i) = 0.1; 
        prim(NHYDRO-NSCALARS+1,ks,js,ie+i) = 0.0; // ambient
      }  
      r  = pco->x1v(ie+i);
      dr = pco->x1v(ie+i+1)-pco->x1v(ie+i); 
      for (int l=0; l<3; ++l) yarr0(l) = yarr1(l);
      GetSteadyStateAdbWind(r,yarr0,dr,yarr1);
    }
    return;
  }

  if (iprof == -4) {
    for (int i=1; i<=ngh; i++) {
      Real r = pco->x1v(ie+i);
      prim(IDN,ks,js,ie+i) = HaloProfile(r);
      prim(IVX,ks,js,ie+i) = 0.0;
      prim(IVY,ks,js,ie+i) = 0.0;
      prim(IVZ,ks,js,ie+i) = 0.0;
      prim(IPR,ks,js,ie+i) = 1.0;
      if (DUAL_ENERGY)
        prim(IGE,ks,js,ie+i) = 1.0;
      if (NSCALARS == 2) {
        prim(NHYDRO-NSCALARS  ,ks,js,ie+i) = 0.1;
        prim(NHYDRO-NSCALARS+1,ks,js,ie+i) = 0.0; // ambient
      }
    }
    return;
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real r = pco->x1v(ie+i);
        Real d = HaloProfile(r);
        prim(IDN,k,j,ie+i) = d;
        prim(IPR,k,j,ie+i) = d*pparam->Temp();
        if (DUAL_ENERGY) 
          prim(IGE,k,j,ie+i) = d*pparam->Temp();
        prim(IVX,k,j,ie+i) = 0.0;
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = 0.0;
        if (NSCALARS == 2) {
          prim(NHYDRO-NSCALARS  ,k,j,ie+i) = 0.1; // wind
          prim(NHYDRO-NSCALARS+1,k,j,ie+i) = 1.0; // ambient 
        }
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

  if (iprof == -3) {
    yarr0(0) = 1.0; // density
    yarr0(1) = pparam->Vwind(); // velocity
    yarr0(2) = 0.1; // pressure
    Real dr  = -(pco->x1f(is)-pco->x1v(is-1));
    Real r   = pco->x1f(is);
    GetSteadyStateAdbWind(r,yarr0,dr,yarr1);
    for (int i=1; i<=ngh; i++) {
      prim(IDN,ks,js,is-i) = yarr1(0);
      prim(IVX,ks,js,is-i) = yarr1(1);
      prim(IVY,ks,js,is-i) = 0.0;
      prim(IVZ,ks,js,is-i) = 0.0;
      prim(IPR,ks,js,is-i) = yarr1(2);
      if (DUAL_ENERGY)
        prim(IGE,ks,js,is-i) = yarr1(2);
      if (NSCALARS == 2) {
        prim(NHYDRO-NSCALARS  ,ks,js,is-i) = 1.0;
        prim(NHYDRO-NSCALARS+1,ks,js,is-i) = 0.0; // ambient
      }
      r  = pco->x1v(is-i);
      dr = -(pco->x1v(is-i)-pco->x1v(is-(i+1))); 
      for (int l=0; l<3; ++l) yarr0(l) = yarr1(l);
      GetSteadyStateAdbWind(r,yarr0,dr,yarr1);
    }
    return;
  }

  if (iprof == -4) {
    for (int i=1; i<=ngh; i++) {
      Real r = pco->x1v(is-i);
      prim(IDN,ks,js,is-i) = HaloProfile(r);
      prim(IVX,ks,js,is-i) = 0.0;
      prim(IVY,ks,js,is-i) = 0.0;
      prim(IVZ,ks,js,is-i) = 0.0;
      prim(IPR,ks,js,is-i) = 1.0;
      if (DUAL_ENERGY)
        prim(IGE,ks,js,is-i) = 1.0;
      if (NSCALARS == 2) {
        prim(NHYDRO-NSCALARS  ,ks,js,is-i) = 1.0; 
        prim(NHYDRO-NSCALARS+1,ks,js,is-i) = 0.0; // ambient
      }
    }
    return;
  }

  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        Real r = pco->x1v(is-i);
        Real d = WindProfile(r);
        prim(IDN,k,j,is-i) = d;
        prim(IPR,k,j,is-i) = d*pparam->Temp();
        if (DUAL_ENERGY) 
          prim(IGE,k,j,is-i) = d*pparam->Temp();
        prim(IVX,k,j,is-i) = pparam->Vwind();
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;
        if (NSCALARS == 2) {
          prim(NHYDRO-NSCALARS  ,k,j,is-i) = 1.0; // wind
          prim(NHYDRO-NSCALARS+1,k,j,is-i) = 0.0; // ambient 
        }
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
//  \brief Generates grid following r_i = r_0*delta**i
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
//! \fn void DdensDr(const Real r, const Real dens)
//  \brief Derivative for RHS of RK4.
//    Constants hardwired;
//========================================================================================

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

  // cooling
  pcoolfunc = new CoolingFunction();

  iprof     = pin->GetOrAddReal("problem","iprof",0); // 0: isothermal, 1: beta-model, 2: NFW
  icool     = pin->GetOrAddReal("problem","icool",0); // 0: no cooling, 1: power-law cooling, 2: WSS09 cooling
  // potential parameters
  Real vesc = pin->GetReal("problem","vesc"); // km/s 
  Real temp = pin->GetReal("problem","temp0");
  Real mdot = pin->GetReal("problem","mdot"); // Msol/yr
  Real vwind= pin->GetReal("problem","vwind"); // km/s
  Real n0   = pin->GetReal("problem","n0"); // cm^(-3)
  rmin = pin->GetReal("mesh","x1min"); 
  Real rmax = pin->GetReal("mesh","x1max"); 
  pparam    = new Parameters(vesc,temp,mdot,vwind,n0,rmin,rmax);

  // wind parameters
  fwind   = pin->GetOrAddReal("problem","fwind",0.1);
  gamma   = peos->GetGamma();
  gm1     = gamma - 1.0;
  csound2 = pparam->Temp();

  if (icool == 1) {
    pcoolfunc = new CoolingFunction();
  } else if (icool == 2) {
    pcoolfunc = new CoolingFunction("/nas/longleaf/home/fheitsch/cooling/hm12/data/z_0.000.hdf5");
  }

  // constant wind test case (compare to analytic steady-state wind solution p55)
  if (iprof == -3) {
    k1.NewAthenaArray(3);
    k2.NewAthenaArray(3);
    k3.NewAthenaArray(3);
    k4.NewAthenaArray(3);
    yarr0.NewAthenaArray(3);
    yarr1.NewAthenaArray(3);
    ytemp_.NewAthenaArray(3);
    // at cell wall
    yarr0(0) = 1.0; // density
    yarr0(1) = pparam->Vwind(); // velocity
    yarr0(2) = 0.1; // pressure
    Real dr  = pcoord->x1v(is)-pcoord->x1f(is); 
    Real r   = pcoord->x1f(is);
    GetSteadyStateAdbWind(r,yarr0,dr,yarr1);
    for (int i=is; i<=ie; i++) {
      phydro->u(IDN,ks,js,i) = yarr1(0);
      phydro->u(IM1,ks,js,i) = phydro->u(IDN,ks,js,i)*yarr1(1);
      phydro->u(IM2,ks,js,i) = 0.0;
      phydro->u(IM3,ks,js,i) = 0.0;
      phydro->u(IEN,ks,js,i) = yarr1(2)/gm1 + 0.5*SQR(phydro->u(IM1,ks,js,i))/phydro->u(IDN,ks,js,i);
      if (DUAL_ENERGY)
        phydro->u(IIE,ks,js,i) = yarr1(2)/gm1;
      if (NSCALARS == 2) {
        phydro->u(NHYDRO-NSCALARS  ,ks,js,i) = phydro->u(IDN,ks,js,i); // wind
        phydro->u(NHYDRO-NSCALARS+1,ks,js,i) = 0.0; // ambient
      }
      r  = pcoord->x1v(i);
      dr = pcoord->x1v(i+1)-pcoord->x1v(i); 
      for (int l=0; l<3; ++l) yarr0(l) = yarr1(l);
      GetSteadyStateAdbWind(r,yarr0,dr,yarr1);
    }
    return;
  }

  if (iprof == -4) {
    for (int i=is; i<=ie; ++i) {
      Real r = pcoord->x1v(i);
      phydro->u(IDN,ks,js,i) = HaloProfile(r);
      phydro->u(IM1,ks,js,i) = 0.0;
      phydro->u(IM2,ks,js,i) = 0.0;
      phydro->u(IM3,ks,js,i) = 0.0;
      phydro->u(IEN,ks,js,i) = 1.0/gm1;
      if (DUAL_ENERGY)
        phydro->u(IIE,ks,js,i) = 1.0/gm1;
      if (NSCALARS == 2) {
        phydro->u(NHYDRO-NSCALARS  ,ks,js,i) = phydro->u(IDN,ks,js,i); // wind
        phydro->u(NHYDRO-NSCALARS+1,ks,js,i) = 0.0; // ambient
      }
    }
    return;
  }
  
  // only spherical coordinates
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real r     = pcoord->x1v(i); 
        Real den   = HaloProfile(r);
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = den*pparam->Temp()/gm1 + 0.5*(  SQR(phydro->u(IM1,k,j,i))
                                                               + SQR(phydro->u(IM2,k,j,i))
                                                               + SQR(phydro->u(IM3,k,j,i)))
                                                             / phydro->u(IDN,k,j,i);
          if (DUAL_ENERGY)
            phydro->u(IIE,k,j,i) = den*pparam->Temp()/gm1;
        }
        if (NSCALARS == 2) {
          phydro->u(NHYDRO-NSCALARS  ,k,j,i) = 0.1*den; // metallicity
          phydro->u(NHYDRO-NSCALARS+1,k,j,i) = den; // ambient
        }
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
        Real xtest = pmb->pcoord->x1f(pmb->is);
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
              stop_this();
            } 
          } else {
            Real eint = pmb->phydro->u(IEN,k,j,i)-0.5*(SQR(pmb->phydro->u(IM1,k,j,i))+SQR(pmb->phydro->u(IM2,k,j,i))+SQR(pmb->phydro->u(IM3,k,j,i)))
                                                     /pmb->phydro->u(IDN,k,j,i);
            if (   isnan(pmb->phydro->u(IEN,k,j,i))
                || isnan(pmb->phydro->u(IDN,k,j,i))
                || (pmb->phydro->u(IEN,k,j,i) <= 0.0)
                || (eint <= 0.0)
                || (pmb->phydro->u(IDN,k,j,i) <= 0.0)) {
              std::cout << "[UserWorkInLoop]: Warning: i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
                        << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IDN,k,j,i)
                        << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM1,k,j,i)
                        << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM2,k,j,i)
                        << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM3,k,j,i)
                        << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IEN,k,j,i)
                        << " ei=" << std::scientific << std::setw(11) << std::setprecision(3) << eint
                        << std::endl;
              stop_this();
            }
          }
          Real dx1  = pmb->pcoord->dx1f(i);
          Real x1  = pmb->pcoord->x1v(i);
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
          if (DUAL_ENERGY) {
            ener[6] = u[IIE];
          }
          ener[3] = ener[2]-ener[4]-ener[5];
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
          Real temp = gm1*ener[2]/u[IDN];
          Real zmet = u[NHYDRO-NSCALARS]/u[IDN];
          lengrat[0] = std::min(lengrat[0],temp*std::sqrt(temp)/(dx1*fabs(pcoolfunc->HeatCoolFunc(u[IDN],temp,zmet)))); // cooling length
          lengrat[1] = std::min(lengrat[1],std::sqrt(PI*temp/u[IDN])/dx1); // Jeans length
          //fprintf(stdout,"[UserWorkInLoop]: i=%2i d=%13.5e v=%13.5e e=%13.5e\n",i,u[IDN],u[IM1]/u[IDN],u[IEN]);
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
  
  if (qmin[3] <= 0.0) {
    std::cout << "[UserWorkInLoop]: eint < 0" << std::endl;
    stop_this();
  }

  return;
}
