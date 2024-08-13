#ifndef EXPANSION_EXPANSION_HPP_
#define EXPANSION_EXPANSION_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file expansion.hpp
//  \brief definitions for Expansion class

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
//#include "../task_list/task_list.hpp"

class MeshBlock;
class ParameterInput;
struct IntegratorWeight;

//! \class Expansion
//  \brief Expanding Grid information and edits

class Expansion {
friend class Field;
friend class Hydro;
friend class Mesh;
friend class Reconstruction;
public:
  Expansion(MeshBlock *pmb, ParameterInput *pin);
  ~Expansion();

  //Boolean Direction variables
  bool x1Move;
  bool x2Move;
  bool x3Move;


  // Expansion Data
  AthenaArray<Real> vol;

  //Integration Registers
  AthenaArray<Real> x1_0, x2_0, x3_0;
  AthenaArray<Real> x1_1, x2_1, x3_1;
  AthenaArray<Real> x1_2, x2_2, x3_2;

  AthenaArray<Real> expFlux[3];  // face-averaged flux vector
  AthenaArray<Real> vf[3];  // face-averaged wall velocity
  AthenaArray<Real> vv[3];  // cell-centered expansion velocity (for MHD)

  Real mydt;
  int il, iu, jl, ju, kl, ku, ng; //With Ghost cells
  int ie,is,je,js,ke,ks; //Without ghost cells

  void WeightedAveX(const int low, const int up, AthenaArray<Real> &x_out, AthenaArray<Real> &x_in1, AthenaArray<Real> &x_in2, const Real wght[3]);
  void IntegrateWalls(Real dt);
  void ExpansionSourceTerms(const Real dt, const AthenaArray<Real> *flux, const AthenaArray<Real> &prim, AthenaArray<Real> &cons);
  void RescaleField(const Real dt, FaceField &b_out);
  void GridEdit(MeshBlock *pmb, bool lastStage);
  void UpdateVelData(MeshBlock *pmb,Real time, Real dt);
  Real GridTimeStep(MeshBlock *pmb);

private:
  MeshBlock* pmy_block;    // ptr to MeshBlock containing this Expansion
  // Scratch arrays
  AthenaArray<Real> face_area_old_;





};
#endif // EXPANSION_EXPANSION_HPP_
