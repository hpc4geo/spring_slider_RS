
#include <petscdm.h>
#include <petsc/private/tsimpl.h>
#include <fstream>
#include <iomanip>

#include "DieterichRuinaAgeing.h"

#define DIM 2

static PetscErrorCode RHSFunction_spring_slider_batch(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscScalar           *f;
  const PetscScalar     *u;
  double                D, psi, tau, V;
  DieterichRuinaAgeing  *alwa = static_cast<DieterichRuinaAgeing*>(ctx);
  PetscInt              npoints, len, nvar_per_point, k;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));
  nvar_per_point = (DIM -1) + 1;
  npoints = len / nvar_per_point;

  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));

  for (k=0; k<npoints; k++) {

    D = (double)PetscRealPart(u[nvar_per_point*k+0]);
    psi = (double)PetscRealPart(u[nvar_per_point*k+1]);
    tau = alwa->k * ((alwa->Vp * ((double)t) + alwa->yield_point_init) - D);
    V = alwa->slip_rate(tau, psi);

    f[nvar_per_point*k+0] = (PetscScalar)V;
    f[nvar_per_point*k+1] = (PetscScalar)alwa->state_rhs(V, psi);
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ts_soln_view(TS ts)
{
  Vec U;
  const PetscScalar *u;
  PetscInt step;
  PetscReal time;
  double D, psi, V, tau;
  void *ctx = NULL, *ctx_app = NULL;
  DieterichRuinaAgeing* alwa = NULL;
  std::ofstream  *out_file;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts,&ctx_app));
  out_file = static_cast<std::ofstream*>(ctx_app);
  PetscCall(TSGetRHSFunction(ts, NULL, NULL, &ctx));
  alwa = static_cast<DieterichRuinaAgeing*>(ctx);
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetSolution(ts, &U));
  PetscCall(VecGetArrayRead(U, &u));
  D = (double)PetscRealPart(u[0]);
  psi = (double)PetscRealPart(u[1]);
  tau = alwa->k * (alwa->Vp * time - D);
  V = alwa->slip_rate(tau, psi);
  (*out_file) << std::scientific << std::setprecision(4) << (double)time << "," << D << "," << psi << "," << V << "," << tau << std::endl;
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **argv)
{
  DieterichRuinaAgeing alwa;
  std::ofstream out_file;
  double       final_time, tau_init, psi_init;
  TS           ts; /* ODE integrator */
  Vec          U;  /* solution will be stored here */
  PetscMPIInt  commsize;
  PetscInt     npoints = 1, len, nvar_per_point, k, d;
  PetscScalar  *u = NULL;
  TSAdapt      adapt;
  PetscBool    found;
  PetscReal    arg_r;
  char         fname[PETSC_MAX_PATH_LEN];
  MPI_Comm     comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));

  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &commsize));
  if (commsize > 1) SETERRQ(comm,PETSC_ERR_SUP,"This is a serial example. MPI parallelism is not supported");

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-V0",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -V0"); }
  else { alwa.V0 = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-f0",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -f0"); }
  else { alwa.f0 = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-a",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -a"); }
  else { alwa.a = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-b",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -b"); }
  else { alwa.b = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-eta",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -eta"); }
  else { alwa.eta = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-L",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -L"); }
  else { alwa.L = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-sn",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -sn"); }
  else { alwa.sn = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-Vinit",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -Vinit"); }
  else { alwa.Vinit = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-Vp",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -Vp"); }
  else { alwa.Vp = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-k",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -k"); }
  else { alwa.k = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-yield_point_init",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -yield_point_init"); }
  else { alwa.yield_point_init = (double)arg_r; }

  found = PETSC_FALSE; PetscOptionsGetReal(NULL,NULL,"-final_time",&arg_r,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -final_time"); }
  else { final_time = (double)arg_r; }

  std::string out_file_name = std::string("");
  found = PETSC_FALSE; PetscOptionsGetString(NULL,NULL,"-output",fname,sizeof(fname),&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -output"); }
  else {  out_file_name.replace(out_file_name.begin(), out_file_name.end(), fname); }


  out_file.open(out_file_name);
  if (!out_file) {
      std::cerr << "Error opening output file" << std::endl;
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Error opening output file");
  }
  out_file << "t,D,psi,V,tau" << std::endl;

  npoints = 1;
  found = PETSC_FALSE; PetscOptionsGetInt(NULL,NULL,"-npoints",&npoints,&found);
  if (!found) { SETERRQ(comm,PETSC_ERR_SUP,"Require value be provided for -npoints"); }


  tau_init = alwa.k * alwa.yield_point_init;
  psi_init = alwa.psi_init(tau_init);

  std::cout << "Parameters:" << std::endl;
  std::cout << "V0 = " << alwa.V0 << std::endl;
  std::cout << "f0 = " << alwa.f0 << std::endl;
  std::cout << "a = " << alwa.a << std::endl;
  std::cout << "b = " << alwa.b << std::endl;
  std::cout << "eta = " << alwa.eta << std::endl;
  std::cout << "L = " << alwa.L << std::endl;
  std::cout << "sn = " << alwa.sn << std::endl;
  std::cout << "Vinit = " << alwa.Vinit << std::endl;
  std::cout << "Vp = " << alwa.Vp << std::endl;
  std::cout << "k = " << alwa.k << std::endl;
  std::cout << "Initial state:" << psi_init <<std::endl;
  std::cout << "final_time = " << final_time << std::endl;

  std::cout << "npoints = " << (int)npoints << std::endl;


  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));

  PetscCall(TSSetApplicationContext(ts, static_cast<void*>(&out_file)));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction_spring_slider_batch, static_cast<void*>(&alwa)));

  /* initial condition */
  PetscCall(VecCreate(comm, &U));
  PetscCall(VecSetSizes(U, ((DIM -1) + 1) * npoints, PETSC_DETERMINE)); // slip-rate + state
  PetscCall(VecSetFromOptions(U));
  PetscCall(VecSetUp(U));

  PetscCall(VecGetLocalSize(U,&len));
  nvar_per_point = (DIM -1) + 1;
  PetscCall(VecGetArray(U, &u));
  for (k=0; k<npoints; k++) {
    u[nvar_per_point * k + 0] = 0.0;
    u[nvar_per_point * k + 1] = psi_init;
  }
  PetscCall(VecRestoreArray(U, &u));

  PetscCall(TSSetSolution(ts, U));
  PetscCall(TSSetMaxTime(ts, final_time));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts, 1.0e-10));
  /* The adaptive time step controller could take very
     large timesteps. An upper limit is enforced here to avoid this. */
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptSetStepLimits(adapt, 0.0, 2000000.0));

  PetscCall(TSSetPostStep(ts, ts_soln_view));

  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSetUp(ts));

  PetscCall(TSSolve(ts, U));

  VecView(U,PETSC_VIEWER_STDOUT_(comm));

  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());

  out_file.close();
  return 0;
}
