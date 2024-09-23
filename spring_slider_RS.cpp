/*

make -f Makefile.basic.user app PETSC_DIR=/projappl/project_465001082/dmay/software/petsc-dev-git PETSC_ARCH=arch-cray-c-debug-rocm-hip-tandem-vanil

salloc --account=project_465001082 --partition=dev-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --time=00:10:00

srun ./app -V0 1e-06 -f0 0.6 -a 0.015 -b 0.02 -eta 4500000.0 -L 0.1 -sn 50000000.0 -Vinit 1e-09 -Vp 1e-08 -k 500000 -yield_point_init 10 -output out.txt -final_time 15768000000.0 -npoints 1 -vec_type hip -device_view
*/


#include <petscdm.h>
#include <petsc/private/tsimpl.h>
#include <fstream>
#include <iomanip>
#include <hip/hip_runtime.h>

#include "DieterichRuinaAgeing.h"
#include "params.h"

#define DIM 2

extern "C" PetscErrorCode RHSFunction_spring_slider_batch_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx);

void pack_vals(DieterichRuinaAgeing *law, Params *p, int idx)
{
  law->V0    = p->V0[idx];
  law->b     = p->b[idx];
  law->f0    = p->f0[idx];
  law->a     = p->a[idx];
  law->eta   = p->eta[idx];
  law->L     = p->L[idx];
  law->sn    = p->sn[idx];
  law->Vinit = p->Vinit[idx];
  law->Vp    = p->Vp[idx];
  law->k     = p->k[idx];
  law->yield_point_init = p->yield_point_init[idx];
}

static PetscErrorCode RHSFunction_spring_slider_batch(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscScalar           *f;
  const PetscScalar     *u;
  double                D, psi, tau, V;
  Context               *c = (Context*)ctx;
  DieterichRuinaAgeing  *alwa = c->statelaw;
  PetscInt              npoints, len, nvar_per_point, k;
  Params                *p = c->host;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));
  nvar_per_point = (DIM -1) + 1;
  npoints = len / nvar_per_point;

  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));

  for (k=0; k<npoints; k++) {
    D = (double)PetscRealPart(u[nvar_per_point*k+0]);
    psi = (double)PetscRealPart(u[nvar_per_point*k+1]);

    // pack
    pack_vals(alwa, p, k);

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
  Context              *c;
  Params               *p = NULL;
  DieterichRuinaAgeing *alwa = NULL;
  std::ofstream        *out_file;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts,&ctx_app));
  out_file = static_cast<std::ofstream*>(ctx_app);
  PetscCall(TSGetRHSFunction(ts, NULL, NULL, &ctx));
  c = (Context*)ctx;
  alwa = c->statelaw;
  p = c->host;
  pack_vals(alwa, p, 0);

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
  double       final_time;
  TS           ts; /* ODE integrator */
  Vec          U;  /* solution will be stored here */
  PetscMPIInt  commsize;
  PetscInt     npoints = 1, len, nvar_per_point, k;
  PetscScalar  *u = NULL;
  TSAdapt      adapt;
  PetscBool    found;
  PetscReal    arg_r;
  char         fname[PETSC_MAX_PATH_LEN];
  MPI_Comm     comm;
  Context      *ctx = NULL;

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
  {
    double tau_init, psi_init;

    tau_init = alwa.k * alwa.yield_point_init;
    psi_init = alwa.psi_init(tau_init);
    std::cout << "Initial state:" << psi_init <<std::endl;
  }
  std::cout << "final_time = " << final_time << std::endl;

  std::cout << "npoints = " << (int)npoints << std::endl;

  //PetscCalloc1(1,&ctx);
  PetscMalloc(sizeof(struct _p_Context),&ctx);
  ctx->statelaw = &alwa;
  ctx->out_file = &out_file;

  PetscMalloc(sizeof(struct _p_Params),&ctx->host);
  ctx->host->npoints = npoints;
  PetscCalloc1(npoints,&ctx->host->V0);
  PetscCalloc1(npoints,&ctx->host->b);
  PetscCalloc1(npoints,&ctx->host->f0);
  PetscCalloc1(npoints,&ctx->host->a);
  PetscCalloc1(npoints,&ctx->host->eta);
  PetscCalloc1(npoints,&ctx->host->L);
  PetscCalloc1(npoints,&ctx->host->sn);
  PetscCalloc1(npoints,&ctx->host->Vinit);
  PetscCalloc1(npoints,&ctx->host->Vp);
  PetscCalloc1(npoints,&ctx->host->k);
  PetscCalloc1(npoints,&ctx->host->yield_point_init);
  for (k=0; k<npoints; k++) {
    ctx->host->V0[k] = alwa.V0;
    ctx->host->b[k] = alwa.b;
    ctx->host->f0[k] = alwa.f0;
    ctx->host->a[k] = alwa.a;
    ctx->host->eta[k] = alwa.eta;
    ctx->host->L[k] = alwa.L;
    ctx->host->sn[k] = alwa.sn;
    ctx->host->Vinit[k] = alwa.Vinit;
    ctx->host->Vp[k] = alwa.Vp;
    ctx->host->k[k] = alwa.k;
    ctx->host->yield_point_init[k] = alwa.yield_point_init;
  }

  int ierr;
  size_t nbytes = sizeof(double)*npoints;
  PetscMalloc(sizeof(struct _p_Params),&ctx->device);
  ctx->device->npoints = ctx->host->npoints;
  ierr = hipMalloc(&ctx->device->yield_point_init, nbytes);
  ierr = hipMemcpy(ctx->device->yield_point_init, ctx->host->yield_point_init, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->k, nbytes);
  ierr = hipMemcpy(ctx->device->k, ctx->host->k, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->Vp, nbytes);
  ierr = hipMemcpy(ctx->device->Vp, ctx->host->Vp, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->Vinit, nbytes);
  ierr = hipMemcpy(ctx->device->Vinit, ctx->host->Vinit, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->sn, nbytes);
  ierr = hipMemcpy(ctx->device->sn, ctx->host->sn, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->L, nbytes);
  ierr = hipMemcpy(ctx->device->L, ctx->host->L, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->eta, nbytes);
  ierr = hipMemcpy(ctx->device->eta, ctx->host->eta, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->a, nbytes);
  ierr = hipMemcpy(ctx->device->a, ctx->host->a, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->f0, nbytes);
  ierr = hipMemcpy(ctx->device->f0, ctx->host->f0, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->b, nbytes);
  ierr = hipMemcpy(ctx->device->b, ctx->host->b, nbytes, hipMemcpyHostToDevice);

  ierr = hipMalloc(&ctx->device->V0, nbytes);
  ierr = hipMemcpy(ctx->device->V0, ctx->host->V0, nbytes, hipMemcpyHostToDevice);

  /* initial condition */
  PetscCall(VecCreate(comm, &U));
  PetscCall(VecSetSizes(U, ((DIM -1) + 1) * npoints, PETSC_DETERMINE)); // slip-rate + state
  PetscCall(VecSetType(U, "standard"));
  PetscCall(VecSetFromOptions(U));
  PetscCall(VecSetUp(U));

  PetscCall(VecGetLocalSize(U,&len));
  nvar_per_point = (DIM -1) + 1;
  PetscCall(VecGetArray(U, &u));
  for (k=0; k<npoints; k++) {
    double tau_init, psi_init;

    tau_init = ctx->host->k[k] * ctx->host->yield_point_init[k];
    psi_init = alwa.psi_init(tau_init);

    u[nvar_per_point * k + 0] = 0.0;
    u[nvar_per_point * k + 1] = psi_init;
  }
  PetscCall(VecRestoreArray(U, &u));







  PetscCall(TSCreate(comm, &ts));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));

  PetscCall(TSSetApplicationContext(ts, static_cast<void*>(&out_file)));

  {
    PetscBool isseq;
    PetscCall(PetscObjectTypeCompare((PetscObject)U,"seq",&isseq));
    if (isseq) {
	     PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction_spring_slider_batch, static_cast<void*>(ctx)));
	  } else {
      PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction_spring_slider_batch_hip, static_cast<void*>(ctx)));
	  }
  }

  PetscCall(TSSetSolution(ts, U));
  PetscCall(TSSetMaxTime(ts, final_time));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts, 1.0e-10));
  /* The adaptive time step controller could take very
     large timesteps. An upper limit is enforced here to avoid this. */
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptSetStepLimits(adapt, 0.0, 2000000.0));

  //PetscCall(TSSetPostStep(ts, ts_soln_view));

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
