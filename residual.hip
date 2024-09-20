/*
hipcc -c residual.hip -I/users/maydave2/software/petsc-3.21.5/include -I/users/maydave2/software/petsc-3.21.5/arch-cray-c-debug-rocm-hip-vanil/include -I/opt/cray/pe/mpich/8.1.29/ofi/crayclang/17.0/include -I/opt/rocm-6.0.3/include
*/


#include <petsc.h>
#include <hip/hip_runtime.h>
#include "DieterichRuinaAgeing.h"
#include "params.h"

#include "Zero_hip.h"


#define DIM 2


#define WARPS_PER_BLOCK    4

#if defined(USE_ROCM)
  #define WARP_SIZE warpSize // = 64 or 32 (Defined in hip_runtime.h)
#else
  #define WARP_SIZE 32
#endif


static __global__ void set_zero_HIP_kernel(PetscScalar *Yu, PetscInt localsize)
{
  for (PetscInt i = blockDim.x * blockIdx.x + threadIdx.x; i<localsize; i += blockDim.x * gridDim.x)
    Yu[i] = 0;
}


__device__ __inline__ void pack_vals_hip(DieterichRuinaAgeing *law, Params p, int idx)
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

__device__ __inline__ double law_state_rhs(double V, double psi, DieterichRuinaAgeing *law)
{
        return law->b * law->V0 / law->L * (exp((law->f0 - psi) / law->b) - V / law->V0);
    }


__device__ __inline__  double law_F(double sn, double V, double psi, double a, double V0)
{
        double e = exp(psi / a);
        double f = a * asinh((V / (2.0 * V0)) * e);
        double tau = sn * f;
        return tau;
    }


__device__ __inline__ double law_slip_rate(double tau, double psi, DieterichRuinaAgeing *law)
{
        double V = 0.0;
        double alpha = 0.0;
        double beta = tau / law->eta;
        if (alpha > beta)
        {
            double t1 = beta;
            beta = alpha;
            alpha = t1;
        }
        auto fF = [law, &tau, &psi](double V) -> double {
            double sn = law->sn;
            double V0 = law->V0;
            double a = law->a;
            return tau - law_F(sn, V, psi, a, V0) - law->eta * V;
        };
        //try {
        //    V = zeroIn(alpha, beta, fF);
        //} catch (std::exception const&) {
        //    throw;
        //}

        return V;
    }


static __global__ void rs_batch_kernel(
PetscReal time,
DieterichRuinaAgeing *law,
int npoints,
Params p,
const PetscScalar *u,
PetscScalar *f)
{
#if 0
  PetscInt    elidx = (blockDim.x * blockIdx.x + threadIdx.x) / 32;  // WARP_SIZE? // one warp per colored element. elidx is here the index within the same color.
  PetscInt    id_in_warp = threadIdx.x % 32;  // WARP_SIZE?

  PetscInt    c = id_in_warp % 3;
  PetscInt    b = (id_in_warp % 9) / 3;
  PetscInt    a = id_in_warp / 9;


  if (id_in_warp < Q2_NODES_PER_EL_3D) {
}
#endif
  int k, tidx_warp = threadIdx.x % 32;
  double D, psi, tau, V;
  int nvar_per_point;

  if (tidx_warp > npoints) return;

  k = tidx_warp;

    D = (double)PetscRealPart(u[nvar_per_point*k+0]);
    psi = (double)PetscRealPart(u[nvar_per_point*k+1]);

    // pack
    pack_vals_hip(law, p, k);

    tau = law->k * ((law->Vp * ((double)time) + law->yield_point_init) - D);
    V = law_slip_rate(tau, psi, law);
    f[nvar_per_point*k+0] = (PetscScalar)V;
    f[nvar_per_point*k+1] = (PetscScalar)law_state_rhs(V, psi, law);
  
}

extern "C" {

PetscErrorCode RHSFunction_spring_slider_batch_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscScalar           *f;
  const PetscScalar     *u;
  double                D, psi, tau, V;
  Params                p = (Params)ctx;
  DieterichRuinaAgeing  *alwa = p->statelaw;
  PetscInt              npoints, len, nvar_per_point, k;
  PetscMemType mt_f,mt_u;
  int ierr;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));
  nvar_per_point = (DIM -1) + 1;
  npoints = len / nvar_per_point;

  PetscCall(VecGetArrayReadAndMemType(U, &u, &mt_u));
  PetscCall(VecGetArrayAndMemType(F, &f, &mt_f));

  rs_batch_kernel<<<(npoints)/WARPS_PER_BLOCK + 1, WARPS_PER_BLOCK*32>>>(t, alwa, npoints, p, u, f );


  ierr = hipDeviceSynchronize();

  PetscCall(VecRestoreArrayReadAndMemType(U, &u));
  PetscCall(VecRestoreArrayAndMemType(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}


}
