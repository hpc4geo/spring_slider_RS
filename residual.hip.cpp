/*
hipcc -c residual.hip -I/users/maydave2/software/petsc-3.21.5/include -I/users/maydave2/software/petsc-3.21.5/arch-cray-c-debug-rocm-hip-vanil/include -I/opt/cray/pe/mpich/8.1.29/ofi/crayclang/17.0/include -I/opt/rocm-6.0.3/include
*/


#include <petsc.h>
#include <hip/hip_runtime.h>
#include "DieterichRuinaAgeing.h"
#include "params.h"

#define HCC_ENABLE_PRINTF

#define DIM 2


#define GRID_MAX 1024
#define THREAD_MAX 1024


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

__device__ __inline__ double _rs_residual_F(DieterichRuinaAgeing *law, double tau, double psi, double V)
{
            double sn = law->sn;
            double V0 = law->V0;
            double a = law->a;
            return tau - law_F(sn, V, psi, a, V0) - law->eta * V;
}

__device__ __inline__ double zeroIn_hip(double a, double b, DieterichRuinaAgeing *law, double tau, double psi, double tol) {

   double eps = std::numeric_limits<double>::epsilon();
    double Fa = _rs_residual_F(law,tau,psi, a);
    if (Fa == 0.0) {
        return a;
    }
    double Fb = _rs_residual_F(law,tau,psi, b);

    if (std::isnan(Fa) || std::isinf(Fa)) {
        //function_nan_inf(a, Fa);
        return NAN;
    }
    if (std::isnan(Fb) || std::isinf(Fb)) {
        //function_nan_inf(b, Fb);
        return NAN;

    }
    if (Fb != 0.0 && std::copysign(Fa, Fb) == Fa) {
        //std::cout << "a - " << a << std::endl;
        //std::cout << "Fa - " << Fa << std::endl;
        //std::cout << "b - " << b << std::endl;
        //std::cout << "Fb - " << Fb << std::endl;
        //throw std::logic_error("F(a) and F(b) must have different sign.");
        return NAN;
    }
    double c = a;
    double Fc = Fa;
    double d = b - a;
    double e = d;
    while (Fb != 0.0) {
        if (std::copysign(Fb, Fc) == Fb) {
            c = a;
            Fc = Fa;
            d = b - a;
            e = d;
        }
        if (std::fabs(Fc) < std::fabs(Fb)) {
            a = b;
            b = c;
            c = a;
            Fa = Fb;
            Fb = Fc;
            Fc = Fa;
        }
        // Convergence test
        double xm = 0.5 * (c - b);
        double tol1 = 2.0 * eps * std::fabs(b) + 0.5 * tol;
        if (std::fabs(xm) <= tol1 || Fb == 0.0) {
            break;
        }
        if (std::fabs(e) < tol1 || std::fabs(Fa) <= std::fabs(Fb)) {
            // bisection
            d = xm;
            e = d;
        } else {
            double s = Fb / Fa;
            double p, q;
            if (a != c) {
                // linear interpolation
                q = Fa / Fc;
                double r = Fb / Fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            } else {
                // inverse quadratic interpolation
                p = 2.0 * xm * s;
                q = 1.0 - s;
            }
            if (p > 0) {
                q = -q;
            } else {
                p = -p;
            }
            if (2.0 * p < 3.0 * xm * q - std::fabs(tol1 * q) && p < std::fabs(0.5 * e * q)) {
                e = d;
                d = p / q;
            } else {
                // bisection
                d = xm;
                e = d;
            }
        }
        a = b;
        Fa = Fb;
        if (std::fabs(d) > tol1) {
            b += d;
        } else {
            b += std::copysign(tol1, xm);
        }
        Fb = _rs_residual_F(law,tau,psi, b);
        if (std::isnan(Fb) || std::isinf(Fb)) {
            //function_nan_inf(b, Fb);
            return NAN;
        }
    }
    return b;
}


__device__ __inline__ void pack_vals_hip(DieterichRuinaAgeing *law, Params *p, int idx)
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


__device__ __inline__ double law_slip_rate(double tau, double psi, DieterichRuinaAgeing *law)
{
        double V = 0.0;
        double alpha = 0.0;
        double beta = tau / law->eta;
        if (alpha > beta) {
            double t1 = beta;
            beta = alpha;
            alpha = t1;
        }

        //auto fF = [law, &tau, &psi](double V) -> double {
        //    double sn = law->sn;
        //    double V0 = law->V0;
        //    double a = law->a;
        //    return tau - law_F(sn, V, psi, a, V0) - law->eta * V;
        //};

        //try {
        //    V = zeroIn(alpha, beta, fF);
        //} catch (std::exception const&) {
        //    throw;
        //}

	V = zeroIn_hip(alpha, beta, law, tau, psi, 1.0e-12);


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
  int k = threadIdx.x + blockIdx.x*blockDim.x;
  double D, psi, tau, V;
  int nvar_per_point = (DIM-1) + 1;

  if (k > npoints) return;
  printf("tindex %d\n",k);

    D = (double)PetscRealPart(u[nvar_per_point*k+0]);
    psi = (double)PetscRealPart(u[nvar_per_point*k+1]);

    printf("    d_alwa %p | d_p %p \n",law, p);

    printf("p->V0[k] %+1.10e\n",p->V0[k]);

#if 0
    // pack
    pack_vals_hip(law, p, k);
//#if 0

    tau = law->k * ((law->Vp * ((double)time) + law->yield_point_init) - D);
    V = law_slip_rate(tau, psi, law);

    f[nvar_per_point*k+0] = (PetscScalar)V;
    f[nvar_per_point*k+1] = (PetscScalar)law_state_rhs(V, psi, law);
#endif
}

extern "C" {

long int printd3(dim3 *b) {
  printf("(%ld %ld %ld) -> max items %ld\n",(long)b->x,(long)b->y,(long)b->z, (long)b->x*b->y*b->z);
  return (long int)b->x*b->y*b->z;
}


void build_sizes(long int N,dim3 *b, dim3 *t)
{
  b->x = b->y = b->z = 1;
  t->x = t->y = t->z = 1;

  if (N < GRID_MAX * THREAD_MAX) {
    printf("1d ->\n");

    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = (N + GRID_MAX-1)/GRID_MAX; b->y = 1; b->z = 1;

    return;
  }

  if (N < GRID_MAX * GRID_MAX * THREAD_MAX) {
    printf("2d ->\n");

    long int bJ = (N-1)/(GRID_MAX*GRID_MAX);
    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = GRID_MAX; b->y = bJ+1; b->z = 1;

    return;
  }


  long int Nt = (long int)(N/THREAD_MAX);
  if ( Nt < GRID_MAX * GRID_MAX * GRID_MAX) {
    printf("3d ->\n");

    long int bK = (long int) ( Nt/(GRID_MAX*GRID_MAX) );
    long int N2d = Nt - GRID_MAX * bK;
    long int bJ = N2d/GRID_MAX;
    t->x = THREAD_MAX; t->y = 1; t->z = 1;
    b->x = GRID_MAX; b->y = GRID_MAX; b->z = bK+1;

    return;
  }
}

PetscErrorCode RHSFunction_spring_slider_batch_hip(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscScalar           *f;
  const PetscScalar     *u;
  double                D, psi, tau, V;
  Params                p = (Params)ctx;
  DieterichRuinaAgeing  *alwa = p->statelaw;
  PetscInt              npoints, len, nvar_per_point, k;
  int ierr;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U,&len));
  nvar_per_point = (DIM -1) + 1;
  npoints = len / nvar_per_point;


  dim3 blocks, threads;
  build_sizes(len, &blocks, &threads);
  {
    long int bm,tm;
    printf("blocks "); bm = printd3(&blocks);
    printf("threads "); tm = printd3(&threads);
    printf("max %ld | N %ld\n",bm * tm,(long int)len);

  }

  PetscCall(VecHIPGetArrayRead(U, &u));
  PetscCall(VecHIPGetArray(F, &f));

  alwa = p->statelaw;

  DieterichRuinaAgeing  *d_alwa;
  Params                d_p;

  ierr = hipMalloc(&d_alwa, sizeof(DieterichRuinaAgeing*));
  ierr = hipMemcpy(d_alwa, alwa, sizeof(DieterichRuinaAgeing*), hipMemcpyHostToDevice);

  ierr = hipMalloc(&d_p, sizeof(Params));
  ierr = hipMemcpy(d_p, p, sizeof(Params), hipMemcpyHostToDevice);

  printf("alwa %p | p %p \n",alwa, p);
  rs_batch_kernel<<< blocks, threads >>>(t, d_alwa, npoints, d_p, u, f );

  PetscCall(VecHIPRestoreArrayRead(U, &u));
  PetscCall(VecHIPRestoreArray(F, &f));
  ierr = hipDeviceSynchronize();

  ierr = hipFree(d_alwa);
  ierr = hipFree(d_p);

  PetscFunctionReturn(PETSC_SUCCESS);
}


}
