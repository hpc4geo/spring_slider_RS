#ifndef params_h
#define params_h

struct _p_Params {
  int npoints;
  double *V0;
  double *b;
  double *f0;
  double *a;
  double *eta;
  double *L;
  double *sn;
  double *Vinit;
  double *Vp;
  double *k;
  double *yield_point_init;
};

typedef struct _p_Params Params;

struct _p_Context {
  std::ofstream  *out_file;
  DieterichRuinaAgeing *statelaw;
  Params         *host;
  Params         *device;
};

typedef struct _p_Context Context;

#endif
