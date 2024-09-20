#ifndef params_h
#define params_h

struct _p_Params {
  DieterichRuinaAgeing *statelaw;
  std::ofstream  *out_file;
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
typedef struct _p_Params *Params;

#endif
