#ifndef SPECIAL_H
#define SPECIAL_H

double nsugar_chi2_sf(int k, double x);
double nsugar_lgamma(double x);
double nsugar_normal_pdf(double x);
double nsugar_normal_cdf(double x);
double nsugar_normal_icdf(double x);
double nsugar_normal_sf(double x);
double nsugar_normal_isf(double x);
double nsugar_normal_logpdf(double x);
double nsugar_normal_logcdf(double x);
double nsugar_normal_logsf(double x);

double nsugar_beta_isf(double a, double b, double x);

double nsugar_logaddexp(double x, double y);
double nsugar_logaddexps(double x, double y, double sx, double sy);
double nsugar_logaddexpss(double x, double y, double sx, double sy,
                         double* sign);

double nsugar_logbinom(double N, double K);

#endif
