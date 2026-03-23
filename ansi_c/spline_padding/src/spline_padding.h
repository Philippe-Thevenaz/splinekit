#ifndef spline_padding_h
#define spline_padding_h

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void get_samples_to_coeff_p (
	double *data,
	int32_t dataCount,
	double *pole,
	int32_t poleCount
);

extern void get_samples_to_coeff_n (
	double *data,
	int32_t dataCount,
	double *pole,
	int32_t poleCount
);

extern void get_samples_to_coeff_w (
	double *data,
	int32_t dataCount,
	double *pole,
	int32_t poleCount
);

extern void get_samples_to_coeff_a (
	double *data,
	int32_t dataCount,
	double *pole,
	int32_t poleCount
);

extern void get_samples_to_coeff_np (
	double *data,
	int32_t dataCount,
	double *pole,
	int32_t poleCount
);

extern void get_samples_to_coeff_nn (
	double *data,
	int32_t dataCount,
	double *pole,
	int32_t poleCount
);

extern void get_samples_to_coeff_nw (
	double *data,
	int32_t dataCount,
	double *pole,
	int32_t poleCount
);

#endif /* spline_padding_h */
