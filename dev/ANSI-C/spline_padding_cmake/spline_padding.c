#include "spline_padding.h"

extern void get_samples_to_coeff_p (
	double data[],
	int32_t dataCount,
	double pole[],
	int32_t poleCount
) {
	if ((NULL == data) || (0 >= dataCount)) {
		return;
	}
	if ((NULL == pole) || (0 >= poleCount)) {
		return;
	}
	if ((0 != poleCount) && (1 != dataCount)) {
		for (int q = 0; q < poleCount; q++) {
			const double z = pole[q];
			double sigma = data[0];
			double zeta = z;
			int k = 1;
			while ((k < dataCount) && (0.0 != zeta)) {
				sigma += zeta * data[dataCount - k];
				zeta *= z;
				k++;
			}
			data[0] = sigma / (1.0 - zeta);
			for (int k = 1; k < dataCount; k++) {
				data[k] += z * data[k - 1];
			}
			sigma = data[dataCount - 1];
			zeta = z;
			k = 0;
			while ((k < (dataCount - 1)) && (0.0 != zeta)) {
				sigma += zeta * data[k];
				zeta *= z;
				k++;
			}
			const double z12 = (1.0 - z) * (1.0 - z);
			data[dataCount - 1] = z12 * sigma / (1.0 - zeta);
			for (int k = 1; k < dataCount; k++) {
				data[dataCount - 1 - k] = z * data[dataCount - k] + z12 * data[dataCount - 1 - k];
			}
		}
	}
} /* end get_samples_to_coeff_p */

extern void get_samples_to_coeff_n (
	double data[],
	int32_t dataCount,
	double pole[],
	int32_t poleCount
) {
	if ((NULL == data) || (0 >= dataCount)) {
		return;
	}
	if ((NULL == pole) || (0 >= poleCount)) {
		return;
	}
	if ((0 != poleCount) && (1 != dataCount)) {
		for (int q = 0; q < poleCount; q++) {
			const double z = pole[q];
			double sigma1 = data[0];
			double sigma2 = data[dataCount - 1];
			double zeta = z;
			int k = 1;
			while ((k < (dataCount - 1)) && (0.0 != zeta)) {
				sigma1 += zeta * data[k];
				sigma2 += zeta * data[dataCount - 1 - k];
				zeta *= z;
				k++;
			}
			data[0] = (sigma1 + zeta * sigma2) / (1.0 - zeta * zeta);
			for (int k = 1; k < dataCount; k++) {
				data[k] += z * data[k - 1];
			}
			const double z12 = (1.0 - z) * (1.0 - z);
			data[dataCount - 1] = z12 * (z * data[dataCount - 2] + data[dataCount - 1]) / (1.0 - z * z);
			for (int k = 1; k < dataCount; k++) {
				data[dataCount - 1 - k] = z * data[dataCount - k] + z12 * data[dataCount - 1 - k];
			}
		}
	}
} /* end get_samples_to_coeff_n */

extern void get_samples_to_coeff_w (
	double data[],
	int32_t dataCount,
	double pole[],
	int32_t poleCount
) {
	if ((NULL == data) || (0 >= dataCount)) {
		return;
	}
	if ((NULL == pole) || (0 >= poleCount)) {
		return;
	}
	if ((0 != poleCount) && (1 != dataCount)) {
		for (int q = 0; q < poleCount; q++) {
			const double z = pole[q];
			double sigma1 = 0.0;
			double sigma2 = 0.0;
			double zeta = 1.0;
			int k = 0;
			while ((k < dataCount) && (0.0 != zeta)) {
				sigma1 += zeta * data[k];
				sigma2 += zeta * data[dataCount - 1 - k];
				zeta *= z;
				k++;
			}
			data[0] += z * (sigma1 + zeta * sigma2) / (1.0 - zeta * zeta);
			for (int k = 1; k < dataCount; k++) {
				data[k] += z * data[k - 1];
			}
			const double z12 = (1.0 - z) * (1.0 - z);
			data[dataCount - 1] *= 1.0 - z;
			for (int k = 1; k < dataCount; k++) {
				data[dataCount - 1 - k] = z * data[dataCount - k] + z12 * data[dataCount - 1 - k];
			}
		}
	}
} /* end get_samples_to_coeff_w */

extern void get_samples_to_coeff_a (
	double data[],
	int32_t dataCount,
	double pole[],
	int32_t poleCount
) {
	if ((NULL == data) || (0 >= dataCount)) {
		return;
	}
	if ((NULL == pole) || (0 >= poleCount)) {
		return;
	}
	if ((0 != poleCount) && (1 != dataCount)) {
		for (int q = 0; q < poleCount; q++) {
			const double z = pole[q];
			double sigma1 = 0.0;
			double sigma2 = 0.0;
			double zeta = z;
			int k = 1;
			while ((k < (dataCount - 1)) && (0.0 != zeta)) {
				sigma1 += zeta * data[k];
				sigma2 += zeta * data[dataCount - 1 - k];
				zeta *= z;
				k++;
			}
			data[0] = ((data[0] - zeta * data[dataCount - 1]) * (1.0 + z) / (1.0 - z) - sigma1 + zeta * sigma2) / (1.0 - zeta * zeta);
			for (int k = 1; k < dataCount; k++) {
				data[k] += z * data[k - 1];
			}
			const double z12 = (1.0 - z) * (1.0 - z);
			data[dataCount - 1] -= z * data[dataCount - 2];
			for (int k = 1; k < dataCount; k++) {
				data[dataCount - 1 - k] = z * data[dataCount - k] + z12 * data[dataCount - 1 - k];
			}
		}
	}
} /* end get_samples_to_coeff_a */

extern void get_samples_to_coeff_np (
	double data[],
	int32_t dataCount,
	double pole[],
	int32_t poleCount
) {
	if ((NULL == data) || (0 >= dataCount)) {
		return;
	}
	if ((NULL == pole) || (0 >= poleCount)) {
		return;
	}
	for (int q = 0; q < poleCount; q++) {
		const double z = pole[q];
		double sigma1 = data[0];
		double sigma2 = data[dataCount - 1];
		double zeta = z;
		int k = 1;
		while ((k < dataCount) && (0.0 != zeta)) {
			sigma1 += zeta * data[k];
			sigma2 += zeta * data[dataCount - 1 - k];
			zeta *= z;
			k++;
		}
		const double Z0 = z / (1.0 + zeta);
		sigma1 *= Z0;
		sigma2 *= Z0;
		data[0] -= sigma2;
		for (int k = 1; k < dataCount; k++) {
			data[k] += z * data[k - 1];
		}
		zeta *= zeta;
		const double z12 = (1.0 - z) * (1.0 - z);
		const double Z1 = (1.0 - z) / (1.0 + z);
		data[dataCount - 1] *= (1.0 + zeta) * Z1;
		data[dataCount - 1] -= (sigma2 * zeta / z + sigma1) * Z1;
		for (int k = 1; k < dataCount; k++) {
			data[dataCount - 1 - k] = z * data[dataCount - k] + z12 * data[dataCount - 1 - k];
		}
	}
} /* end get_samples_to_coeff_np */

extern void get_samples_to_coeff_nn (
	double data[],
	int32_t dataCount,
	double pole[],
	int32_t poleCount
) {
	if ((NULL == data) || (0 >= dataCount)) {
		return;
	}
	if ((NULL == pole) || (0 >= poleCount)) {
		return;
	}
	for (int q = 0; q < poleCount; q++) {
		const double z = pole[q];
		double sigma1 = 0.0;
		double sigma2 = 0.0;
		double zeta = 1.0;
		int k = 0;
		while ((k < dataCount) && (0.0 != zeta)) {
			sigma1 += zeta * data[k];
			sigma2 += zeta * data[dataCount - 1 - k];
			zeta *= z;
			k++;
		}
		zeta *= z;
		data[0] -= (sigma1 - zeta * sigma2) * z * z / (1.0 - zeta * zeta);
		for (int k = 1; k < dataCount; k++) {
			data[k] += z * data[k - 1];
		}
		const double z12 = (1.0 - z) * (1.0 - z);
		data[dataCount - 1] *= z12;
		for (int k = 1; k < dataCount; k++) {
			data[dataCount - 1 - k] = z * data[dataCount - k] + z12 * data[dataCount - 1 - k];
		}
	}
} /* end get_samples_to_coeff_nn */

extern void get_samples_to_coeff_nw (
	double data[],
	int32_t dataCount,
	double pole[],
	int32_t poleCount
) {
	if ((NULL == data) || (0 >= dataCount)) {
		return;
	}
	if ((NULL == pole) || (0 >= poleCount)) {
		return;
	}
	for (int q = 0; q < poleCount; q++) {
		const double z = pole[q];
		double sigma1 = 0.0;
		double sigma2 = 0.0;
		double zeta = 1.0;
		int k = 0;
		while ((k < dataCount) && (0.0 != zeta)) {
			sigma1 += zeta * data[k];
			sigma2 += zeta * data[dataCount - 1 - k];
			zeta *= z;
			k++;
		}
		data[0] -= (sigma1 - zeta * sigma2) * z / (1.0 - zeta * zeta);
		for (int k = 1; k < dataCount; k++) {
			data[k] += z * data[k - 1];
		}
		const double z12 = (1.0 - z) * (1.0 - z);
		data[dataCount - 1] *= z12 / (1.0 + z);
		for (int k = 1; k < dataCount; k++) {
			data[dataCount - 1 - k] = z * data[dataCount - k] + z12 * data[dataCount - 1 - k];
		}
	}
} /* end get_samples_to_coeff_nw */

