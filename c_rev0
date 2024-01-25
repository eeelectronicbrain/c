
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 



double Icell(double G, double V, double R_BL) {
    if (G == 0) {
        return 0;
    }
    else {
        return V / (1.0 / G + R_BL);
    }
}

double sneak_cal_a(double* G_arr, double* V_arr, double Isb, double Vsl, double G_offset, double* G_var, double R_SL, double R_BL, int arr) {
    double V_n = Vsl;
    double I_sum = 0;
    double GV_sum = 0;
    double average_V_arr = 0;
    double average_G_arr = 0;

    for (int n = 0; n < arr; n++) {
        double G_target = G_arr[n];
        double G_write = G_target + G_var[n];
        double I_n;
        double V_in = V_arr[n];


        if (G_target > 0) {
            I_n = Icell(G_write + G_offset, V_in - V_n, R_BL) + Icell(G_offset, -V_in - V_n, R_BL);
        }
        else if (G_target < 0) {
            I_n = Icell(G_offset, V_in - V_n, R_BL) + Icell(-G_write + G_offset, -V_in - V_n, R_BL);
        }
        else {
            I_n = Icell(G_offset, V_in - V_n, R_BL) + Icell(G_offset, -V_in - V_n, R_BL);
        }

        Isb = Isb - I_n;
        V_n = V_n + Isb * R_SL;
        double GV_n = G_target * V_in;
        I_sum = I_sum + I_n;
        GV_sum = GV_sum + GV_n;

        average_V_arr += V_in;
        average_G_arr += G_target;
    }

    // Calculate average values


    average_V_arr /= arr;
    average_G_arr /= arr;

    //printf("I_sum: %f, Isb: %f, GV_sum: %f, Average V_arr: %f, Average G_arr: %f\n", I_sum, Isb, GV_sum, average_V_arr, average_G_arr);
    return Isb;
}


double* sneak_cal_b(double* G_arr, double* V_arr, double Isb, double Vsl, double G_offset, double* G_var, double R_SL, double R_BL, int arr) {
    double V_n = Vsl;
    double I_sum = 0;
    double GV_sum = 0;
    double average_V_arr = 0;
    double average_G_arr = 0;

    for (int n = 0; n < arr; n++) {
        double G_target = G_arr[n];
        double G_write = G_target + G_var[n];
        double I_n;
        double V_in = V_arr[n];


        if (G_target > 0) {
            I_n = Icell(G_write + G_offset, V_in - V_n, R_BL) + Icell(G_offset, -V_in - V_n, R_BL);
        }
        else if (G_target < 0) {
            I_n = Icell(G_offset, V_in - V_n, R_BL) + Icell(-G_write + G_offset, -V_in - V_n, R_BL);
        }
        else {
            I_n = Icell(G_offset, V_in - V_n, R_BL) + Icell(G_offset, -V_in - V_n, R_BL);
        }

        Isb = Isb - I_n;
        V_n = V_n + Isb * R_SL;
        double GV_n = G_target * V_in;
        I_sum = I_sum + I_n;
        GV_sum = GV_sum + GV_n;

        average_V_arr += V_in;
        average_G_arr += G_target;
    }

    // Calculate average values


    average_V_arr /= arr;
    average_G_arr /= arr;
    static double result[5];

    result[0] = I_sum;
    result[1] = Isb;
    result[2] = GV_sum;
    result[3] = average_V_arr;
    result[4] = average_G_arr;


    //printf("I_sum: %f, Isb: %f, GV_sum: %f, Average V_arr: %f, Average G_arr: %f\n", I_sum, Isb, GV_sum, average_V_arr, average_G_arr);
    return result;
}


double* sneak_cal_Rthev(double* G_arr, double Isb, double G_offset, double* G_var, double R_SL, double R_BL, int arr) {
    double V_n = 0;
    double I_sum = 0;
    double GV_sum = 0;

    for (int n = 0; n < arr; n++) {
        double G_target = G_arr[n];
        double G_write = G_target + G_var[n];
        double I_n;




        I_n = Icell(fabs(G_write) + G_offset, 1 - V_n, R_BL) + Icell(G_offset, 1 - V_n, R_BL);


        Isb = Isb - I_n;
        V_n = V_n + Isb * R_SL;
        double GV_n = G_target * 1;
        I_sum = I_sum + I_n;
        GV_sum = GV_sum + GV_n;

    }

    // Calculate average values


    static double result[2];

    result[0] = I_sum;
    result[1] = Isb;


    //printf("I_sum: %f, Isb: %f, GV_sum: %f, Average V_arr: %f, Average G_arr: %f\n", I_sum, Isb, GV_sum, average_V_arr, average_G_arr);
    return result;
}


double R_thev(double* G_arr, double G_offset, double* G_var, double R_SL, double R_BL, int arr) {

    double I_a = sneak_cal_Rthev(G_arr, 0, G_offset, G_var, R_SL, R_BL, arr)[1];
    double I_b = sneak_cal_Rthev(G_arr, 10, G_offset, G_var, R_SL, R_BL, arr)[1];
    double c = 10 - I_b * (10) / (I_b - I_a);




    return c;
}

double* thev(double* G_arr, double* V_arr, double G_offset, double* G_var, double R_SL, double R_BL, double R_ext, int arr) {

    double I_sum = 0;
    double GV_sum = 0;


    double I_a = sneak_cal_a(G_arr, V_arr, 0, 0, G_offset, G_var, R_SL, R_BL, arr);
    double* I_b = sneak_cal_b(G_arr, V_arr, 10, 0, G_offset, G_var, R_SL, R_BL, arr);
    double Ithev = 10 - I_b[1] * (10) / (I_b[1] - I_a);

    double Rthev = R_thev(G_arr, G_offset, G_var, R_SL, R_BL, arr);
    double Vthev = Rthev * Ithev;
    double Iout = Vthev / (Rthev + R_ext);


    static double result[7];

    result[0] = Iout;
    result[1] = Vthev;
    result[2] = Rthev;
    result[3] = Ithev;
    result[4] = I_b[2];
    result[5] = I_b[3];
    result[6] = I_b[4];

    return result;
}

/*
void process_array(double(*arr_test)[8][8], int size1) {
    // Your C code to process the array goes here
    // For example, print the values
    for (int i = 0; i < size1; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                printf("%f ", arr_test[i][j][k]);
            }
            printf("\n");
        }
    }
}

double sum_of_array(double(*arr_test)[128][8][8]) {
    double sum = 0.0;

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                sum += (*arr_test)[i][j][k];
            }
        }
    }

    return sum;
}

double sum_of_array_a(double***arr_test) {
    double sum = 0.0;

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                sum += (*arr_test)[i][j][k];
            }
        }
    }

    return sum;
}


double conv(double* G_arr, double* V_arr, double G_offset, double* G_var, double R_SL, double R_BL, double R_ext, int arr) {
    double sum = 0.0;

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                sum += (*arr_test)[i][j][k];
            }
        }
    }

    return sum;
}
*/
double* cal_array(int n_image, int n_out, int n_slide, int arr_size, double* x_np, double* w_np) {
    double* result_a = malloc(n_image * n_out * n_slide * sizeof(double));
    int n = 0;
    if (result_a == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i_b = 0; i_b < n_image; i_b++) {
        for (int out_ch = 0; out_ch < n_out; out_ch++) {
            for (int in_n = 0; in_n < n_slide; in_n++) {
                double* arr_in = malloc(arr_size * sizeof(double));

                for (int arr_n = 0; arr_n < arr_size; arr_n++) {
                    arr_in[arr_n] = x_np[in_n + arr_n * n_slide + i_b * n_slide * arr_size];
                }

                double result = 0;
                for (int arr_n = 0; arr_n < arr_size; arr_n++) {
                    result += arr_in[arr_n] * w_np[arr_n + out_ch * arr_size];
                }
                result_a[n] = result;
                n = n + 1;
                free(arr_in);
            }
        }
    }

    return result_a;
}

double* cal_array_a(int n_image, int n_out, int n_slide, int arr_size, double* x_np, double* w_np) {
    double* result_a = malloc(n_image * n_out * n_slide * sizeof(double));
    int n = 0;
    if (result_a == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i_b = 0; i_b < n_image; i_b++) {
        for (int out_ch = 0; out_ch < n_out; out_ch++) {
            for (int in_n = 0; in_n < n_slide; in_n++) {
                double* arr_in = malloc(arr_size * sizeof(double));
                double* G_arr= malloc(arr_size * sizeof(double));
                double* G_var = malloc(arr_size * sizeof(double));

                for (int arr_n = 0; arr_n < arr_size; arr_n++) {
                    arr_in[arr_n] = x_np[in_n + arr_n * n_slide + i_b * n_slide * arr_size];
                    G_arr[arr_n] = w_np[arr_n + out_ch * arr_size];
                    G_var[arr_n] = 0;

                }

                
                //result_a[n] = thev(G_arr, arr_in, 0, G_var, 0, 0, 0, arr_size)[0];
                result_a[n] = thev(G_arr, arr_in, 0, G_var, 0, 0, 0, arr_size)[0];
                n = n + 1;
                free(arr_in);
                free(G_arr);
                free(G_var);
            }
        }
    }

    return result_a;
}


double* cal_array_d(int n_image, int n_out, int n_slide, int arr_size, double* x_np, double* w_np, double R_SL) {
    double* result_a = malloc(n_image * n_out * n_slide * sizeof(double));
    int n = 0;
    if (result_a == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i_b = 0; i_b < n_image; i_b++) {
        for (int out_ch = 0; out_ch < n_out; out_ch++) {
            for (int in_n = 0; in_n < n_slide; in_n++) {
                double* arr_in = malloc(arr_size * sizeof(double));
                double* G_arr = malloc(arr_size * sizeof(double));
                double* G_var = malloc(arr_size * sizeof(double));

                for (int arr_n = 0; arr_n < arr_size; arr_n++) {
                    arr_in[arr_n] = x_np[in_n + arr_n * n_slide + i_b * n_slide * arr_size];
                    G_arr[arr_n] = w_np[arr_n + out_ch * arr_size];
                    G_var[arr_n] = 0;

                }


                //result_a[n] = thev(G_arr, arr_in, 0, G_var, 0, 0, 0, arr_size)[0];
                result_a[n] = thev(G_arr, arr_in, 0, G_var, R_SL, 0, 0, arr_size)[0];
                n = n + 1;
                free(arr_in);
                free(G_arr);
                free(G_var);
            }
        }
    }

    return result_a;
}

double* cal_array_e(int n_image, int n_out, int n_slide, int arr_size, double* x_np, double* w_np, double* b_np, double R_SL) {
    double* result_a = malloc(n_image * n_out * n_slide * sizeof(double));
    int n = 0;
    if (result_a == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i_b = 0; i_b < n_image; i_b++) {
        for (int out_ch = 0; out_ch < n_out; out_ch++) {
            for (int in_n = 0; in_n < n_slide; in_n++) {
                double* arr_in = malloc(arr_size * sizeof(double));
                double* G_arr = malloc(arr_size * sizeof(double));
                double* G_var = malloc(arr_size * sizeof(double));

                for (int arr_n = 0; arr_n < arr_size; arr_n++) {
                    arr_in[arr_n] = x_np[in_n + arr_n * n_slide + i_b * n_slide * arr_size];
                    G_arr[arr_n] = w_np[arr_n + out_ch * arr_size];
                    G_var[arr_n] = 0;

                }


                //result_a[n] = thev(G_arr, arr_in, 0, G_var, 0, 0, 0, arr_size)[0];
                result_a[n] = thev(G_arr, arr_in, 0, G_var, R_SL, 0, 0, arr_size)[0] + b_np[out_ch];
                n = n + 1;
                free(arr_in);
                free(G_arr);
                free(G_var);
            }
        }
    }

    return result_a;
}

double* cal_array_fc(int n_image, int n_out, int n_slide, int arr_size, double* x_np, double* w_np, double* b_np, double R_SL) {
    double* result_a = malloc(n_image * n_out * sizeof(double));
    int n = 0;
    if (result_a == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i_b = 0; i_b < n_image; i_b++) {
        for (int out_ch = 0; out_ch < n_out; out_ch++) {
             
            double* arr_in = malloc(arr_size * sizeof(double));
            double* G_arr = malloc(arr_size * sizeof(double));
            double* G_var = malloc(arr_size * sizeof(double));

            for (int arr_n = 0; arr_n < arr_size; arr_n++) {
                arr_in[arr_n] = x_np[arr_n + i_b * arr_size];
                G_arr[arr_n] = w_np[arr_n + out_ch * arr_size];
                G_var[arr_n] = 0;

            }


            //result_a[n] = thev(G_arr, arr_in, 0, G_var, 0, 0, 0, arr_size)[0];
            result_a[n] = thev(G_arr, arr_in, 0, G_var, R_SL, 0, 0, arr_size)[0] + b_np[out_ch];
            n = n + 1;
            free(arr_in);
            free(G_arr);
            free(G_var);
            
        }
    }

    return result_a;
}

void free_array(double* arr) {
    free(arr);
}

int main() {

    return 0;
}

