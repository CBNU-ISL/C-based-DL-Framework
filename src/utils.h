#ifndef UTILS_H
#define UTILS_H

#define MAX_MATRIX_DIM 7
#define PI 3.14159265358979323846
#define MIN_NUM 1e-7
#define DEFAULT -999
#define NONE 0
#include <stdbool.h>
#include<stdarg.h>
#include "cJSON.h"


typedef struct _Matrix Matrix;

/*
    maximum dimension: 7
*/
struct _Matrix{
    int dim;
    int size;
    int shape[7];
    double *matrix;
    union{
        void *connection_ptr;
        double *matrix1d;
        double **matrix2d;
        double ***matrix3d;
        double ****matrix4d;
        double *****matrix5d;
        double ******matrix6d;
        double *******matrix7d;
    };
};

typedef enum _InitializeNum{
    ZEROS = 30,
    CUSTOM,
    HE,
    XAVIER,
    SHAPE_ONLY
};


void print_matrix(Matrix *arr, bool shape_only);
void print_recursive_matrix(Matrix* arr, int cur_dim, int shape[7]);
double sign(double num);
int get_index(Matrix *arr, ...);
int get_index_by_array(Matrix *arr, int* multiIndex);
double get_value(Matrix *arr, ...);
double get_value_by_array(Matrix *arr, int* multiIndex);
void make_connection_matrix(Matrix *arr);
bool delete_connection_matrix(Matrix *arr);
double randn();
void copy_matrix(Matrix* arr, Matrix** _dest, bool form_only); //Form only: default false // if true : all element -> 0
void delete_matrix(Matrix* arr);    //inplace
void init_by_shape_arr_matrix(Matrix** _arr, int* shape, int dim, int initType, double inputNodeNum_or_customValue); //inplace, if initType None or Zeros: inputNodeNum_or_customValue = 0,
void init_matrix(Matrix** _arr, int dim, int initType, double inputNodeNum_or_customValue, ...);
void reshape_matrix(Matrix* arr, int dim, ...);    //inplace
void reshape_by_arr_matrix(Matrix* arr, int dim, int*shape);    //inplace
void transpose_matrix(Matrix *arr, Matrix ** _dest, ...);
void dot2d_matrix(Matrix* A, Matrix* B, Matrix** _dest, double alpha);
void dot2d_TN_matrix(Matrix* A, Matrix* B, Matrix** _dest, double alpha);
void dot2d_NT_matrix(Matrix* A, Matrix* B, Matrix** _dest, double alpha);
void dot2d_TT_matrix(Matrix* A, Matrix* B, Matrix** _dest, double alpha);
void add_two_matrix(Matrix*A,Matrix*B,Matrix**_dest);
void sub_two_matrix(Matrix*A,Matrix*B,Matrix**_dest);
void mul_two_matrix(Matrix*A,Matrix*B,Matrix**_dest);
void div_two_matrix(Matrix*A,Matrix*B,Matrix**_dest);
void add_variable_matrix(Matrix*A,double b,Matrix**_dest);
void mul_variable_matrix(Matrix*A,double b,Matrix**_dest);
void div_variable_matrix(Matrix*A,double b,Matrix**_dest);
void im2col_matrix(Matrix *input, Matrix **_col, int filter_h, int filter_w, int stride_h,int stride_w,int pad_up,int pad_down,int pad_left, int pad_right);
void col2im_matrix(Matrix *input, Matrix *_col, Matrix **_img, int filter_h, int filter_w, int stride_h,int stride_w,int pad_up,int pad_down,int pad_left, int pad_right);
bool is_shape_matching_matrix(Matrix *A, Matrix *B);
double sigmoid_(double x);
void softmax_(Matrix *x, Matrix **_y);
void cross_entropy_error(Matrix *y, Matrix *t, double *loss);
void one_hot_encoding(Matrix *arr, Matrix **_dest,int min_index, int max_index);
void categorical_encoding(Matrix *arr, Matrix **_dest);
void argmax_2d_matrix(Matrix *arr,Matrix **_dest);
void swap(int *a, int *b);
void train_test_split_matrix(Matrix *data_x, Matrix *data_y,Matrix **train_x, Matrix **train_y, Matrix **test_x, Matrix **test_y, double train_ratio, bool stratify);
void init_batch(Matrix **_batch_x, Matrix **_batch_y,  const Matrix *x_sample, const Matrix *y_sample, int batchsize);
void get_batch_by_index(int index, Matrix *batch_x, Matrix *batch_y,  const Matrix *x_sample, const Matrix *y_sample, int batchsize);
void cJSON_AddMatrixToObject(cJSON *object, const Matrix matrix, const char* matrix_name);
void removeBrackets(char *str);
void cJSON_GetMatrix(cJSON *object, Matrix **matrix);
void saveToCSV(const char* filename, int rows, int cols, double** arr, const char* colNames[]);
void saveToCSV_int(const char* filename, int rows, int cols, int** arr, const char* colNames[]);
#endif
