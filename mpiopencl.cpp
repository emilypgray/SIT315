#include <iostream>
#include<stdio.h>
#include<time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>
#include <chrono>
#include <stdio.h>
#include <CL/cl.h>
#include <unistd.h>

using namespace std;

// define the matrices and the subsets of A and C that
// each node will work on
int *A, *B, *C, *ASub, *CSub;

// define default size of the matrix
int SZ = 4;

int num_elements_to_scatter_or_gather;
int num_rows_per_process_from_A;
int num_elements_to_bcast;

// define cl objects
cl_mem bufA, bufB, bufC;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_event event = NULL;

int err;

// define functions
cl_device_id create_device();
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
void setup_kernel_memory();
void copy_kernel_args();
void free_memory();

void init(int *&A, int size, bool initialize);
void print(int *A, int size);


int main(int argc, char** argv) {

    // take user input for matrix size
   if(argc > 1) SZ = atoi(argv[1]);
   
   // Initialize the MPI environment
   MPI_Init(NULL, NULL);

   // get the number of processes
   int num_processes;
   MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

   // get the process rank
   int process_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // if the size of the matrix is not evenly divisible by the number of
    // processes, set the number of elements to send to the node equal 
    // to zero, and carry out the whole multiplication on the head
   if (SZ % num_processes == 0){
      num_rows_per_process_from_A = SZ / num_processes;
      num_elements_to_scatter_or_gather = num_rows_per_process_from_A * SZ;
      
   }
   else {
      num_elements_to_scatter_or_gather = 0;
      num_rows_per_process_from_A = SZ;
   }

   // num elements to braodcast equal to the size squared
   num_elements_to_bcast = SZ * SZ;

   init(ASub, num_elements_to_scatter_or_gather, false);
   init(CSub, num_elements_to_scatter_or_gather, false);
   
   if(process_rank == 0)
   {
      // initiliaze A and B with values. Initialize memory for C
      init(A, SZ * SZ, true), init(B, SZ * SZ, true), init(C, SZ*SZ, false);
      print(A, SZ * SZ);
      print(B, SZ * SZ);
   } else {
      // initialize memory for B
      init(B, SZ * SZ, false);
   }
   
   auto start = std::chrono::high_resolution_clock::now();

   // scatter A out into its subsets
   MPI_Scatter(A, num_elements_to_scatter_or_gather,  MPI_INT, ASub, num_elements_to_scatter_or_gather, MPI_INT, 0 , MPI_COMM_WORLD);

   // braodcast B
   MPI_Bcast(B, num_elements_to_bcast , MPI_INT, 0 , MPI_COMM_WORLD);

   size_t global[] = {(size_t)num_rows_per_process_from_A, (size_t)SZ};

   // set up the cl environment on each node
   setup_openCL_device_context_queue_kernel((char *)"./matrix_ops.cl", (char *)"add_matrix");

   setup_kernel_memory();

   copy_kernel_args();

   clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);

   clWaitForEvents(1, &event);

   // read the C values into S subsets
   clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, num_elements_to_scatter_or_gather * sizeof(int), &CSub[0], 0, NULL, NULL);

   // gather the C subsets
   MPI_Gather(CSub, num_elements_to_scatter_or_gather, MPI_INT, C, num_elements_to_scatter_or_gather, MPI_INT, 0, MPI_COMM_WORLD);

   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);

   if (process_rank == 0)
   {   
      print(C, SZ * SZ);
      cout << "The program took: " << duration.count() << " microseconds\n";
   }

   free_memory();

   MPI_Finalize();   
 
}


void init(int *&A, int size, bool initialize)
{
   // assign the memory
   A = (int *)malloc(sizeof(int) * size);

   if (!initialize)
   {
       return;
   }

   for (long i = 0; i < size; i++)
   {
      A[i] = rand() % 10; // any number less than 100
   }
}

void print(int *A, int size)
{
   for (long i = 0; i < size; i++)
   {
      printf("%d ", A[i]); 
      if ((i + 1) % SZ == 0)
      {
         printf("\n");
      }     
   }       
   printf("----------------------------\n");
}

void free_memory()
{
   //free the buffers
   clReleaseMemObject(bufA);
   clReleaseMemObject(bufB);
   clReleaseMemObject(bufC);

   //free opencl objects
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   free(A);
   free(B);
   free(C);
}

void copy_kernel_args()
{
   clSetKernelArg(kernel, 0, sizeof(int), (void *)&num_rows_per_process_from_A);
   clSetKernelArg(kernel, 1, sizeof(int), (void *)&SZ);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufA);
   clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufB);
   clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bufC);

   if (err < 0)
   {
      perror("Couldn't create a kernel argument");
      printf("error = %d", err);
      exit(1);
   }
}

void setup_kernel_memory()
{
   bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, num_elements_to_scatter_or_gather * sizeof(int), NULL, NULL);
   bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, num_elements_to_bcast * sizeof(int), NULL, NULL);
   bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_elements_to_scatter_or_gather * sizeof(int), NULL, NULL);

   // Copy matrices to the GPU
   clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, num_elements_to_scatter_or_gather * sizeof(int), &ASub[0], 0, NULL, NULL);
   clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, num_elements_to_bcast * sizeof(int), &B[0], 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
   device_id = create_device();
   cl_int err;
   context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
   if (err < 0)
   {
      perror("Couldn't create a context");
      exit(1);
   }

   program = build_program(context, device_id, filename);
   queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
   if (err < 0)
   {
      perror("Couldn't create a command queue");
      exit(1);
   };

   kernel = clCreateKernel(program, kernelname, &err);
   if (err < 0)
   {
      perror("Couldn't create a kernel");
      printf("error =%d", err);
      exit(1);
   };
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if (program_handle == NULL)
   {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char *)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file 

   Creates a program from the source code in the add_numbers.cl file. 
   Specifically, the code reads the file's content into a char array 
   called program_buffer, and then calls clCreateProgramWithSource.
   */
   program = clCreateProgramWithSource(ctx, 1,
                                       (const char **)&program_buffer, &program_size, &err);
   if (err < 0)
   {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program 

   The fourth parameter accepts options that configure the compilation. 
   These are similar to the flags used by gcc. For example, you can 
   define a macro with the option -DMACRO=VALUE and turn off optimization 
   with -cl-opt-disable.
   */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err < 0)
   {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                            0, NULL, &log_size);
      program_log = (char *)malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

cl_device_id create_device()
{

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if (err < 0)
   {
      perror("Couldn't identify a platform");
      exit(1);
   }

   // Access a device
   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if (err == CL_DEVICE_NOT_FOUND)
   {
      // CPU
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if (err < 0)
   {
      perror("Couldn't access any devices");
      exit(1);
   }

   return dev;
}