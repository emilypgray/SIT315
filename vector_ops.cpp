#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <chrono>
#include <iostream>

#define PRINT 1

int SZ = 8;
int *v1, *v2, *v3;

//OpenCL memory object, in this case it is the buffer where the vector will be stored
cl_mem bufV1;
cl_mem bufV2;
cl_mem bufV3;

//This is the openCL device ID which is outputted when the OpenCL device wrapper is created
cl_device_id device_id;

//Variable to hold context, the environment where kernels execute wherein synchronisation and memory management are defined
cl_context context;

//Variable which holds the OpenCL prograintm object
cl_program program;

//Variable which holds a created Kernel object (object here is not to be confused with an instantiated class as far as I am aware)
cl_kernel kernel;

//Variable to hold OpenCL command queue, part of the runtime api which manges OpenCL objects, in this case command-queues
cl_command_queue queue;

//Variable used to hold event objects which can be used to refer to various commands
cl_event event = NULL;

int err;

//Create device wrapper object
cl_device_id create_device();

//As the name of the function suggests it calls create_device() which attempts to create a device  and outputs the device id and then attempts to create a context and then a command queue 
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);

//Use the created context and device to run the vector_ops.cl program which is loaded and created via clCreateProgramWithSource and built with clBuildProgram
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);

//Create OpenCL buffer and copy over the vector
void setup_kernel_memory();

//Uses clSetKernelArgs to pass arguments to the specified kernel object
void copy_kernel_args();

//frees memory for device, kernel, queue, etc.
void free_memory();

void init(int *&A, int size);
void print(int *A, int size);

int main(int argc, char **argv)
{
    if (argc > 1)
        SZ = atoi(argv[1]);

    init(v1, SZ);
    init(v2, SZ);
    init(v3, SZ);

    //Global work size, array which holds the number of global work items
    size_t global[1] = {(size_t)SZ};

    //initial vector
    printf("Vector 1: ");
    print(v1, SZ);
    printf("Vector 2: ");
    print(v2, SZ);
    

    setup_openCL_device_context_queue_kernel((char *)"./vector_ops.cl", (char *)"square_magnitude");

    setup_kernel_memory();
    copy_kernel_args();

    //Enqueues a command to execute a kernel on a device. Arugments are as follows:
    /*
     * A valid host command queue
     * Valid Kernel object
     * Number of dimensions used to specify the global work-items and work-items in the work-group.
     * Argument used to specify an array of work_dim unsigned values that describe the offset used to calculate the global ID of a work-item
     * Argument which points to an array of work_dim unsigned values that describe the number of global work-items in work_dim dimensions that will execute the kernel function.
     * Argument which points to an array of work_dim unsigned values that describe the number of work-items that make up a work-group 
     * Next two arguments specify events that need to complete before this particular command can be executed. 
     * Argument which returns an event object that identifies this particular kernel-instance.
    */

    auto start = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);

    //Used to read from a buffer object to host memory
    clEnqueueReadBuffer(queue, bufV3, CL_TRUE, 0, SZ * sizeof(int), &v3[0], 0, NULL, NULL);

    //result vector
    printf("Vector 3: ");
    print(v3, SZ);

    std::cout << "Total time taken by function: " << duration.count() << " microseconds\n";

    //frees memory for device, kernel, queue, etc.
    //you will need to modify this to free your own buffers
    free_memory();
}

void init(int *&A, int size)
{
    A = (int *)malloc(sizeof(int) * size);

    for (long i = 0; i < size; i++)
    {
        A[i] = rand() % 100; // any number less than 100
    }
}

void print(int *A, int size)
{
    if (PRINT == 0)
    {
        return;
    }

    if (PRINT == 1 && size > 15)
    {
        for (long i = 0; i < 5; i++)
        {                        //rows
            printf("%d ", A[i]); // print the cell value
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++)
        {                        //rows
            printf("%d ", A[i]); // print the cell value
        }
    }
    else
    {
        for (long i = 0; i < size; i++)
        {                        //rows
            printf("%d ", A[i]); // print the cell value
        }
    }
    printf("\n----------------------------\n");
}

void free_memory()
{
    //free the buffers
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV3);

    //free opencl objects
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(v1);
    free(v2);
    free(v3);
}


void copy_kernel_args()
{
    //Set the argument value for a specific argument of a kernel. A valid kernel object, argument index, argument size, argument value
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV3);

    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

void setup_kernel_memory()
{
    //To create a buffer object (as stated in the documentation)
    //The second parameter of the clCreateBuffer is cl_mem_flags flags. Check the OpenCL documention to find out what is it's purpose and read the List of supported memory flag values 
    //Purpose of memory flag is to specify the access modifier of the memory, i.e. is it read-only etc.
    bufV1 = clCreateBuffer(context, CL_MEM_READ_ONLY, SZ * sizeof(int), NULL, NULL);
    bufV2 = clCreateBuffer(context, CL_MEM_READ_ONLY, SZ * sizeof(int), NULL, NULL);
    bufV3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SZ * sizeof(int), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
    device_id = create_device();
    cl_int err;

    // Creates a context which is used by the OpenCL runtime for management of objects such as command-queues, memory, program and kernel objects also used for kernel execution of more or more devices speicified in the context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, filename);

    //To create a host or device command-queue on a specific device.
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

    //Creates a program object for a context, and loads source code specified by text strings into the program object. Its arguments are a context, number of strings, array of count pointers, number of chars in each string, and an error code which returns null if there are no errors
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

cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   // Access a device
   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      // CPU
      printf("GPU not found\n");
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}