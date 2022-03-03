#include <iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<string>
#include<cmath>
#include<chrono>
#include<cstdlib>
#include<omp.h>


using namespace std;
using namespace std::chrono;

// Define the number of clusters that the program will
// classify the data into
#define NUM_CLUSTERS 8

// Define number of iterations that algorithm will perform
#define NUM_ITERATIONS 10

// Define number of threads omp will use
#define omp_set_num_threads = 8;

// Define the structure for each point
struct dataPoint
{
	double x, y; // x and y coordinates
    int clusterLabel; // cluster label for the point. May change in early iterations of program
    double minDistance; // the distance between the point and the cluster centre nearest to it

    // Define initialisation list to set default values for a point
    dataPoint() :
        // x and y don't need default values as they are always assigned,
        // but the initialization list requires that all attributes 
        // have a default value
        x(0.0),
        y(0.0),
        // set default cluster label to NUM_CLUSTERS, as this values is outside the available
        // cluster labels (0 - (NUM_CLUSTERS - 1))
        clusterLabel(NUM_CLUSTERS), 
        // set default minDistance to arbitrary large value
        minDistance(100000) {}

    double CalculateDistance(dataPoint point1)
    {
        // calculate and return the Euclidean distance between current point
        // and another point
        double distance = sqrt(pow((point1.x - x), 2) + pow((point1.y - y), 2));
        return distance;
    }  
};

struct Information
{
    vector<dataPoint>* data;
    dataPoint* clusterCentres;
    int dataSize;
};

void recentreClusters(void* args, int iteration)
{
    Information* task = ((struct Information*)args);

    double numPoints;
    double xcoord, ycoord;

    #pragma omp parallel for schedule(guided)
    // loop through cluster centres
    for (int i = 0; i < NUM_CLUSTERS; i++)
    {
        // set numPoints equal to zero. This will count how many points are in the
        // cluster so that an average of the points can be calculated
        numPoints = 0;
        // xcoord and ycoord will add the value of the x and y coordinates of each data
        // point in a cluster label and then be divided by the total number of points
        // to get the average coordinate
        xcoord = 0;
        ycoord = 0;

        #pragma omp parallel for schedule(guided)
        // loop through the data points in the vector
        for (int j = 0; j < task->dataSize; j++)
        {
            // if the cluster label of the data point is equal to the index of the current cluster
            // label, then add the value of the x and y coordinates to xcoord and ycoord
            if (task->data->at(j).clusterLabel == i)
            {
                // increment the number of points
                numPoints += 1;
                xcoord += task->data->at(j).x;
                ycoord += task->data->at(j).y;
                // reset the min distance and clusterLabel back to the initial arbitrary high value
                // if it is not the last iteration
                if (iteration < NUM_CLUSTERS - 1) {               
                    task->data->at(j).minDistance = 100000;
                    task->data->at(j).clusterLabel = NUM_CLUSTERS;

                }

            }
        }
        // set the new cluster centre coordinates as the average value of all coordinates in
        // the cluster
        task->clusterCentres[i].x = xcoord / numPoints;
        task->clusterCentres[i].y = ycoord / numPoints;
    }
};

void kCluster(void* args)
{
    Information* task = ((struct Information*)args);

    // assign number of iterations to a variable
    int iteration = 0;

    while (iteration < NUM_ITERATIONS)
    {
        // loop through each point in the vector data to calculate minimum distance
        // from closest cluster centre

        // define local variables
        // distance will be calculated in each iteration of for loop
        double distance;

        #pragma omp parallel for schedule(guided)
        // loop through data points in vector
        for (int i = 0; i < task->dataSize; i++)
        {
            // loop through data points in clusterCentres
            for (int j = 0; j < NUM_CLUSTERS; j++)
            {
                // set distance equal to distance between point in vector and cluster centre
                distance = task->data->at(i).CalculateDistance(task->clusterCentres[j]);
                // if the distance is less than the current minDistance, then update minDistance and clusterlabel
                if (distance < task->data->at(i).minDistance)
                {
                    task->data->at(i).minDistance = distance;
                    task->data->at(i).clusterLabel = j;
                } 
            }
        }
        //call the recentreClusters function and pass the task to it
        recentreClusters(task, iteration);
        cout << "iteration: " << iteration << "\n";
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            cout << "cluster center: " << i << "\tx: " << task->clusterCentres[i].x << "\ty : " << task->clusterCentres[i].y << "\n";
        }
        cout << "\n";
        iteration += 1;
    }
};
    
int main()
{

    srand(time(NULL));

    // create a read-only file object from data.csv
    ifstream file("unbalance.csv");

    string line, x, y;

    // create a vector to store data being read in from file
    // data stored in a vector because extra data can be added to
    // the end of the vector and we do not need to define it's 
    // size on initializing it
    vector<dataPoint> data;
    vector<dataPoint>* dataPtr = &data;

    // while there is still data to be read from the file,
    // continue on with loop
    while (getline(file, line)) {

        // create new instance of dataPoint for each line in the file
        dataPoint* point = new dataPoint;

        // use stringstream on the line so that the x and y values
        // stored on one line can be read in separately
        stringstream lineStream(line);

        // store value before the comma as x
        getline(lineStream, x, ',');
        point->x = stod(x);

        // store value after the comma as y
        getline(lineStream, y, '\n');
        point->y = stod(y);

        // add the point to the data vector
        data.push_back(*point);
    }

    // check the size of the vector and therefore the number of
    // points to be classified
    int dataSize = data.size();

    // create array to contain the number of cluster centres. Length
    // equal to numer of cluster centres
    dataPoint* clusterCentres = new dataPoint[NUM_CLUSTERS * sizeof(dataPoint)];
    
    for (int i = 0; i < NUM_CLUSTERS; i++)
    {
        // randomly select points from initial data to 
        // populate the array
        clusterCentres[i] = data[rand() % dataSize];
    }

    Information* task = new Information;
    task->data = dataPtr;
    task->dataSize = dataSize;
    task->clusterCentres = clusterCentres;

    auto start = high_resolution_clock::now();
    kCluster(task);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    //select 50 random numbers from the data set and print them out with their cluster labels
    //int j;
    //cout << "\tx\t\ty\t  Cluster Label\n";
    //cout << "  ----------------------------------------------\n";
    //for (int i = 0; i < 50 ; i++) {
    //    /*j = rand() % dataSize;*/
    //    cout << "\t" << data[i].x << "\t|\t" << data[i].y << "\t|\t" << data[i].clusterLabel << "\n";
    //    i++;
    //}

    cout << "The total time of the program is: " << duration.count() << " microseconds";
}

