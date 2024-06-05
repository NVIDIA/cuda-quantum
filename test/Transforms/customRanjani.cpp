
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/optimizers.h>
#include <cstdlib>

int main() {
    auto kernel =[]() __qpu__{
        cudaq::qubit q;
        int a=6;
        double r=5.7695;
        //(double)rand()/RAND_MAX;
        //float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        cudaq::qubit q2;
        cudaq::qubit q3;
        h(q);
        /*
        cudaq::qvector v(5);
        for(int i=0;i<5;i++)
        {
            x<cudaq::ctrl> (q,v[i]);
        }*/
        
    };
};
