#ifdef PAIR_CLASS

PairStyle(fireann,PairFIREANN)   // fireann is the name in the input script

#else

#ifndef LMP_PAIR_FIREANN_H
#define LMP_PAIR_FIREANN_H

#include "pair.h"
#include <torch/torch.h>
#include <torch/script.h> 
#include <string>

namespace LAMMPS_NS 
{
    class PairFIREANN : public Pair 
    { 
         public:
             torch::jit::script::Module module;
             PairFIREANN(class LAMMPS *);
             virtual ~PairFIREANN();
             virtual void compute(int, int);
             virtual void init_style();
             virtual double init_one(int, int);
             virtual void settings(int, char **);
             virtual void coeff(int, char **);
         protected:
             virtual void allocate();
             virtual int select_gpu();
             double cutoff; 
             double cutoffsq;
             std::string datatype;
             torch::Dtype tensor_type=torch::kDouble;
             torch::Tensor ef = torch::zeros({3},torch::kDouble);
             torch::TensorOptions option1=torch::TensorOptions().dtype(torch::kDouble);
             torch::TensorOptions option2=torch::TensorOptions().dtype(torch::kLong);
             torch::Tensor device_tensor=torch::empty(1);
    };
}

#endif
#endif
