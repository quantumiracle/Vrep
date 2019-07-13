import numpy as np
import time
from tqdm import tqdm

import torch

# !! this one works with MAXMIMIZATION
class BboptMaximizer:
    def __init__(self, model, state, bounds, iterations, state_img_batch=None, state_numerical_batch=None, state_is_image=False):
#        if torch.cuda.is_available():
#            print("cuda is available :D")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.state_tensor = state

        # bounds = [[low1,up1],[low2,up2],..]
        self.iterations = iterations
        self.bounds = bounds

        # this will be a tensor of best action, shape = DIM_ACTION
        self.argmax = None
        self.max_function_value = None

        self.state_img_batch = None
        self.state_numerical_batch = None

        self.STATE_IS_IMAGE = state_is_image
        if(not state_img_batch is None):
            # state batch is a mode to use the optimizer on the batch of state
            # size will be a tensor of n_batch,n_dim_states
            self.state_img_batch = state_img_batch.cpu()
        if(not state_numerical_batch is None):
            # state batch is a mode to use the optimizer on the batch of state
            # size will be a tensor of n_batch,n_dim_states
            self.state_numerical_batch = state_numerical_batch.cpu()

        # the model will feed self.BATCH_SIZE_TO_HANDLE states at a time to the model
        # the input state batch will be repeated self.CEM_N_BATCH_BATCH_MODE times, so it's best to keep it as close to N as possible
        # this is created so that small GPUs can train the model
        self.BATCH_SIZE_TO_HANDLE = 256
        self.N = 64
        self.CEM_N_BATCH_BATCH_MODE = None # will be determined once the batch size is known
        self.M = 6

        self.DISABLE_TQDM = True

    def optimize(self, iter, max_time=1):
        self.optimization_problem.run_optimization(iter, max_time)

    def get_argmax(self):
        if self.argmax is None:
            self.optimize_cem_batch()
        return self.argmax

    def get_function_value(self):
        if self.max_function_value is None:
            self.optimize_cem_batch()
        return self.max_function_value

    def optimize_cem_batch(self):
        with torch.no_grad():
    #        print("bbopt: getting argmin...")
            n_batch = None
            if self.state_img_batch is not None and self.state_numerical_batch is not None:
#                print("bbopt: both image and numerical states provided")
                n_batch_img = self.state_img_batch.size()[0]
                n_batch_numerical = self.state_numerical_batch.size()[0]
                if n_batch_img != n_batch_numerical:
                    print("bbopt: number of samples in batches are not equal, img:",n_batch_img,"num:",n_batch_numerical)
                    return
                else:
                    n_batch = n_batch_img
            elif self.state_img_batch is not None:
                n_batch = self.state_img_batch.size()[0]
            elif self.state_numerical_batch is not None:
                n_batch = self.state_numerical_batch.size()[0]
            else:
                print("bbopt: neither numerical nor image state given")
                return

            self.CEM_N_BATCH_BATCH_MODE = int(self.N*n_batch/self.BATCH_SIZE_TO_HANDLE)
            # the batch will be repeated self.CEM_N_BATCH_BATCH_MODE times more
            if self.CEM_N_BATCH_BATCH_MODE < 1:
                self.CEM_N_BATCH_BATCH_MODE = 1
                self.BATCH_SIZE_TO_HANDLE = self.N*n_batch
            if (self.N*n_batch) % self.BATCH_SIZE_TO_HANDLE != 0:
                # account for the last batch that is not divisible
                self.CEM_N_BATCH_BATCH_MODE += 1
#                raise Exception("the self.CEM_N_BATCH_BATCH_MODE is not an int:",self.N,"*",n_batch,"/",self.BATCH_SIZE_TO_HANDLE)

            mean = np.zeros((n_batch,len(self.bounds)))
            cov = np.zeros((n_batch,len(self.bounds),len(self.bounds)))
            x_best = None
            best_results = None
            self.argmax = [None]*n_batch
            self.max_function_value = [None]*n_batch
            action_tensor = None
            mean = 0.5 * (self.bounds[:,0] + self.bounds[:,1]) * np.ones((n_batch, len(self.bounds)))
            shape = (n_batch, len(self.bounds), len(self.bounds))
            cov = np.zeros(shape)
            idx = np.arange(shape[1])
            cov[:,idx,idx]= 0.5 * (self.bounds[:,1]-self.bounds[:,0]) * 0.5

            repeated_img_state_tensor = None
            repeated_numerical_state_tensor = None
            repeated_img_state_tensor = self.state_img_batch.repeat(self.N,1,1,1)
            repeated_numerical_state_tensor = self.state_numerical_batch.repeat(self.N,1)

            for iter in tqdm(range(self.iterations),disable=self.DISABLE_TQDM):
                results = np.array([])
                # Define an output queue
                time_start = time.time()

                action_array = np.zeros((n_batch*self.N,len(self.bounds)))
                action_tensor_list = [None]*n_batch

#                act_array = torch.cat(torch.Tensor(np.random.multivariate_normal(mean=mean, cov=cov, size=N)).to(self.device).view(N,-1))
#                print('debug', act_array)
                for n in range(n_batch):
                    # load values in a window of M at a time
                    action_array[n*self.N:(n+1)*self.N] = np.random.multivariate_normal(mean=mean[n], cov=cov[n], size=self.N)
                    action_tensor_list[n] = torch.Tensor(action_array[n*self.N:(n+1)*self.N]).to(self.device).view(self.N,-1)
                action_tensor = torch.cat(tuple(action_tensor_list),0)
#                print('before', action_tensor)

                for i in tqdm(range(self.CEM_N_BATCH_BATCH_MODE),disable=self.DISABLE_TQDM):
                    from_idx = self.BATCH_SIZE_TO_HANDLE*i
                    to_idx = self.BATCH_SIZE_TO_HANDLE*(i+1)
                    if(to_idx > n_batch*self.N-1):
#                        print("setting to_idx to",(n_batch*self.N-1),"instead of",to_idx)
                        to_idx = n_batch*self.N-1
                    # this will predict n_batch*N batch size, not sure if it can do it or not
                    result = self.model(
                                    repeated_img_state_tensor[from_idx:to_idx].to(self.device),
                                    repeated_numerical_state_tensor[from_idx:to_idx].to(self.device),
                                    action_tensor[from_idx:to_idx].to(self.device)
                                ).view(-1).cpu().numpy()
                    results = np.concatenate((results,result))

                for n in range(n_batch):
                    # move in a window of M at a time
                    # get the index of the M largest values
                    ind = np.argpartition(results[n*self.N:(n+1)*self.N], -self.M)[-self.M:]

                    # get the x that corresponds to these M first values
                    # x_best[n] = array of N best values
                    x_best = action_array[n*self.N:(n+1)*self.N][ind]
                    best_results = results[n*self.N:(n+1)*self.N][ind]

                    # fit a gaussian to these M values
                    mean[n] = np.mean(x_best,axis=0)

                    # cov needs the features to be along the rows, and the samples along the cols
                    cov[n] = np.cov(x_best.T)

                    self.argmax[n] = torch.Tensor(x_best[np.argmax(best_results)]).to(self.device)
                    self.max_function_value[n] = best_results[np.argmax(best_results)]

#            print("bb: argmax",self.argmax)
            self.argmax = torch.cat(tuple(self.argmax),0).to(self.device)
            self.max_function_value = torch.Tensor(self.max_function_value).to(self.device)

            time_end = time.time()
    #        print("bbopt: optimization done in =",(time_end - time_start),"sec")
