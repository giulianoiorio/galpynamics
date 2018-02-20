from __future__ import division, print_function
import numpy as np
import multiprocessing as mp

#todo It is better to parallel along Z
class ParDo:
    '''
    Manage Multiprocess things
    '''
    def __init__(self, nproc, func=None):
        """Init

        :param nproc: Number of processes
        :param func:  function to parallelize
        """

        self.n=nproc
        self.process=list(np.zeros(nproc))
        self.initialize()

        if func is not None: self.set_func(func=func)

    def initialize(self):
        """Initialize Queue

        :return:
        """
        self.output=mp.Queue()

    def set_func(self,func):
        """Set function to parallelize

        :param func: function to parallelize. It should be in the form func(array,*args) where array will be divided in
                chunks in the parallelization. It should return an array with the first column filled with the quantity array
        """

        self.func=func

    def _target(self,*args):
        """Internal function

        :param args: args of self.func (see set_func)
        :return:
        """

        self.output.put(self.func(*args))


    def run_grid(self, array, args, _sorted=True):
        """Run func in parallel
        It parallelizes the func dividing the first argument in chunks.
        If _sorted=True, the ouput is re-sorted following the input order of the first array
        :param array: Firt argument of function self.func
        :param args: other args
        :param _sorted: If True, sort the output to match the order of array.
                        Otherwise the order is casual and depends on the output order of the chunks.
                        In any case the match between array and results are safe.
        :return: results of func ordered with array as key
        """


        target=self._target
        # Initialize process
        if self.n == 1:
            self.process[0] = mp.Process(target=target, args=(array[:],) + args)
        else:
            dim = int(len(array) / self.n)
            for i in range(self.n - 1):
                start = int(dim * i)
                end = int(dim * (i + 1))
                self.process[i] = mp.Process(target=target, args=(array[start:end],) + args)
            self.process[-1] = mp.Process(target=target, args=(array[end:],) + args)

        # Run
        ##start
        for p in self.process:
            p.start()
        ##dequeue
        results = np.concatenate([self.output.get() for p in self.process])
        ##Join
        for p in self.process:
            p.join()
        ##Order
        if _sorted:
            #original_order=np.argsort(array, kind='mergesort')
            #final_order = np.argsort(results[:, 0], kind='mergesort')
            #results[original_order]=results[final_order]
            idx_sort=np.argsort(results[:,0], kind='mergesort')
            results=results[idx_sort]
        else:
            pass

        return results



    def run(self, array1, array2, args, _sorted=True):
        """Run func in parallel
        It parallelizes the func dividing the first and second argument in chunks.
        If _sorted=True, the ouput is re-sorted following the input order of the first array
        :param array1: First argument of function self.func
        :param array2: Secon argument of function self.func
        :param args: other args
        :param _sorted: If True, sort the output to match the order of array1.
                        Otherwise the order is casual and depends on the output order of the chunks.
                        In any case the match between array and results are safe.
        :return: results of func ordered with array as key
        """

        if len(array1)!=len(array2): raise ValueError('Unequal length')


        target=self._target
        # Initialize process
        if self.n == 1:
            self.process[0] = mp.Process(target=target, args=(array1[:],array2[:]) + args)
        else:
            dim = int(len(array1) / self.n)
            for i in range(self.n - 1):
                start = int(dim * i)
                end = int(dim * (i + 1))
                self.process[i] = mp.Process(target=target, args=(array1[start:end],array2[start:end]) + args)
            self.process[-1] = mp.Process(target=target, args=(array1[end:],array2[end:]) + args)


        # Run
        ##start
        for p in self.process:
            p.start()
        ##dequeue
        results = np.concatenate([self.output.get() for p in self.process])
        ##Join
        for p in self.process:
            p.join()
        ##Order
        if _sorted:
            original_order=np.argsort(array1, kind='mergesort')
            final_order = np.argsort(results[:, 0], kind='mergesort')
            results[original_order]=results[final_order]
        else:
            pass

        return results
