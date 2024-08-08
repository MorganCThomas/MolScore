# Copyright 2018 Jacob D. Durrant
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallelizer.py

Abstract parallel computation utility.

The "parallelizer" object exposes a simple map interface that takes a function
and a list of arguments and returns the result of applying the function to
each argument. Internally, the parallelizer class can determine what parallel
capabilities are present on a system and automatically pick between "mpi",
"multiprocessing" or "serial" in order to speed up the map operation. This
approach simplifies development and allows the same program to run on a laptop
or a high-performance computer cluster, utilizing the full resources of each
system. (Description provided by Harrison Green.)
"""

import __future__
import multiprocessing
import platform
# ------- Explicitly set fork context -------
if platform.system() == "Linux":
    context = multiprocessing.get_context("fork")
else:
    context = multiprocessing.get_context("spawn")
# -------------------------------------------
import sys

MPI_installed = False
try:
    import mpi4py

    MPI_installed = True
except:
    MPI_installed = False


class Parallelizer(object):
    """
    Abstract parallelization class
    """

    def __init__(self, mode=None, num_procs=None, flag_for_low_level=False):
        """
        This will initialize the Parallelizer class and kick off the specific classes for multiprocessing and MPI.

        Default num_procs is all the processesors possible

        This will also establish:
            :self   bol     self.HAS_MPI: If true it can import mpi4py;
                        if False it either cant import mpi4py or the mode/flag_for_low_level dictates not to check mpi4py import
                            due to issues which arrise when mpi is started within a program which is already mpi enabled
            :self   str     self.mode:  The mode which will be used for this paralellization. This determined by mode and the enviorment
                                        default is mpi if the enviorment allows, if not mpi then multiprocessing on all processors unless stated otherwise.
            :self   class   self.parallel_obj: This is the obstantiated object of the class of parallizations.
                            ie) if self.mode=='mpi' self.parallel_obj will be an instance of the mpi class
                                This style of retained parallel_obj to be used later is important because this is the object which controls the work nodes and maintains the mpi universe
                            self.parallel_obj will be set to None for simpler parallization methods like serial
            :self   int     self.num_processor:   the number of processors or nodes that will be used. If None than we will use all available nodes/processors
                                                This will be overriden and fixed to a single processor if mode==serial
        Inputs:
        :param str mode: the multiprocess mode to be used, ie) serial, multiprocessing, mpi, or None:
                            if None then we will try to pick a possible multiprocessing choice. This should only be used for
                            top level coding. It is best practice to specify which multiprocessing choice to use.
                            if you have smaller programs used by a larger program, with both mpi enabled there will be problems, so specify multiprocessing is important.
        :param int num_procs:   the number of processors or nodes that will be used. If None than we will use all available nodes/processors
                                        This will be overriden and fixed to a single processor if mode==serial
        :param bol flag_for_low_level: this will override mode and number of processors and set it to a multiprocess as serial. This is useful because
                                a low-level program in mpi mode referenced by a top level program in mpi mode will have terrible problems. This means you can't mpi-multiprocess inside an mpi-multiprocess.
        """

        if mode == "none" or mode == "None":
            mode = None

        self.HAS_MPI = self.test_import_MPI(mode, flag_for_low_level)

        # Pick the mode
        self.pick_mode = self.pick_mode()

        if mode == None:
            if self.pick_mode == "mpi" and self.HAS_MPI == True:
                # THIS IS TO BE RUN IN MPI
                self.mode = "mpi"
            else:
                self.mode = "multiprocessing"

        elif mode == "mpi":
            if self.HAS_MPI == True:
                # THIS IS EXPLICITILY CHOSEN TO BE RUN IN MPI AND CAN WORK WITH MPI
                self.mode = "mpi"
            else:
                raise Exception("mpi4py package must be available to use mpi mode")

        elif mode == "multiprocessing":
            self.mode = "multiprocessing"

        elif mode == "Serial" or mode == "serial":
            self.mode = "serial"

        else:
            # Default setting will be multiprocessing
            self.mode = "multiprocessing"

        # Start MPI MODE if applicable
        if self.mode == "mpi":
            self.parallel_obj = self.start(self.mode)

        else:
            self.parallel_obj = None

        if self.mode == "serial":
            self.num_procs = 1

        elif num_procs == None:
            self.num_procs = self.compute_nodes()

        elif num_procs >= 1:
            self.num_procs = num_procs

        else:
            self.num_procs = self.compute_nodes()

    def test_import_MPI(self, mode, flag_for_low_level=False):
        """
        This tests for the ability of importing the MPI sublibrary from mpi4py.

        This import is problematic when run inside a program which was already mpi parallelized (ie a program run inside an mpi program)
            - for some reason from mpi4py import MPI is problematic in this sub-program structuring.
        To prevent these errors we do a quick check outside the class with a Try statement to import mpi4py
            - if it can't do the import mpi4py than the API isn't installed and we can't run MPI, so we won't even attempt from mpi4py import MPI

        it then checks if the mode has been already establish or if there is a low level flag.

        If the user explicitly or implicitly asks for mpi  (ie mode=None or mode="mpi") without flags and mpi4py is installed, then we will
            run the from mpi4py import MPI check. if it passes then we will return a True and run mpi mode; if not we return False and run multiprocess

        Inputs:
        :param str mode: the multiprocess mode to be used, ie) serial, multiprocessing, mpi, or None:
                            if None then we will try to pick a possible multiprocessing choice. This should only be used for
                            top level coding. It is best practice to specify which multiprocessing choice to use.
                            if you have smaller programs used by a larger program, with both mpi enabled there will be problems, so specify multiprocessing is important.
        :param int num_procs:   the number of processors or nodes that will be used. If None than we will use all available nodes/processors
                                        This will be overriden and fixed to a single processor if mode==serial
        :param bol flag_for_low_level: this will override mode and number of processors and set it to a multiprocess as serial. This is useful because
                                a low-level program in mpi mode referenced by a top level program in mpi mode will have terrible problems. This means you can't mpi-multiprocess inside an mpi-multiprocess.

        Returns:
        :returns: bol bol:  Returns True if MPI can be run and there aren't any flags against running mpi mode
                        Returns False if it cannot or should not run mpi mode.
        """
        if MPI_installed == False:
            # mpi4py isn't installed and we will need to multiprocess
            return False

        if flag_for_low_level == True:
            # Flagged for low level and testing import mpi4py.MPI can be a problem
            return False

        if mode == "mpi" or mode == "None" or mode == None:
            # This must be either mpi or None, mpi4py can be installed and it hasn't been flagged at low level

            # Before executing Parallelizer with mpi4py (which override python raise Exceptions)
            # We must check that it is being run with the "-m mpi4py" runpy flag
            # Although this is lower priority over mpi4py version (as mpi4py.__versions__ less than 2.1.0 do not offer the -m feature)
            #      This should get checked before loading the mpi4py api
            sys_modules = sys.modules
            if "runpy" not in sys_modules.keys():
                printout = "\nTo run in mpi mode you must run with -m flag. ie) mpirun -n $NTASKS python -m mpi4py run_gypsum_dl.py\n"
                print(printout)
                return False

            try:
                import mpi4py
                from mpi4py import MPI

                mpi4py_version = mpi4py.__version__
                mpi4py_version = [int(x) for x in mpi4py_version.split(".")]

                if mpi4py_version[0] == 2:
                    if mpi4py_version[1] < 1:
                        print(
                            "\nmpi4py version 2.1.0 or higher is required. Use the 'python -m mpi4py' flag to run in mpi mode.\nPlease update mpi4py to a newer version, or switch job_manager to multiprocessing or serial.\n"
                        )
                        return False
                elif mpi4py_version[0] < 2:
                    print(
                        "\nmpi4py version 2.1.0 or higher is required. Use the 'python -m mpi4py' flag to run in mpi mode.\nPlease update mpi4py to a newer version, or switch job_manager to multiprocessing or serial.\n"
                    )
                    return False

                return True
            except:
                return False

        return False

    def start(self, mode=None):
        """
        One must call this method before `run()` in order to configure MPI parallelization

        This creates the object for parallizing in a given mode.

        mode=None can be used at a top level program, but if using a program enabled
        with this multiprocess, referenced by a top level program, make sure the mode
        is explicitly chosen.

        Inputs:
        :param str mode: the multiprocess mode to be used, ie) serial, multiprocessing, mpi, or None:
                            if None then we will try to pick a possible multiprocessing choice. This should only be used for
                            top level coding. It is best practice to specify which multiprocessing choice to use.
                            if you have smaller programs used by a larger program, with both mpi enabled there will be problems, so specify multiprocessing is important.
        Returns:
        :returns: class parallel_obj: This is the obstantiated object of the class of parallizations.
                            ie) if self.mode=='mpi' self.parallel_obj will be an instance of the mpi class
                                This style of retained parallel_obj to be used later is important because this is the object which controls the work nodes and maintains the mpi universe
                            self.parallel_obj will be set to None for simpler parallization methods like serial
        """

        if mode == None:
            mode = self.mode

        if mode == "mpi":
            if self.HAS_MPI == True:
                # THIS IS EXPLICITILY CHOSEN TO BE RUN IN MPI AND CAN WORK WITH MPI
                ParallelMPI_obj = ParallelMPI()
                ParallelMPI_obj.start()
                return ParallelMPI_obj
            else:
                raise Exception("mpi4py package must be available to use mpi mode")

        else:
            return None

    def end(self, mode=None):
        """
        Call this method before exit to terminate MPI workers


        Inputs:
        :param str mode: the multiprocess mode to be used, ie) serial, multiprocessing, mpi, or None:
                            if None then we will try to pick a possible multiprocessing choice. This should only be used for
                            top level coding. It is best practice to specify which multiprocessing choice to use.
                            if you have smaller programs used by a larger program, with both mpi enabled there will be problems, so specify multiprocessing is important.
        """

        if mode == None:
            mode = self.mode
        if mode == "mpi":

            if self.HAS_MPI == True and self.parallel_obj != None:
                # THIS IS EXPLICITILY CHOSEN TO BE RUN IN MPI AND CAN WORK WITH MPI
                self.parallel_obj.end()

            else:
                raise Exception("mpi4py package must be available to use mpi mode")

    def run(self, args, func, num_procs=None, mode=None):
        """
        Run a task in parallel across the system.

        Mode can be one of 'mpi', 'multiprocessing' or 'none' (serial). If it is not
        set, the best value will be determined automatically.

        By default, this method will use the full resources of the system. However,
        if the mode is set to 'multiprocessing', num_procs can control the number
        of threads initialized when it is set to a nonzero value.

        Example: If one wants to multiprocess function  def foo(x,y) which takes 2 ints and one wants to test all permutations of x and y between 0 and 2:
                    args = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)]
                    func = foo      The namespace of foo


        Inputs:
        :param python_obj func: This is the object of the function which will be used.
        :param list args: a list of lists/tuples, each sublist/tuple must contain all information required by the function for a single object which will be multiprocessed
        :param int num_procs:  (Primarily for Developers)  the number of processors or nodes that will be used. If None than we will use all available nodes/processors
                                        This will be overriden and fixed to a single processor if mode==serial
        :param str mode:  (Primarily for Developers) the multiprocess mode to be used, ie) serial, multiprocessing, mpi, or None:
                            if None then we will try to pick a possible multiprocessing choice. This should only be used for
                            top level coding. It is best practice to specify which multiprocessing choice to use.
                            if you have smaller programs used by a larger program, with both mpi enabled there will be problems, so specify multiprocessing is important.
                            BEST TO LEAVE THIS BLANK
        Returns:
        :returns: list results: A list containing all the results from the multiprocess
        """

        # determine the mode
        if mode == None:
            mode = self.mode
        else:
            if self.mode != mode:
                if mode != "mpi" and mode != "serial" and mode != "multiprocessing":
                    printout = (
                        "Overriding function with a multiprocess mode which doesn't match: "
                        + mode
                    )
                    raise Exception(printout)
                if mode == "mpi":
                    printout = (
                        "Overriding multiprocess can't go from non-mpi to mpi mode"
                    )
                    raise Exception(printout)

        if num_procs == None:
            num_procs = self.num_procs

        if num_procs != self.num_procs:
            if mode == "serial":
                printout = "Can't override num_procs in serial mode"
                raise Exception(printout)

        if mode != self.mode:
            printout = "changing mode from {} to {} for development purpose".format(
                mode, self.mode
            )
            print(printout)
        # compute
        if mode == "mpi":
            if not self.HAS_MPI:
                raise Exception("mpi4py package must be available to use mpi mode")

            return self.parallel_obj.run(func, args)

        elif mode == "multiprocessing":
            return MultiThreading(args, num_procs, func)
        else:
            # serial is running the ParallelThreading with num_procs=1
            return MultiThreading(args, 1, func)

    def pick_mode(self):
        """
        Determines the parallelization cababilities of the system and returns one
        of the following modes depending on the configuration:

        Returns:
        :returns: str mode: the mode which is to be used 'mpi', 'multiprocessing', 'serial'
        """
        # check if mpi4py is loaded and we have more than one processor in our MPI world
        if self.HAS_MPI:
            try:
                if mpi4py.MPI.COMM_WORLD.Get_size() > 1:
                    return "mpi"
            except:
                return "multiprocessing"
            else:
                return "multiprocessing"
        # # check if we could utilize more than one processor
        # if multiprocessing.cpu_count() > 1:
        #     return 'multiprocessing'

        # default to multiprocessing
        return "multiprocessing"

    def return_mode(self):
        """
        Returns the mode chosen for the parallelization cababilities of the system and returns one
        of the following modes depending on the configuration:
        :param str mode: the multiprocess mode to be used, ie) serial, multiprocessing, mpi, or None:
                    if None then we will try to pick a possible multiprocessing choice. This should only be used for
                    top level coding. It is best practice to specify which multiprocessing choice to use.
                    if you have smaller programs used by a larger program, with both mpi enabled there will be problems, so specify multiprocessing is important.
                    BEST TO LEAVE THIS BLANK
        Returns:
        :returns: str mode: the mode which is to be used 'mpi', 'multiprocessing', 'serial'
        """
        return self.mode

    def compute_nodes(self, mode=None):
        """
        Computes the number of "compute nodes" according to the selected mode.

        For mpi, this is the universe size
        For multiprocessing this is the number of available cores
        For serial, this value is 1
        Returns:
        :returns: int num_procs: the number of nodes/processors which is to be used
        """
        if mode is None:
            mode = self.mode

        if mode == "mpi":
            if not self.HAS_MPI:
                raise Exception("mpi4py package must be available to use mpi mode")
            return mpi4py.MPI.COMM_WORLD.Get_size()
        elif mode == "multiprocessing":
            return multiprocessing.cpu_count()
        else:
            return 1

    def return_node(self):
        """
        Returns the number of "compute nodes" according to the selected mode.

        For mpi, this is the universe size
        For multiprocessing this is the number of available cores
        For serial, this value is 1
        Returns:
        :returns: int num_procs: the number of nodes/processors which is to be used
        """
        return self.num_procs


class ParallelMPI(object):
    """
    Utility code for running tasks in parallel across an MPI cluster.
    """

    def __init__(self):
        """
        Default num_procs is all the processesors possible
        """

        self.COMM = mpi4py.MPI.COMM_WORLD

        self.Empty_object = Empty_obj()

    def start(self):
        """
        Call this method at the beginning of program execution to put non-root processors
        into worker mode.
        """

        rank = self.COMM.Get_rank()

        if rank == 0:
            return
        else:
            worker = self._worker()

    def end(self):
        """
        Call this method to terminate worker processes
        """

        self.COMM.bcast(None, root=0)

    def _worker(self):
        """
        Worker processors wait in this function to receive new jobs
        """
        while True:
            # receive function for new job
            func = self.COMM.bcast(None, root=0)

            # kill signal
            if func is None:
                exit(0)

            # receive arguments
            args_chunk = self.COMM.scatter([], root=0)

            if type(args_chunk[0]) == type(
                self.Empty_object
            ):  # or  args_chunk[0] == [[self.Empty_object]]:
                result_chunk = [[self.Empty_object]]
                result_chunk = self.COMM.gather(result_chunk, root=0)

            else:
                # perform the calculation and send results
                result_chunk = [
                    func(*arg)
                    for arg in args_chunk
                    if type(arg[0]) != type(self.Empty_object)
                ]
                result_chunk = self.COMM.gather(result_chunk, root=0)

    def handle_undersized_jobs(self, arr, n):
        if len(arr) > n:
            printout = "the length of the package is bigger than the length of the number of nodes!"
            print(printout)
            raise Exception(printout)

        filler_slot = [[self.Empty_object]]
        while len(arr) < n:
            arr.append(filler_slot)
            if len(arr) == n:
                break

        return arr

    def _split(self, arr, n):
        """
        Takes an array of items and splits it into n "equally" sized
        chunks that can be provided to a worker cluster.
        """

        s = len(arr) // n
        remainder = len(arr) - int(s) * int(n)

        chuck_list = []
        temp = []
        counter = 0
        for x in range(0, len(arr)):

            # add 1 per group until remainder is removed
            if remainder != 0:
                r = 1
                if counter == s + 1:
                    remainder = remainder - 1
            else:
                r = 0

            if counter == s + r:
                chuck_list.append(temp)
                temp = []
                counter = 1
            else:
                counter += 1

            temp.append(list(arr[x]))
            if x == len(arr) - 1:
                chuck_list.append(temp)

        if len(chuck_list) != n:
            chuck_list = self.handle_undersized_jobs(chuck_list, n)

        return chuck_list

    def _join(self, arr):
        """
        Joins a "list of lists" that was previously split by _split().

        Returns a single list.
        """
        arr = tuple(arr)
        arr = [x for x in arr if type(x) != type(self.Empty_object)]
        arr = [a for sub in arr for a in sub]
        arr = [x for x in arr if type(x) != type(self.Empty_object)]
        return arr

    def check_and_format_args(self, args):
        # Make sure args is a list of lists
        if type(args) != list and type(args) != tuple:
            printout = "args must be a list of lists"
            print(printout)
            raise Exception(printout)

        item_type = type(args[0])
        for i in range(0, len(args)):
            if type(args[i]) == item_type:
                continue
            else:
                printout = "all items within args must be the same type and must be either a list or tuple"
                print(printout)
                raise Exception(printout)
        if item_type == list:
            return args
        elif item_type == tuple:
            args = [list(x) for x in args]
            return args
        else:
            printout = "all items within args must be either a list or tuple"
            print(printout)
            raise Exception(printout)

    def run(self, func, args):
        """
        Run a function in parallel across the current MPI cluster.

        * func is a pure function of type (A)->(B)
        * args is a list of type list(A)

        This method batches the computation across the MPI cluster and returns
        the result of type list(B) where result[i] = func(args[i]).

        Important note: func must exist in the namespace at initialization.
        """
        num_of_args_start = len(args)
        if len(args) == 0:
            return []
        args = self.check_and_format_args(args)

        size = self.COMM.Get_size()

        # broadcast function to worker processors
        self.COMM.bcast(func, root=0)

        # chunkify the argument list
        args_chunk = self._split(args, size)

        # scatter argument chunks to workers
        args_chunk = self.COMM.scatter(args_chunk, root=0)

        if type(args_chunk) != list:
            raise Exception("args_chunk needs to be a list")

        # perform the calculation and get results
        result_chunk = [func(*arg) for arg in args_chunk]
        sys.stdout.flush()

        result_chunk = self.COMM.gather(result_chunk, root=0)

        if type(result_chunk) != list:
            raise Exception("result_chunk needs to be a list")

        # group results
        results = self._join(result_chunk)

        if len(results) != num_of_args_start:
            results = [x for x in results if type(x) != type(self.Empty_object)]
            results = flatten_list(results)

        if type(results) != list:
            raise Exception("results needs to be a list")

        results = [x for x in results if type(x) != type(self.Empty_object)]
        sys.stdout.flush()
        return results


#


class Empty_obj(object):
    """
    Create a unique Empty Object to hand to empty processors
    """

    pass


#


"""
Run commands on multiple processors in python.

Adapted from examples on https://docs.python.org/2/library/multiprocessing.html
"""



def MultiThreading(inputs, num_procs, task_name):
    """Initialize this object.

    Args:
        inputs ([data]): A list of data. Each datum contains the details to
            run a single job on a single processor.
        num_procs (int): The number of processors to use.
        task_class_name (class): The class that governs what to do for each
            job on each processor.
    """

    results = []

    # If there are no inputs, just return an empty list.
    if len(inputs) == 0:
        return results

    inputs = check_and_format_inputs_to_list_of_tuples(inputs)

    num_procs = count_processors(len(inputs), num_procs)

    tasks = []


    for index, item in enumerate(inputs):
        if not isinstance(item, tuple):
            item = (item,)
        task = (index, (task_name, item))
        tasks.append(task)

    if num_procs == 1:
        for item in tasks:
            job, args = item[1]
            output = job(*args)
            results.append(output)
    else:
        results = start_processes(tasks, num_procs)

    return results


###
# Worker function
###


def worker(input, output):
    for seq, job in iter(input.get, "STOP"):
        func, args = job
        result = func(*args)
        ret_val = (seq, result)
        output.put(ret_val)


def check_and_format_inputs_to_list_of_tuples(args):
    # Make sure args is a list of tuples
    if type(args) != list and type(args) != tuple:
        printout = "args must be a list of tuples"
        print(printout)
        raise Exception(printout)

    item_type = type(args[0])
    for i in range(0, len(args)):
        if type(args[i]) == item_type:
            continue
        else:
            printout = "all items within args must be the same type and must be either a list or tuple"
            print(printout)
            raise Exception(printout)
    if item_type == tuple:
        return args
    elif item_type == list:
        args = [tuple(x) for x in args]
        return args
    else:
        printout = "all items within args must be either a list or tuple"
        print(printout)
        raise Exception(printout)


def count_processors(num_inputs, num_procs):
    """
    Checks processors available and returns a safe number of them to
    utilize.

    :param int num_inputs: The number of inputs.
    :param int num_procs: The number of desired processors.

    :returns: The number of processors to use.
    """
    # first, if num_procs <= 0, determine the number of processors to
    # use programatically
    if num_procs <= 0:
        num_procs = multiprocessing.cpu_count()

    # reduce the number of processors if too many have been specified
    if num_inputs < num_procs:
        num_procs = num_inputs

    return num_procs


def start_processes(inputs, num_procs):
    """
    Creates a queue of inputs and outputs
    """

    # Create queues
    task_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    # Submit tasks
    for item in inputs:
        task_queue.put(item)

    # Start worker processes
    for i in range(num_procs):
        multiprocessing.Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    results = []
    for i in range(len(inputs)):
        results.append(done_queue.get())

    # Tell child processes to stop
    for i in range(num_procs):
        task_queue.put("STOP")

    results.sort(key=lambda tup: tup[0])

    return [item[1] for item in map(list, results)]


###
# Helper functions
###


def flatten_list(tier_list):
    """
    Given a list of lists, this returns a flat list of all items.

    :params list tier_list: A 2D list.

    :returns: A flat list of all items.
    """
    if tier_list is None:
        return []
    already_flattened = False
    for item in tier_list:
        if type(item) != list:
            already_flattened = True
    if already_flattened == True:
        return tier_list

    flat_list = [item for sublist in tier_list for item in sublist]
    return flat_list


def strip_none(none_list):
    """
    Given a list that might contain None items, this returns a list with no
    None items.

    :params list none_list: A list that may contain None items.

    :returns: A list stripped of None items.
    """
    if none_list is None:
        return []
    results = [x for x in none_list if x != None]
    return results
