# Based on runtests.py in the mpinoseutils project here:
# https://github.com/dagss/mpinoseutils
#

from mpi4py import MPI

import nose

from nose.plugins import Plugin


class NoopStream(object):
    def write(self, *args):
        pass
    
    def writeln(self, *args):
        pass

    def flush(self):
        pass


class MpiOutput(Plugin):
    """
    Have only rank 0 report test results. Test results are aggregated
    across processes, i.e., if an exception happens in a single
    process then that is reported, otherwise if an assertion failed in any
    process then that is reported, otherwise it's a success.
    """
    # Required attributes:
    
    name = 'mpi'
    enabled = True

    def __init__(self, comm):
        super(MpiOutput, self).__init__()
        self.comm = comm
        self.rank = self.comm.rank
        self.procs = self.comm.size

    def formatErr(self, err):
        exctype, value, tb = err
        return ''.join(traceback.format_exception(exctype, value, tb))

    def addSuccess(self, test):
        print("success from rank {}".format(self.rank))

        
    def addError(self, test, err):
        err = self.formatErr(err)
        print("Error {} from rank {}".format(err, self.rank))


    def addFailure(self, test, err):
        err = self.formatErr(err)
        print("Failure {} from rank {}".format(err, self.rank))


    def finalize(self, result):
        print("finalize from rank {}".format(self.rank))
        

    def setOutputStream(self, stream):
        if not is_root:
            return NoopStream()
        else:
            return None


    def startContext(self, ctx):
        try:
            n = ctx.__name__
        except AttributeError:
            n = str(ctx)
        try:
            path = ctx.__file__.replace('.pyc', '.py')
        except AttributeError:
            pass


    def stopContext(self, ctx):
        print("stopContext from rank {}".format(self.rank))
    

    def startTest(self, test):
        print("startTest from rank {}".format(self.rank))
        

    def stopTest(self, test):
        print("stopTest from rank {}".format(self.rank))



if __name__ == '__main__':
    import sys
    import os

    args = sys.argv
    args += ['--nocapture', '--verbose']

    nose.main(addplugins=[MpiOutput(MPI.COMM_WORLD)], argv=args)
