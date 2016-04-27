# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import contextlib

from unittest import TestCase
from unittest import TextTestRunner
from unittest import TestResult, TextTestResult

if 'PYTOAST_NOMPI' in os.environ.keys():
    from . import fakempi as MPI
else:
    from mpi4py import MPI


def testcase_name(test_method):
    testcase = type(test_method)

    # Ignore module name if it is '__main__'
    module = testcase.__module__ + '.'
    if module == '__main__.':
        module = ''
    result = module + testcase.__name__
    return result


class NoopStream(object):
    """
    A fake stream for non-root processes to use.
    """
    def write(self, *args):
        pass
    
    def writeln(self, *args):
        pass

    def flush(self):
        pass


class MPITestCase(TestCase):
    """
    A simple wrapper around the standard TestCase which provides
    one extra method to set the communicator.
    """
    def __init__(self, *args, **kwargs):
        super(MPITestCase, self).__init__(*args, **kwargs)

    def setComm(self, comm):
        self.comm = comm

    def print_in_turns(self, msg):
        for i in range(self.comm.size):
            if i == 0:
                if self.comm.rank == 0:
                    print('proc {:04d}: {}'.format(i, msg))
            else:
                if self.comm.rank == 0:
                    othermsg = self.comm.recv(source=i, tag=i)
                    print('proc {:04d}: {}'.format(i, othermsg))
                elif i == self.comm.rank:
                    self.comm.send(msg, dest=0, tag=i)
            self.comm.Barrier()


class MPITestInfo(object):
    """
    This class keeps useful information about the execution of a
    test method.
    """

    # Possible test outcomes
    (SUCCESS, FAILURE, ERROR, SKIP) = range(4)

    def __init__(self, test_result, test_method, outcome=SUCCESS, err=None, subTest=None):
        self.test_result = test_result
        self.outcome = outcome
        self.elapsed_time = 0
        self.err = err
        self.stdout = test_result._stdout_data
        self.stderr = test_result._stderr_data

        self.test_description = self.test_result.getDescription(test_method)
        self.test_exception_info = (
            '' if outcome in (self.SUCCESS, self.SKIP)
            else self.test_result._exc_info_to_string(
                    self.err, test_method)
        )

        self.test_name = testcase_name(test_method)
        self.test_id = test_method.id()
        if subTest:
            self.test_id = subTest.id()

    def id(self):
        return self.test_id

    def test_finished(self):
        """Save info that can only be calculated once a test has run.
        """
        self.elapsed_time = \
            self.test_result.stop_time - self.test_result.start_time

    def get_description(self):
        """
        Return a text representation of the test method.
        """
        return self.test_description

    def get_error_info(self):
        """
        Return a text representation of an exception thrown by a test
        method.
        """
        return self.test_exception_info


class MPITestResult(TextTestResult):
    """
    A test result class that gathers per-process results and writes
    them to a stream on the root process.

    Used by MPITestRunner.
    """
    def __init__(self, comm, stream, descriptions=1, verbosity=1):
        self.comm = comm
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.stream = stream
        TextTestResult.__init__(self, self.stream, descriptions, verbosity)
        self.buffer = True  # we are capturing test output
        self._stdout_data = None
        self._stderr_data = None
        self.successes = []
        self.callback = None
        self.properties = None  # junit testsuite properties

    def _prepare_callback(self, test_info, target_list, verbose_str,
                          short_str):
        """
        Appends a MPITestInfo to the given target list and sets a callback
        method to be called by stopTest method.
        """
        target_list.append(test_info)

        def callback():
            """Prints the test method outcome to the stream, as well as
            the elapsed time.
            """
            test_info.test_finished()

            if self.showAll:
                self.stream.writeln('{} ({:.3f}s)'.format(verbose_str, test_info.elapsed_time))
            elif self.dots:
                self.stream.write(short_str)

        self.callback = callback

    def startTest(self, test):
        """
        Called before executing each test method.
        """
        self.comm.Barrier()
        if isinstance(test, MPITestCase):
            test.setComm(self.comm)
        self.start_time = MPI.Wtime()
        TestResult.startTest(self, test)

        if self.showAll:
            self.stream.write('  ' + self.getDescription(test))
            self.stream.write(" ... ")

    def _save_output_data(self):
        self._stdout_data = sys.stdout.getvalue()
        self._stderr_data = sys.stderr.getvalue()

    def stopTest(self, test):
        """
        Called after executing each test method.
        """
        self._save_output_data()
        TextTestResult.stopTest(self, test)
        self.comm.Barrier()
        self.stop_time = MPI.Wtime()

        if self.callback and callable(self.callback):
            self.callback()
            self.callback = None

    def addSuccess(self, test):
        """
        Called when a test executes successfully.
        """
        self._save_output_data()
        self._prepare_callback(
            MPITestInfo(self, test), self.successes, 'OK', '.'
        )

    def addFailure(self, test, err):
        """
        Called when a test method fails.
        """
        self._save_output_data()
        testinfo = MPITestInfo(self, test, MPITestInfo.FAILURE, err)
        self.failures.append((
            testinfo,
            self._exc_info_to_string(err, test)
        ))
        self._prepare_callback(testinfo, [], 'FAIL', 'F')

    def addError(self, test, err):
        """
        Called when a test method raises an error.
        """
        self._save_output_data()
        testinfo = MPITestInfo(self, test, MPITestInfo.ERROR, err)
        self.errors.append((
            testinfo,
            self._exc_info_to_string(err, test)
        ))
        self._prepare_callback(testinfo, [], 'ERROR', 'E')

    def addSubTest(self, testcase, test, err):
        """
        Called when a subTest method raises an error.
        """
        if err is not None:
            self._save_output_data()
            testinfo = MPITestInfo(self, testcase, MPITestInfo.ERROR, err, subTest=test)
            self.errors.append((
                testinfo,
                self._exc_info_to_string(err, testcase)
            ))
            self._prepare_callback(testinfo, [], 'ERROR', 'E')

    def addSkip(self, test, reason):
        """
        Called when a test method was skipped.
        """
        self._save_output_data()
        testinfo = MPITestInfo(self, test, MPITestInfo.SKIP, reason)
        self.skipped.append((testinfo, reason))
        self._prepare_callback(testinfo, [], 'SKIP', 'S')

    def printErrorList(self, flavour, errors):
        """
        Writes information about the FAIL or ERROR to the stream.
        """
        for test_info, error in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln(
                '{} [{:.3f}s]: {}'.format(flavour, test_info.elapsed_time,
                                    test_info.get_description())
            )
            self.stream.writeln(self.separator2)
            self.stream.writeln('{}'.format(test_info.get_error_info()))

    def _get_info_by_testcase(self):
        """
        Organizes test results by TestCase module. This information is
        used during the report generation, where a XML report will be created
        for each TestCase.
        """
        tests_by_testcase = {}

        for tests in (self.successes, self.failures, self.errors,
                      self.skipped):
            for test_info in tests:
                if isinstance(test_info, tuple):
                    # This is a skipped, error or a failure test case
                    test_info = test_info[0]
                testcase_name = test_info.test_name
                if testcase_name not in tests_by_testcase:
                    tests_by_testcase[testcase_name] = []
                tests_by_testcase[testcase_name].append(test_info)

        return tests_by_testcase



class MPITestRunner(TextTestRunner):
    """
    A test runner class that collects output from all processes.
    """
    def __init__(self, stream=sys.stderr, descriptions=True, verbosity=1,
                 failfast=False, buffer=False):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        if self.rank == 0:
            self.stream = stream
        else:
            self.stream = NoopStream()
        TextTestRunner.__init__(self, self.stream, descriptions, verbosity,
                                failfast=failfast, buffer=buffer)
        self.verbosity = verbosity


    def _make_result(self):
        """
        Creates a TestResult object which will be used to store
        information about the executed tests.
        """
        return MPITestResult(self.comm, self.stream, self.descriptions, 
            self.verbosity)


    def run(self, test):
        """
        Runs the given test case or test suite.
        """
        try:
            # Prepare the test execution
            result = self._make_result()
            if hasattr(test, 'properties'):
                # junit testsuite properties
                result.properties = test.properties

            # Print a nice header
            self.stream.writeln()
            self.stream.writeln('Running tests...')
            self.stream.writeln(result.separator2)

            # Execute tests
            start_time = MPI.Wtime()
            test(result)
            stop_time = MPI.Wtime()
            time_taken = stop_time - start_time

            # Print results
            result.printErrors()
            self.stream.writeln(result.separator2)
            run = result.testsRun
            self.stream.writeln("Ran {} test{} in {:.3f}s".format(run, (run != 1) and "s" or "", time_taken))
            self.stream.writeln()

            expectedFails = unexpectedSuccesses = skipped = 0
            try:
                results = map(len, (result.expectedFailures,
                                    result.unexpectedSuccesses,
                                    result.skipped))
            except AttributeError:
                pass
            else:
                expectedFails, unexpectedSuccesses, skipped = results

            # Error traces
            infos = []
            if not result.wasSuccessful():
                self.stream.write("FAILED")
                failed, errored = map(len, (result.failures, result.errors))
                if failed:
                    infos.append("failures={0}".format(failed))
                if errored:
                    infos.append("errors={0}".format(errored))
            else:
                self.stream.write("OK")

            if skipped:
                infos.append("skipped={0}".format(skipped))
            if expectedFails:
                infos.append("expected failures={0}".format(expectedFails))
            if unexpectedSuccesses:
                infos.append("unexpected successes={0}".format(
                    unexpectedSuccesses))

            if infos:
                self.stream.writeln(" ({0})".format(", ".join(infos)))
            else:
                self.stream.write("\n")

        finally:
            pass

        return result



