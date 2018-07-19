# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import os
import sys
import time
import traceback

import warnings

from unittest.signals import registerResult
from unittest import TestCase
from unittest import TestResult

from functools import wraps


class MPITestCase(TestCase):
    """
    A simple wrapper around the standard TestCase which provides
    one extra method to set the communicator.
    """
    def __init__(self, *args, **kwargs):
        super(MPITestCase, self).__init__(*args, **kwargs)

    def setComm(self, comm):
        self.comm = comm


class MPITestResult(TestResult):
    """A test result class that can print formatted text results to a stream.

    The actions needed are coordinated across all processes.

    Used by MPITestRunner.
    """
    separator1 = "=" * 70
    separator2 = "-" * 70

    def __init__(self, comm, stream=None, descriptions=None, verbosity=None,
        **kwargs):
        super(MPITestResult, self).__init__(stream=stream,
            descriptions=descriptions, verbosity=verbosity, **kwargs)
        self.comm = comm
        self.stream = stream
        self.descriptions = descriptions
        self.buffer = False
        self.failfast = False


    def getDescription(self, test):
        doc_first_line = test.shortDescription()
        if self.descriptions and doc_first_line:
            return "\n".join((str(test), doc_first_line))
        else:
            return str(test)


    def startTest(self, test):
        if isinstance(test, MPITestCase):
            test.setComm(self.comm)
        self.stream.flush()
        self.comm.barrier()
        if self.comm.rank == 0:
            self.stream.write("\n")
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")
            self.stream.flush()
        self.comm.barrier()
        super(MPITestResult, self).startTest(test)
        return


    def addSuccess(self, test):
        super(MPITestResult, self).addSuccess(test)
        self.stream.write("[{}]ok ".format(self.comm.rank))
        self.stream.flush()
        return


    def addError(self, test, err):
        super(MPITestResult, self).addError(test, err)
        self.stream.write("[{}]error ".format(self.comm.rank))
        self.stream.flush()
        return


    def addFailure(self, test, err):
        super(MPITestResult, self).addFailure(test, err)
        self.stream.write("[{}]fail ".format(self.comm.rank))
        self.stream.flush()
        return


    def addSkip(self, test, reason):
        super(MPITestResult, self).addSkip(test, reason)
        self.stream.write("[{}]skipped({}) ".format(self.comm.rank, reason))
        self.stream.flush()
        return


    def addExpectedFailure(self, test, err):
        super(MPITestResult, self).addExpectedFailure(test, err)
        self.stream.write("[{}]expected-fail ".format(self.comm.rank))
        self.stream.flush()
        return


    def addUnexpectedSuccess(self, test):
        super(MPITestResult, self).addUnexpectedSuccess(test)
        self.stream.writeln("[{}]unexpected-success ".format(self.comm.rank))
        return


    def printErrorList(self, flavour, errors):
        for test, err in errors:
            self.stream.writeln("[{}] {}".format(self.comm.rank,
                self.separator1))
            self.stream.writeln(\
                "[{}] {}: {}".format(self.comm.rank,
                flavour, self.getDescription(test)))
            self.stream.writeln("[{}] {}".format(self.comm.rank,
                    self.separator2))
            self.stream.writeln("[{}] {}".format(self.comm.rank, err))
        return


    def printErrors(self):
        self.comm.barrier()
        if self.comm.rank == 0:
            self.stream.writeln()
            self.stream.flush()
        for p in range(self.comm.size):
            if p == self.comm.rank:
                self.printErrorList("ERROR", self.errors)
                self.printErrorList("FAIL", self.failures)
                self.stream.flush()
            self.comm.barrier()
        return


    def allSuccessful(self):
        mysuccess = self.wasSuccessful()
        total = 0
        if not mysuccess:
            total = 1
        alltotal = self.comm.allreduce(total)
        if alltotal == 0:
            return True
        else:
            return False


class _WritelnDecorator(object):
    """Used to decorate file-like objects with a handy "writeln" method"""
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, attr):
        if attr in ("stream", "__getstate__"):
            raise AttributeError(attr)
        return getattr(self.stream,attr)

    def writeln(self, arg=None):
        if arg:
            self.write(arg)
        self.write("\n") # text-mode streams translate to \r\n if needed


class MPITestRunner(object):
    """A test runner class that displays results in textual form.

    It prints out the names of tests as they are run, errors as they
    occur, and a summary of the results at the end of the test run.

    This information is only printed by the root process.
    """
    resultclass = MPITestResult

    def __init__(self, comm, stream=None, descriptions=True, verbosity=2,
        warnings=None):
        """Construct a MPITestRunner.

        Subclasses should accept **kwargs to ensure compatibility as the
        interface changes.
        """
        self.comm = comm
        if stream is None:
            stream = sys.stderr
        self.stream = _WritelnDecorator(stream)
        self.descriptions = descriptions
        self.verbosity = verbosity
        self.warnings = warnings

    def run(self, test):
        "Run the given test case or test suite."
        result = MPITestResult(self.comm, self.stream, self.descriptions,
            self.verbosity)
        registerResult(result)
        with warnings.catch_warnings():
            if self.warnings:
                # if self.warnings is set, use it to filter all the warnings
                warnings.simplefilter(self.warnings)
                # if the filter is "default" or "always", special-case the
                # warnings from the deprecated unittest methods to show them
                # no more than once per module, because they can be fairly
                # noisy.  The -Wd and -Wa flags can be used to bypass this
                # only when self.warnings is None.
                if self.warnings in ["default", "always"]:
                    warnings.filterwarnings("module",
                            category=DeprecationWarning,
                            message=r"Please use assert\w+ instead.")
            startTime = time.time()
            startTestRun = getattr(result, "startTestRun", None)
            if startTestRun is not None:
                startTestRun()
            try:
                test(result=result)
            finally:
                stopTestRun = getattr(result, "stopTestRun", None)
                if stopTestRun is not None:
                    stopTestRun()
            stopTime = time.time()
        self.stream.flush()
        self.comm.barrier()
        if self.comm.rank == 0:
            self.stream.write("\n")
            self.stream.flush()
        timeTaken = stopTime - startTime

        result.printErrors()

        if self.comm.rank == 0:
            if hasattr(result, "separator2"):
                self.stream.writeln(result.separator2)

        run = result.testsRun
        if self.comm.rank == 0:
            self.stream.writeln("Ran %d test%s in %.3fs" %
                                (run, run != 1 and "s" or "", timeTaken))
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

        for p in range(self.comm.size):
            if p == self.comm.rank:
                infos = []
                if not result.wasSuccessful():
                    self.stream.write("[{}] FAILED".format(self.comm.rank))
                    failed, errored = len(result.failures), len(result.errors)
                    if failed:
                        infos.append("failures=%d" % failed)
                    if errored:
                        infos.append("errors=%d" % errored)
                else:
                    self.stream.write("[{}] OK".format(self.comm.rank))

                if skipped:
                    infos.append("skipped=%d" % skipped)
                if expectedFails:
                    infos.append("expected failures=%d" % expectedFails)
                if unexpectedSuccesses:
                    infos.append("unexpected successes=%d" % unexpectedSuccesses)
                if infos:
                    self.stream.writeln(" ({})".format(", ".join(infos)))
                else:
                    self.stream.write("\n")
                self.stream.flush()
            self.comm.barrier()

        # if not result.allSuccessful():
        #     self.comm.Abort(1)

        return result









#
#     def print_in_turns(self, msg):
#         for i in range(self.comm.size):
#             if i == 0:
#                 if self.comm.rank == 0:
#                     print("proc {:04d}: {}".format(i, msg))
#             else:
#                 if self.comm.rank == 0:
#                     othermsg = self.comm.recv(source=i, tag=i)
#                     print("proc {:04d}: {}".format(i, othermsg))
#                 elif i == self.comm.rank:
#                     self.comm.send(msg, dest=0, tag=i)
#             self.comm.Barrier()
#
#
# class MPITestInfo(object):
#     """
#     This class keeps useful information about the execution of a
#     test method.
#     """
#
#     # Possible test outcomes
#     (SUCCESS, FAILURE, ERROR, SKIP) = range(4)
#
#     def __init__(self, test_result, test_method, outcome=SUCCESS, err=None, subTest=None):
#         self.test_result = test_result
#         self.outcome = outcome
#         self.elapsed_time = 0
#         self.err = err
#         self.stdout = test_result._stdout_data
#         self.stderr = test_result._stderr_data
#
#         self.test_description = self.test_result.getDescription(test_method)
#         self.test_exception_info = (
#             "" if outcome in (self.SUCCESS, self.SKIP)
#             else self.test_result._exc_info_to_string(
#                     self.err, test_method)
#         )
#
#         self.test_name = testcase_name(test_method)
#         self.test_id = test_method.id()
#         if subTest:
#             self.test_id = subTest.id()
#
#     def id(self):
#         return self.test_id
#
#     def test_finished(self):
#         """Save info that can only be calculated once a test has run.
#         """
#         self.elapsed_time = \
#             self.test_result.stop_time - self.test_result.start_time
#
#     def get_description(self):
#         """
#         Return a text representation of the test method.
#         """
#         return self.test_description
#
#     def get_error_info(self):
#         """
#         Return a text representation of an exception thrown by a test
#         method.
#         """
#         return self.test_exception_info
#
#
# class MPITestResult(MPITestResult):
#     """
#     A test result class that gathers per-process results and writes
#     them to a stream on the root process.
#
#     Used by MPITestRunner.
#     """
#     def __init__(self, comm, stream, descriptions=1, verbosity=1):
#         self.comm = comm
#         self.rank = self.comm.rank
#         self.size = self.comm.size
#         self.stream = stream
#         MPITestResult.__init__(self, self.stream, descriptions, verbosity)
#         self.buffer = True  # we are capturing test output
#         self._stdout_data = None
#         self._stderr_data = None
#         self.successes = []
#         self.callback = None
#         self.properties = None  # junit testsuite properties
#
#     def _prepare_callback(self, test_info, target_list, verbose_str,
#                           short_str):
#         """
#         Appends a MPITestInfo to the given target list and sets a callback
#         method to be called by stopTest method.
#         """
#         target_list.append(test_info)
#
#         def callback():
#             """Prints the test method outcome to the stream, as well as
#             the elapsed time.
#             """
#             test_info.test_finished()
#
#             if self.showAll:
#                 self.stream.writeln("{} ({:.3f}s)".format(verbose_str, test_info.elapsed_time))
#             elif self.dots:
#                 self.stream.write(short_str)
#
#         self.callback = callback
#
#     def startTest(self, test):
#         """
#         Called before executing each test method.
#         """
#         self.comm.Barrier()
#         if isinstance(test, MPITestCase):
#             test.setComm(self.comm)
#         self.start_time = MPI.Wtime()
#         TestResult.startTest(self, test)
#
#         if self.showAll:
#             self.stream.write("  " + self.getDescription(test))
#             self.stream.write(" ... ")
#
#     def _save_output_data(self):
#         self._stdout_data = sys.stdout.getvalue()
#         self._stderr_data = sys.stderr.getvalue()
#
#     def stopTest(self, test):
#         """
#         Called after executing each test method.
#         """
#         self._save_output_data()
#         MPITestResult.stopTest(self, test)
#         self.comm.Barrier()
#         self.stop_time = MPI.Wtime()
#
#         if self.callback and callable(self.callback):
#             self.callback()
#             self.callback = None
#
#     def addSuccess(self, test):
#         """
#         Called when a test executes successfully.
#         """
#         self._save_output_data()
#         self._prepare_callback(
#             MPITestInfo(self, test), self.successes, "OK", "."
#         )
#
#     def addFailure(self, test, err):
#         """
#         Called when a test method fails.
#         """
#         self._save_output_data()
#         testinfo = MPITestInfo(self, test, MPITestInfo.FAILURE, err)
#         self.failures.append((
#             testinfo,
#             self._exc_info_to_string(err, test)
#         ))
#         self._prepare_callback(testinfo, [], "FAIL", "F")
#
#     def addError(self, test, err):
#         """
#         Called when a test method raises an error.
#         """
#         self._save_output_data()
#         testinfo = MPITestInfo(self, test, MPITestInfo.ERROR, err)
#         self.errors.append((
#             testinfo,
#             self._exc_info_to_string(err, test)
#         ))
#         self._prepare_callback(testinfo, [], "ERROR", "E")
#
#     def addSubTest(self, testcase, test, err):
#         """
#         Called when a subTest method raises an error.
#         """
#         if err is not None:
#             self._save_output_data()
#             testinfo = MPITestInfo(self, testcase, MPITestInfo.ERROR, err, subTest=test)
#             self.errors.append((
#                 testinfo,
#                 self._exc_info_to_string(err, testcase)
#             ))
#             self._prepare_callback(testinfo, [], "ERROR", "E")
#
#     def addSkip(self, test, reason):
#         """
#         Called when a test method was skipped.
#         """
#         self._save_output_data()
#         testinfo = MPITestInfo(self, test, MPITestInfo.SKIP, reason)
#         self.skipped.append((testinfo, reason))
#         self._prepare_callback(testinfo, [], "SKIP", "S")
#
#     def printErrorList(self, flavour, errors):
#         """
#         Writes information about the FAIL or ERROR to the stream.
#         """
#         for test_info, error in errors:
#             self.stream.writeln(self.separator1)
#             self.stream.writeln(
#                 "{} [{:.3f}s]: {}".format(flavour, test_info.elapsed_time,
#                                     test_info.get_description())
#             )
#             self.stream.writeln(self.separator2)
#             self.stream.writeln("{}".format(test_info.get_error_info()))
#
#     def _get_info_by_testcase(self):
#         """
#         Organizes test results by TestCase module. This information is
#         used during the report generation, where a XML report will be created
#         for each TestCase.
#         """
#         tests_by_testcase = {}
#
#         for tests in (self.successes, self.failures, self.errors,
#                       self.skipped):
#             for test_info in tests:
#                 if isinstance(test_info, tuple):
#                     # This is a skipped, error or a failure test case
#                     test_info = test_info[0]
#                 testcase_name = test_info.test_name
#                 if testcase_name not in tests_by_testcase:
#                     tests_by_testcase[testcase_name] = []
#                 tests_by_testcase[testcase_name].append(test_info)
#
#         return tests_by_testcase
#
#
#
# class MPITestRunner(TextTestRunner):
#     """
#     A test runner class that collects output from all processes.
#     """
#     def __init__(self, stream=sys.stderr, descriptions=True, verbosity=1,
#                  failfast=False, buffer=False):
#         self.comm = MPI.COMM_WORLD
#         self.rank = self.comm.rank
#         self.size = self.comm.size
#         if self.rank == 0:
#             self.stream = stream
#         else:
#             self.stream = stream
#             #self.stream = NoopStream()
#         TextTestRunner.__init__(self, self.stream, descriptions, verbosity,
#                                 failfast=failfast, buffer=buffer)
#         self.verbosity = verbosity
#
#     def _make_result(self):
#         """
#         Creates a TestResult object which will be used to store
#         information about the executed tests.
#         """
#         return MPITestResult(self.comm, self.stream, self.descriptions,
#             self.verbosity)
#
#
#     def run(self, test):
#         """
#         Runs the given test case or test suite.
#         """
#         try:
#             # Prepare the test execution
#             result = self._make_result()
#             if hasattr(test, "properties"):
#                 # junit testsuite properties
#                 result.properties = test.properties
#
#             # Print a nice header
#             self.stream.writeln()
#             self.stream.writeln("Running Python tests...")
#             self.stream.writeln(result.separator2)
#
#             # Execute tests
#             start_time = MPI.Wtime()
#             test(result)
#             stop_time = MPI.Wtime()
#             time_taken = stop_time - start_time
#
#             # Print results
#             result.printErrors()
#             self.stream.writeln(result.separator2)
#             run = result.testsRun
#             self.stream.writeln("Ran {} test{} in {:.3f}s".format(run, (run != 1) and "s" or "", time_taken))
#             self.stream.writeln()
#
#             expectedFails = unexpectedSuccesses = skipped = 0
#             try:
#                 results = map(len, (result.expectedFailures,
#                                     result.unexpectedSuccesses,
#                                     result.skipped))
#             except AttributeError:
#                 pass
#             else:
#                 expectedFails, unexpectedSuccesses, skipped = results
#
#             self.result = result
#             # Error traces
#             infos = []
#             if not result.wasSuccessful():
#                 self.stream.write("FAILED")
#                 failed, errored = map(len, (result.failures, result.errors))
#                 if failed:
#                     infos.append("failures={0}".format(failed))
#                 if errored:
#                     infos.append("errors={0}".format(errored))
#             else:
#                 self.stream.write("OK")
#
#             if skipped:
#                 infos.append("skipped={0}".format(skipped))
#             if expectedFails:
#                 infos.append("expected failures={0}".format(expectedFails))
#             if unexpectedSuccesses:
#                 infos.append("unexpected successes={0}".format(
#                     unexpectedSuccesses))
#
#             if infos:
#                 self.stream.writeln(" ({0})".format(", ".join(infos)))
#             else:
#                 self.stream.write("\n")
#
#         finally:
#             pass
#
#         return result
#
#
#
