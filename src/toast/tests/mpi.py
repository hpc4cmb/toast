# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys
import time

import warnings

from unittest.signals import registerResult
from unittest import TestCase
from unittest import TestResult


class MPITestCase(TestCase):
    """A simple wrapper around the standard TestCase which provides
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

    def __init__(self, comm, stream=None, descriptions=None, verbosity=None, **kwargs):
        super(MPITestResult, self).__init__(
            stream=stream, descriptions=descriptions, verbosity=verbosity, **kwargs
        )
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
        if self.comm is not None:
            self.comm.barrier()
        if (self.comm is None) or (self.comm.rank == 0):
            self.stream.write("\n")
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")
            self.stream.flush()
        if self.comm is not None:
            self.comm.barrier()
        super(MPITestResult, self).startTest(test)
        return

    def addSuccess(self, test):
        super(MPITestResult, self).addSuccess(test)
        if self.comm is None:
            self.stream.write("ok ")
        else:
            self.stream.write("[{}]ok ".format(self.comm.rank))
        self.stream.flush()
        return

    def addError(self, test, err):
        super(MPITestResult, self).addError(test, err)
        if self.comm is None:
            self.stream.write("error ")
        else:
            self.stream.write("[{}]error ".format(self.comm.rank))
        self.stream.flush()
        return

    def addFailure(self, test, err):
        super(MPITestResult, self).addFailure(test, err)
        if self.comm is None:
            self.stream.write("fail ")
        else:
            self.stream.write("[{}]fail ".format(self.comm.rank))
        self.stream.flush()
        return

    def addSkip(self, test, reason):
        super(MPITestResult, self).addSkip(test, reason)
        if self.comm is None:
            self.stream.write("skipped({}) ".format(reason))
        else:
            self.stream.write("[{}]skipped({}) ".format(self.comm.rank, reason))
        self.stream.flush()
        return

    def addExpectedFailure(self, test, err):
        super(MPITestResult, self).addExpectedFailure(test, err)
        if self.comm is None:
            self.stream.write("expected-fail ")
        else:
            self.stream.write("[{}]expected-fail ".format(self.comm.rank))
        self.stream.flush()
        return

    def addUnexpectedSuccess(self, test):
        super(MPITestResult, self).addUnexpectedSuccess(test)
        if self.comm is None:
            self.stream.writeln("unexpected-success ")
        else:
            self.stream.writeln("[{}]unexpected-success ".format(self.comm.rank))
        return

    def printErrorList(self, flavour, errors):
        for test, err in errors:
            if self.comm is None:
                self.stream.writeln("{}".format(self.separator1))
                self.stream.writeln("{}: {}".format(flavour, self.getDescription(test)))
                self.stream.writeln("{}".format(self.separator2))
                self.stream.writeln("{}".format(err))
            else:
                self.stream.writeln("[{}] {}".format(self.comm.rank, self.separator1))
                self.stream.writeln(
                    "[{}] {}: {}".format(
                        self.comm.rank, flavour, self.getDescription(test)
                    )
                )
                self.stream.writeln("[{}] {}".format(self.comm.rank, self.separator2))
                self.stream.writeln("[{}] {}".format(self.comm.rank, err))
        return

    def printErrors(self):
        if self.comm is None:
            self.stream.writeln()
            self.stream.flush()
            self.printErrorList("ERROR", self.errors)
            self.printErrorList("FAIL", self.failures)
            self.stream.flush()
        else:
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
        alltotal = None
        if self.comm is None:
            alltotal = total
        else:
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
        return getattr(self.stream, attr)

    def writeln(self, arg=None):
        if arg:
            self.write(arg)
        self.write("\n")  # text-mode streams translate to \r\n if needed


class MPITestRunner(object):
    """A test runner class that displays results in textual form.

    It prints out the names of tests as they are run, errors as they
    occur, and a summary of the results at the end of the test run.

    This information is only printed by the root process.
    """

    resultclass = MPITestResult

    def __init__(
        self, comm, stream=None, descriptions=True, verbosity=2, warnings=None
    ):
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
        result = MPITestResult(
            self.comm, self.stream, self.descriptions, self.verbosity
        )
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
                    warnings.filterwarnings(
                        "module",
                        category=DeprecationWarning,
                        message=r"Please use assert\w+ instead.",
                    )
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
        if self.comm is None:
            self.stream.write("\n")
            self.stream.flush()
        else:
            self.comm.barrier()
            if self.comm.rank == 0:
                self.stream.write("\n")
                self.stream.flush()
        timeTaken = stopTime - startTime

        result.printErrors()

        if (self.comm is None) or (self.comm.rank == 0):
            if hasattr(result, "separator2"):
                self.stream.writeln(result.separator2)

        run = result.testsRun
        if (self.comm is None) or (self.comm.rank == 0):
            self.stream.writeln(
                "Ran %d test%s in %.3fs" % (run, run != 1 and "s" or "", timeTaken)
            )
            self.stream.writeln()

        expectedFails = unexpectedSuccesses = skipped = 0
        try:
            results = map(
                len,
                (result.expectedFailures, result.unexpectedSuccesses, result.skipped),
            )
        except AttributeError:
            pass
        else:
            expectedFails, unexpectedSuccesses, skipped = results

        psize = 1
        prank = 0
        if self.comm is not None:
            psize = self.comm.size
            prank = self.comm.rank

        for p in range(psize):
            if p == prank:
                infos = []
                if not result.wasSuccessful():
                    if self.comm is None:
                        self.stream.write("FAILED")
                    else:
                        self.stream.write("[{}] FAILED".format(self.comm.rank))
                    failed, errored = len(result.failures), len(result.errors)
                    if failed:
                        infos.append("failures=%d" % failed)
                    if errored:
                        infos.append("errors=%d" % errored)
                else:
                    if self.comm is None:
                        self.stream.write("OK")
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
            if self.comm is not None:
                self.comm.barrier()

        # if not result.allSuccessful():
        #     self.comm.Abort(1)

        return result
