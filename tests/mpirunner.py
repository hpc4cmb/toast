# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from mpi4py import MPI

from unittest import TextTestRunner
from unittest import TestResult, TextTestResult


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


class MPITestResult(TextTestResult):
    """
    A test result class that gathers per-process results and writes
    them to a stream on the root process.

    Used by MPITestRunner.
    """
    def __init__(self, stream=sys.stderr, descriptions=1, verbosity=1):
        TextTestResult.__init__(self, stream, descriptions, verbosity)
        self.buffer = True  # we are capturing test output
        self._stdout_data = None
        self._stderr_data = None
        self.successes = []
        self.callback = None
        self.properties = None  # junit testsuite properties

    def _prepare_callback(self, test_info, target_list, verbose_str,
                          short_str):
        """
        Appends a TestInfo to the given target list and sets a callback
        method to be called by stopTest method.
        """
        target_list.append(test_info)

        def callback():
            """Prints the test method outcome to the stream, as well as
            the elapsed time.
            """

            test_info.test_finished()

            # Ignore the elapsed times for a more reliable unit testing
            if not self.elapsed_times:
                self.start_time = self.stop_time = 0

            if self.showAll:
                self.stream.writeln(
                    '%s (%.3fs)' % (verbose_str, test_info.elapsed_time)
                )
            elif self.dots:
                self.stream.write(short_str)
        self.callback = callback

    def startTest(self, test):
        """
        Called before execute each test method.
        """
        self.start_time = time.time()
        TestResult.startTest(self, test)

        if self.showAll:
            self.stream.write('  ' + self.getDescription(test))
            self.stream.write(" ... ")

    def _save_output_data(self):
        self._stdout_data = sys.stdout.getvalue()
        self._stderr_data = sys.stderr.getvalue()

    def stopTest(self, test):
        """
        Called after execute each test method.
        """
        self._save_output_data()
        # self._stdout_data = sys.stdout.getvalue()
        # self._stderr_data = sys.stderr.getvalue()

        _TextTestResult.stopTest(self, test)
        self.stop_time = time.time()

        if self.callback and callable(self.callback):
            self.callback()
            self.callback = None

    def addSuccess(self, test):
        """
        Called when a test executes successfully.
        """
        self._save_output_data()
        self._prepare_callback(
            _TestInfo(self, test), self.successes, 'OK', '.'
        )

    def addFailure(self, test, err):
        """
        Called when a test method fails.
        """
        self._save_output_data()
        testinfo = _TestInfo(self, test, _TestInfo.FAILURE, err)
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
        testinfo = _TestInfo(self, test, _TestInfo.ERROR, err)
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
            testinfo = _TestInfo(self, testcase, _TestInfo.ERROR, err, subTest=test)
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
        testinfo = _TestInfo(self, test, _TestInfo.SKIP, reason)
        self.skipped.append((testinfo, reason))
        self._prepare_callback(testinfo, [], 'SKIP', 'S')

    def printErrorList(self, flavour, errors):
        """
        Writes information about the FAIL or ERROR to the stream.
        """
        for test_info, error in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln(
                '%s [%.3fs]: %s' % (flavour, test_info.elapsed_time,
                                    test_info.get_description())
            )
            self.stream.writeln(self.separator2)
            self.stream.writeln('%s' % test_info.get_error_info())

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

    def _report_testsuite_properties(xml_testsuite, xml_document, properties):
        xml_properties = xml_document.createElement('properties')
        xml_testsuite.appendChild(xml_properties)
        if properties:
            for key, value in properties.items():
                prop = xml_document.createElement('property')
                prop.setAttribute('name', str(key))
                prop.setAttribute('value', str(value))
                xml_properties.appendChild(prop)
        return xml_properties

    _report_testsuite_properties = staticmethod(_report_testsuite_properties)

    def _report_testsuite(suite_name, tests, xml_document, parentElement,
                          properties):
        """
        Appends the testsuite section to the XML document.
        """
        testsuite = xml_document.createElement('testsuite')
        parentElement.appendChild(testsuite)

        testsuite.setAttribute('name', suite_name)
        testsuite.setAttribute('tests', str(len(tests)))

        testsuite.setAttribute(
            'time', '%.3f' % sum(map(lambda e: e.elapsed_time, tests))
        )
        failures = filter(lambda e: e.outcome == _TestInfo.FAILURE, tests)
        testsuite.setAttribute('failures', str(len(list(failures))))

        errors = filter(lambda e: e.outcome == _TestInfo.ERROR, tests)
        testsuite.setAttribute('errors', str(len(list(errors))))

        _XMLTestResult._report_testsuite_properties(
            testsuite, xml_document, properties)

        systemout = xml_document.createElement('system-out')
        testsuite.appendChild(systemout)

        stdout = StringIO()
        for test in tests:
            # Merge the stdout from the tests in a class
            stdout.write(test.stdout)
        _XMLTestResult._createCDATAsections(
            xml_document, systemout, stdout.getvalue())

        systemerr = xml_document.createElement('system-err')
        testsuite.appendChild(systemerr)

        stderr = StringIO()
        for test in tests:
            # Merge the stderr from the tests in a class
            stderr.write(test.stderr)
        _XMLTestResult._createCDATAsections(
            xml_document, systemerr, stderr.getvalue())

        return testsuite

    _report_testsuite = staticmethod(_report_testsuite)

    def _test_method_name(test_id):
        """
        Returns the test method name.
        """
        return test_id.split('.')[-1]

    _test_method_name = staticmethod(_test_method_name)

    def _createCDATAsections(xmldoc, node, text):
        text = safe_unicode(text)
        pos = text.find(']]>')
        while pos >= 0:
            tmp = text[0:pos+2]
            cdata = xmldoc.createCDATASection(tmp)
            node.appendChild(cdata)
            text = text[pos+2:]
            pos = text.find(']]>')
        cdata = xmldoc.createCDATASection(text)
        node.appendChild(cdata)

    _createCDATAsections = staticmethod(_createCDATAsections)

    def _report_testcase(suite_name, test_result, xml_testsuite, xml_document):
        """
        Appends a testcase section to the XML document.
        """
        testcase = xml_document.createElement('testcase')
        xml_testsuite.appendChild(testcase)

        testcase.setAttribute('classname', suite_name)
        testcase.setAttribute(
            'name', _XMLTestResult._test_method_name(test_result.test_id)
        )
        testcase.setAttribute('time', '%.3f' % test_result.elapsed_time)

        if (test_result.outcome != _TestInfo.SUCCESS):
            elem_name = ('failure', 'error', 'skipped')[test_result.outcome-1]
            failure = xml_document.createElement(elem_name)
            testcase.appendChild(failure)
            if test_result.outcome != _TestInfo.SKIP:
                failure.setAttribute(
                    'type',
                    safe_unicode(test_result.err[0].__name__)
                )
                failure.setAttribute(
                    'message',
                    safe_unicode(test_result.err[1])
                )
                error_info = safe_unicode(test_result.get_error_info())
                _XMLTestResult._createCDATAsections(
                    xml_document, failure, error_info)
            else:
                failure.setAttribute('type', 'skip')
                failure.setAttribute('message', safe_unicode(test_result.err))

    _report_testcase = staticmethod(_report_testcase)

    def generate_reports(self, test_runner):
        """
        Generates the XML reports to a given XMLTestRunner object.
        """
        from xml.dom.minidom import Document
        all_results = self._get_info_by_testcase()

        outputHandledAsString = \
            isinstance(test_runner.output, six.string_types)

        if (outputHandledAsString and not os.path.exists(test_runner.output)):
            os.makedirs(test_runner.output)

        if not outputHandledAsString:
            doc = Document()
            testsuite = doc.createElement('testsuites')
            doc.appendChild(testsuite)
            parentElement = testsuite

        for suite, tests in all_results.items():
            if outputHandledAsString:
                doc = Document()
                parentElement = doc

            suite_name = suite
            if test_runner.outsuffix:
                # not checking with 'is not None', empty means no suffix.
                suite_name = '%s-%s' % (suite, test_runner.outsuffix)

            # Build the XML file
            testsuite = _XMLTestResult._report_testsuite(
                suite_name, tests, doc, parentElement, self.properties
            )
            for test in tests:
                _XMLTestResult._report_testcase(suite, test, testsuite, doc)
            xml_content = doc.toprettyxml(
                indent='\t',
                encoding=test_runner.encoding
            )

            if outputHandledAsString:
                filename = path.join(
                    test_runner.output,
                    'TEST-%s.xml' % suite_name)
                with open(filename, 'wb') as report_file:
                    report_file.write(xml_content)

        if not outputHandledAsString:
            # Assume that test_runner.output is a stream
            test_runner.output.write(xml_content)






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
            TextTestRunner.__init__(self, stream, descriptions, verbosity,
                                failfast=failfast, buffer=buffer)
        else:
            TextTestRunner.__init__(self, NoopStream(), descriptions, verbosity,
                                failfast=failfast, buffer=buffer)
        self.verbosity = verbosity

    def _make_result(self):
        """
        Creates a TestResult object which will be used to store
        information about the executed tests.
        """
        return MPITestResult(
            self.stream, self.descriptions, self.verbosity, self.elapsed_times
        )

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
            start_time = time.time()
            test(result)
            stop_time = time.time()
            time_taken = stop_time - start_time

            # Print results
            result.printErrors()
            self.stream.writeln(result.separator2)
            run = result.testsRun
            self.stream.writeln("Ran %d test%s in %.3fs" % (
                run, run != 1 and "s" or "", time_taken)
            )
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

            # Generate reports
            self.stream.writeln()
            self.stream.writeln('Generating XML reports...')
            result.generate_reports(self)
        finally:
            pass

        return result



