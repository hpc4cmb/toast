
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>

#include <thread>
#include <chrono>


TEST_F(TOASTutilsTest, logging) {
    auto & env = toast::Environment::get();

    std::string orig_level = env.log_level();

    std::cout << "Testing level CRITICAL" << std::endl;
    env.set_log_level("CRITICAL");
    auto & log = toast::Logger::get();

    auto here = TOAST_HERE();
    log.critical("This message level is CRITICAL");
    log.critical("This message level is CRITICAL at ", here);
    log.error("This message level is ERROR");
    log.error("This message level is ERROR at ", here);
    log.warning("This message level is WARNING");
    log.warning("This message level is WARNING at ", here);
    log.info("This message level is INFO");
    log.info("This message level is INFO at ", here);
    log.debug("This message level is DEBUG");
    log.debug("This message level is DEBUG at ", here);

    std::cout << "Testing level ERROR" << std::endl;
    env.set_log_level("ERROR");
    log = toast::Logger::get();

    here = TOAST_HERE();
    log.critical("This message level is CRITICAL");
    log.critical("This message level is CRITICAL at ", here);
    log.error("This message level is ERROR");
    log.error("This message level is ERROR at ", here);
    log.warning("This message level is WARNING");
    log.warning("This message level is WARNING at ", here);
    log.info("This message level is INFO");
    log.info("This message level is INFO at ", here);
    log.debug("This message level is DEBUG");
    log.debug("This message level is DEBUG at ", here);

    std::cout << "Testing level WARNING" << std::endl;
    env.set_log_level("WARNING");
    log = toast::Logger::get();

    here = TOAST_HERE();
    log.critical("This message level is CRITICAL");
    log.critical("This message level is CRITICAL at ", here);
    log.error("This message level is ERROR");
    log.error("This message level is ERROR at ", here);
    log.warning("This message level is WARNING");
    log.warning("This message level is WARNING at ", here);
    log.info("This message level is INFO");
    log.info("This message level is INFO at ", here);
    log.debug("This message level is DEBUG");
    log.debug("This message level is DEBUG at ", here);

    std::cout << "Testing level INFO" << std::endl;
    env.set_log_level("INFO");
    log = toast::Logger::get();

    here = TOAST_HERE();
    log.critical("This message level is CRITICAL");
    log.critical("This message level is CRITICAL at ", here);
    log.error("This message level is ERROR");
    log.error("This message level is ERROR at ", here);
    log.warning("This message level is WARNING");
    log.warning("This message level is WARNING at ", here);
    log.info("This message level is INFO");
    log.info("This message level is INFO at ", here);
    log.debug("This message level is DEBUG");
    log.debug("This message level is DEBUG at ", here);

    std::cout << "Testing level DEBUG" << std::endl;
    env.set_log_level("DEBUG");
    log = toast::Logger::get();

    here = TOAST_HERE();
    log.critical("This message level is CRITICAL");
    log.critical("This message level is CRITICAL at ", here);
    log.error("This message level is ERROR");
    log.error("This message level is ERROR at ", here);
    log.warning("This message level is WARNING");
    log.warning("This message level is WARNING at ", here);
    log.info("This message level is INFO");
    log.info("This message level is INFO at ", here);
    log.debug("This message level is DEBUG");
    log.debug("This message level is DEBUG at ", here);

    // Restore original log level
    env.set_log_level(orig_level.c_str());
}

TEST_F(TOASTutilsTest, singletimer) {
    int incr = 200;
    double dincr = (double)incr / 1000.0;
    double prec = 1.0e-2;
    toast::Timer tm;
    EXPECT_FALSE(tm.is_running());
    tm.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(incr));
    tm.stop();
    ASSERT_NEAR(dincr, tm.seconds(), prec);
    tm.report("Test timer stopped");
    tm.clear();
    tm.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(incr));
    try {
        tm.report("This should throw since timer not stopped...");
    } catch (std::runtime_error & e) {
        std::cout << "This should throw since timer not stopped..."
                  << std::endl;
        std::cout << e.what() << std::endl;
    }
    EXPECT_TRUE(tm.is_running());
    tm.stop();
    tm.report("Original");
    double seconds = tm.seconds();
    size_t calls = tm.calls();
    toast::Timer newtm(seconds, calls);
    newtm.report("Copied");
    tm.clear();
    tm.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(incr));
    tm.report_clear("Original was running");
    EXPECT_TRUE(tm.is_running());
    tm.stop();
    tm.report_clear("Original was stopped");
    EXPECT_FALSE(tm.is_running());
}


TEST_F(TOASTutilsTest, globaltimer) {
    int incr = 200;
    double dincr = (double)incr / 1000.0;
    double prec = 1.0e-2;
    auto & gtm = toast::GlobalTimers::get();

    std::vector <std::string> tnames = {
        "timer1",
        "timer2",
        "timer3"
    };

    for (auto const & tname : tnames) {
        try {
            gtm.stop(tname);
        } catch (std::runtime_error & e) {
            std::cout << "This should throw since timer " << tname
                      << " not yet created" << std::endl;
            std::cout << e.what() << std::endl;
        }
    }

    for (auto const & tname : tnames) {
        gtm.start(tname);
    }

    for (auto const & tname : tnames) {
        EXPECT_TRUE(gtm.is_running(tname));
        try {
            gtm.stop(tname);
        } catch (std::runtime_error & e) {
            std::cout << "This should throw since timer " << tname
                      << " still running" << std::endl;
            std::cout << e.what() << std::endl;
        }
    }

    gtm.stop_all();
    gtm.clear_all();
    for (auto const & tname : tnames) {
        gtm.start(tname);
    }

    for (auto const & tname : tnames) {
        std::this_thread::sleep_for(std::chrono::milliseconds(incr));
        gtm.stop(tname);
    }

    size_t offset = 1;
    for (auto const & tname : tnames) {
        ASSERT_NEAR((double)offset * dincr, gtm.seconds(tname), prec);
        offset++;
    }

    gtm.report();
    gtm.clear_all();

    for (auto const & tname : tnames) {
        gtm.start(tname);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(incr));

    gtm.stop_all();
    for (auto const & tname : tnames) {
        ASSERT_NEAR(dincr, gtm.seconds(tname), prec);
    }

    gtm.report();
}
