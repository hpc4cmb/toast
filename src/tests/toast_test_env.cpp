
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>


TEST_F(TOASTenvTest, print) {
    auto & env = toast::Environment::get();
    env.print();
}
