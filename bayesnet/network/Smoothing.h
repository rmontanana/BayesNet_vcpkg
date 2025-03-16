// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef SMOOTHING_H
#define SMOOTHING_H
namespace bayesnet {
    enum class Smoothing_t {
        NONE = -1,
        ORIGINAL = 0,
        LAPLACE,
        CESTNIK
    };
}
#endif // SMOOTHING_H