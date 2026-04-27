# =============================================================================
# cmake/Vcpkg.cmake - Vcpkg toolchain detection
# =============================================================================

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(_LOCAL_VCPKG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.deps/vcpkg")

    if(EXISTS "${_LOCAL_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(_AUTO_VCPKG_ROOT "${_LOCAL_VCPKG_ROOT}")
        message(STATUS "[tmnn] Using repository-local vcpkg: ${_AUTO_VCPKG_ROOT}")
    elseif(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(_AUTO_VCPKG_ROOT "$ENV{VCPKG_ROOT}")
        message(STATUS "[tmnn] Using VCPKG_ROOT from environment: ${_AUTO_VCPKG_ROOT}")
    endif()

    if(NOT _AUTO_VCPKG_ROOT)
        message(FATAL_ERROR
            "[tmnn] Vcpkg toolchain not found.\n"
            "Set VCPKG_ROOT to an existing vcpkg checkout, e.g.:\n"
            "  git clone https://github.com/microsoft/vcpkg.git ~/vcpkg\n"
            "  ~/vcpkg/bootstrap-vcpkg.sh\n"
            "  export VCPKG_ROOT=~/vcpkg\n"
            "Alternatively, place a vcpkg checkout at ./.deps/vcpkg.")
    endif()

    set(CMAKE_TOOLCHAIN_FILE "${_AUTO_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file" FORCE)
    message(STATUS "[tmnn] Vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}")
    message(FATAL_ERROR "[tmnn] Toolchain not found: ${CMAKE_TOOLCHAIN_FILE}")
endif()
